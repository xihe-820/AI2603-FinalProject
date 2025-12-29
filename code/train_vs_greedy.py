"""
对抗训练脚本
训练agent对抗Greedy策略，这样可以学习如何击败对手
"""
import datetime
import os
import numpy as np
import argparse
from tqdm import tqdm

from gymnasium.spaces import Box, Discrete

import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.policy.policy import Policy

from ChineseChecker import chinese_checker_v0
from ChineseChecker.models.action_masking_rlm import TorchActionMaskRLM
from ChineseChecker.logger import custom_log_creator
from agents import GreedyPolicy


def create_config(env_name: str, triangle_size: int = 4):
    """创建PPO配置"""
    rlm_class = TorchActionMaskRLM
    model_config = {"fcnet_hiddens": [256, 256, 128]}
    rlm_spec = SingleAgentRLModuleSpec(module_class=rlm_class, model_config_dict=model_config)

    action_space_dim = (4 * triangle_size + 1) ** 2 * 6 * 2 + 1
    observation_space_dim = (4 * triangle_size + 1) ** 2 * 4

    config = (
        PPOConfig()
        .environment(
            env=env_name,
            clip_actions=True,
            env_config={
                "triangle_size": triangle_size,
                "action_space": Discrete(action_space_dim),
                "max_iters": 200,
                "render_mode": None,
                "observation_space": Box(low=0, high=1, shape=(observation_space_dim,), dtype=np.int8),
            },
        )
        .training(
            train_batch_size=2048,
            lr=3e-4,
            gamma=0.995,
            lambda_=0.95,
            use_gae=True,
            clip_param=0.2,
            grad_clip=0.5,
            vf_loss_coeff=0.5,
            sgd_minibatch_size=256,
            num_sgd_iter=10,
            entropy_coeff=0.005,
            _enable_learner_api=True
        )
        .experimental(_disable_preprocessor_api=True)
        .framework("torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .rl_module(rl_module_spec=rlm_spec)
    )
    return config


def evaluate_vs_greedy(policy, triangle_size, num_trials=20):
    """评估策略对抗Greedy"""
    env = chinese_checker_v0.env(render_mode=None, triangle_size=triangle_size, max_iters=200)
    greedy = GreedyPolicy(triangle_size)
    
    wins = 0
    for i in range(num_trials):
        env.reset(seed=i)
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                break
            if agent == env.possible_agents[0]:
                action = policy.compute_single_action(obs)[0]
            else:
                action = greedy.compute_single_action(obs)[0]
            env.step(int(action))
        
        if env.unwrapped.winner == env.possible_agents[0]:
            wins += 1
    
    return wins / num_trials


def train_vs_greedy_env(policy, greedy_policy, env, num_episodes=100):
    """
    让RL策略与Greedy对弈收集经验
    返回transitions用于训练
    """
    transitions = []
    
    for ep in range(num_episodes):
        env.reset(seed=ep)
        episode_data = []
        
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                break
            
            if agent == env.possible_agents[0]:
                # RL agent
                action = policy.compute_single_action(obs)[0]
                episode_data.append({
                    'obs': obs,
                    'action': action,
                    'reward': reward
                })
            else:
                # Greedy opponent
                action = greedy_policy.compute_single_action(obs)[0]
            
            env.step(int(action))
        
        # 添加最终奖励
        if episode_data:
            final_reward = 1000 if env.unwrapped.winner == env.possible_agents[0] else -500
            episode_data[-1]['reward'] += final_reward
        
        transitions.extend(episode_data)
    
    return transitions


def main(args):
    """主函数"""
    def env_creator(config):
        return chinese_checker_v0.env(**config)

    env_name = 'chinese_checker_v0'
    register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)
    
    config = create_config(env_name, args.triangle_size)
    
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = f"logs/vs_greedy_{timestr}"
    os.makedirs(logdir, exist_ok=True)
    
    algo = config.build(logger_creator=custom_log_creator(os.path.join(os.curdir, logdir), ''))
    
    greedy = GreedyPolicy(args.triangle_size)
    best_winrate = 0.0
    
    print("开始对抗Greedy训练...")
    print("="*60)
    
    for i in range(args.train_iters):
        # 训练一次迭代
        result = algo.train()
        
        # 获取策略
        policy = algo.get_policy("default_policy")
        
        # 每10次评估一下
        if i % 10 == 0:
            winrate = evaluate_vs_greedy(policy, args.triangle_size, num_trials=20)
            print(f"Iteration {i}: reward_mean={result['episode_reward_mean']:.1f}, vs_Greedy={winrate*100:.1f}%")
            
            # 保存最好的模型
            if winrate > best_winrate:
                best_winrate = winrate
                checkpoint_dir = f"{logdir}/best_checkpoint"
                algo.save(checkpoint_dir=checkpoint_dir)
                print(f"  -> 新最佳模型! 胜率: {winrate*100:.1f}%")
        
        # 定期保存
        if i % 50 == 0:
            checkpoint_dir = f"{logdir}/checkpoint_{i}"
            algo.save(checkpoint_dir=checkpoint_dir)
    
    # 最终评估
    policy = algo.get_policy("default_policy")
    final_winrate = evaluate_vs_greedy(policy, args.triangle_size, num_trials=50)
    print("="*60)
    print(f"训练完成! 最终 vs Greedy 胜率: {final_winrate*100:.1f}%")
    print(f"最佳胜率: {best_winrate*100:.1f}%")
    print(f"模型保存在: {logdir}")
    
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Train vs Greedy')
    parser.add_argument('--train_iters', type=int, default=200)
    parser.add_argument('--triangle_size', type=int, default=2)
    parser.add_argument('--num_cpus', type=int, default=4)
    parser.add_argument('--local_mode', action='store_true')
    args = parser.parse_args()
    main(args)
