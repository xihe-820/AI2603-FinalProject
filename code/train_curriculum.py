"""
课程学习训练脚本
先对抗Greedy，达到一定胜率后逐渐增加RL Baseline对手比例
使用自定义环境wrapper实现对抗外部对手训练
"""
import datetime
import os
import numpy as np
import argparse
import random
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict

import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.policy.policy import Policy

from ChineseChecker import chinese_checker_v0
from ChineseChecker.models.action_masking_rlm import TorchActionMaskRLM
from agents import GreedyPolicy


# 全局对手策略（在worker中共享）
_greedy_policy = None
_rl_baseline_policy = None
_rl_opponent_ratio = 0.0  # RL对手的比例


def get_opponent_policies(triangle_size):
    """懒加载对手策略"""
    global _greedy_policy, _rl_baseline_policy
    if _greedy_policy is None:
        _greedy_policy = GreedyPolicy(triangle_size)
    if _rl_baseline_policy is None:
        _rl_baseline_policy = Policy.from_checkpoint(
            os.path.join(os.path.dirname(__file__), 'pretrained')
        )['default_policy']
    return _greedy_policy, _rl_baseline_policy


class SingleAgentEnvWrapper(gym.Env):
    """
    将PettingZoo双人游戏包装成单agent环境
    对手使用外部策略（Greedy或RL Baseline）
    """
    def __init__(self, config):
        self.triangle_size = config.get("triangle_size", 2)
        self.max_iters = config.get("max_iters", 200)
        self.rl_ratio = config.get("rl_opponent_ratio", 0.0)
        
        # 创建底层环境
        self.env = chinese_checker_v0.env(
            render_mode=None, 
            triangle_size=self.triangle_size, 
            max_iters=self.max_iters
        )
        
        # 获取对手策略
        self.greedy, self.rl_baseline = get_opponent_policies(self.triangle_size)
        
        # 空间定义
        action_space_dim = (4 * self.triangle_size + 1) ** 2 * 6 * 2 + 1
        observation_space_dim = (4 * self.triangle_size + 1) ** 2 * 4
        
        self.action_space = Discrete(action_space_dim)
        self.observation_space = Dict({
            "observation": Box(low=0, high=1, shape=(observation_space_dim,), dtype=np.int8),
            "action_mask": Box(low=0, high=1, shape=(action_space_dim,), dtype=np.int8),
        })
        
        self.my_agent = None
        self.opponent_agent = None
        
    def _select_opponent(self):
        """根据比例选择对手"""
        global _rl_opponent_ratio
        if random.random() < _rl_opponent_ratio:
            return self.rl_baseline
        return self.greedy
    
    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed)
        self.my_agent = self.env.possible_agents[0]
        self.opponent_agent = self.env.possible_agents[1]
        self.opponent = self._select_opponent()
        
        # 获取初始观察
        obs, _, _, _, _ = self.env.last()
        return obs, {}
    
    def step(self, action):
        # 我方行动
        self.env.step(action)
        
        # 检查游戏是否结束
        obs, reward, termination, truncation, info = self.env.last()
        if termination or truncation:
            return obs, reward, termination, truncation, info
        
        # 对手回合（可能多步，因为跳跃链）
        while self.env.agent_selection == self.opponent_agent:
            opp_obs, opp_reward, opp_term, opp_trunc, opp_info = self.env.last()
            if opp_term or opp_trunc:
                break
            opp_action = self.opponent.compute_single_action(opp_obs)[0]
            self.env.step(int(opp_action))
        
        # 返回我方下一个状态
        obs, reward, termination, truncation, info = self.env.last()
        return obs, reward, termination, truncation, info
    
    def close(self):
        self.env.close()


def create_config(env_name: str, triangle_size: int = 4, num_workers: int = 8):
    """创建PPO配置"""
    rlm_class = TorchActionMaskRLM
    model_config = {"fcnet_hiddens": [256, 256, 128]}
    rlm_spec = SingleAgentRLModuleSpec(module_class=rlm_class, model_config_dict=model_config)

    action_space_dim = (4 * triangle_size + 1) ** 2 * 6 * 2 + 1
    observation_space_dim = (4 * triangle_size + 1) ** 2 * 4

    import torch
    num_gpus = 1 if torch.cuda.is_available() else 0

    config = (
        PPOConfig()
        .environment(
            env=env_name,
            clip_actions=True,
            env_config={
                "triangle_size": triangle_size,
                "max_iters": 200,
                "rl_opponent_ratio": 0.0,  # 初始全部对抗Greedy
            },
        )
        .rollouts(
            num_rollout_workers=num_workers,
            num_envs_per_worker=2,
            rollout_fragment_length="auto",
        )
        .training(
            train_batch_size=4096,
            lr=3e-4,
            gamma=0.995,
            lambda_=0.95,
            use_gae=True,
            clip_param=0.2,
            grad_clip=0.5,
            vf_loss_coeff=0.5,
            sgd_minibatch_size=512,
            num_sgd_iter=10,
            entropy_coeff=0.01,
            _enable_learner_api=True
        )
        .experimental(_disable_preprocessor_api=True)
        .framework("torch")
        .resources(num_gpus=num_gpus)
        .rl_module(rl_module_spec=rlm_spec)
    )
    return config


def evaluate_vs_opponent(policy, opponent_policy, triangle_size, num_trials=20):
    """评估策略"""
    env = chinese_checker_v0.env(render_mode=None, triangle_size=triangle_size, max_iters=200)
    
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
                action = opponent_policy.compute_single_action(obs)[0]
            env.step(int(action))
        
        if env.unwrapped.winner == env.possible_agents[0]:
            wins += 1
    
    return wins / num_trials


def main(args):
    global _rl_opponent_ratio
    
    # 注册自定义环境
    env_name = 'chinese_checker_curriculum'
    register_env(env_name, lambda config: SingleAgentEnvWrapper(config))

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    config = create_config(env_name, args.triangle_size, args.num_workers)
    algo = config.build()

    # 加载对手用于评估
    rl_baseline = Policy.from_checkpoint(os.path.join(os.path.dirname(__file__), 'pretrained'))
    rl_baseline = rl_baseline['default_policy']
    greedy = GreedyPolicy(args.triangle_size)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = f"logs/curriculum_{timestamp}"
    os.makedirs(logdir, exist_ok=True)

    best_winrate_rl = 0.0
    _rl_opponent_ratio = 0.0  # 开始时100%对抗Greedy
    
    print("="*60)
    print("课程学习: Greedy -> RL Baseline")
    print("训练时真正对抗外部对手")
    print("="*60)

    for i in range(args.train_iters):
        result = algo.train()
        
        if i % args.eval_period == 0:
            policy = algo.get_policy("default_policy")
            
            wr_greedy = evaluate_vs_opponent(policy, greedy, args.triangle_size, num_trials=20)
            wr_rl = evaluate_vs_opponent(policy, rl_baseline, args.triangle_size, num_trials=20)
            
            reward_mean = result.get('episode_reward_mean', float('nan'))
            print(f"Iter {i}: reward={reward_mean:.1f}, vs_Greedy={wr_greedy*100:.0f}%, vs_RL={wr_rl*100:.0f}%, RL_ratio={_rl_opponent_ratio*100:.0f}%")
            
            # 课程学习：当Greedy胜率>=80%时，增加RL对手比例
            if wr_greedy >= 0.8 and _rl_opponent_ratio < 1.0:
                _rl_opponent_ratio = min(1.0, _rl_opponent_ratio + 0.2)
                print(f"  -> 提升难度! RL对手比例: {_rl_opponent_ratio*100:.0f}%")
            
            # 保存最好的模型
            if wr_rl > best_winrate_rl:
                best_winrate_rl = wr_rl
                algo.save(checkpoint_dir=f"{logdir}/best_checkpoint")
                print(f"  -> 新最佳! vs_RL={wr_rl*100:.0f}%")
            
            # 早停
            if wr_rl >= args.target_winrate:
                print(f"\n达到目标胜率 {args.target_winrate*100:.0f}%!")
                break
        
        if i % 50 == 0:
            algo.save(checkpoint_dir=f"{logdir}/checkpoint_{i}")

    # 最终评估
    policy = algo.get_policy("default_policy")
    final_wr_rl = evaluate_vs_opponent(policy, rl_baseline, args.triangle_size, num_trials=50)
    final_wr_greedy = evaluate_vs_opponent(policy, greedy, args.triangle_size, num_trials=50)
    
    print("="*60)
    print(f"训练完成!")
    print(f"最终: vs_Greedy={final_wr_greedy*100:.1f}%, vs_RL={final_wr_rl*100:.1f}%")
    print(f"最佳 vs_RL: {best_winrate_rl*100:.1f}%")
    print(f"模型: {logdir}")
    
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Curriculum Training')
    parser.add_argument('--train_iters', type=int, default=500)
    parser.add_argument('--triangle_size', type=int, default=2)
    parser.add_argument('--num_cpus', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_period', type=int, default=10)
    parser.add_argument('--target_winrate', type=float, default=0.7)
    parser.add_argument('--local_mode', action='store_true')
    args = parser.parse_args()
    main(args)
