"""
对抗RL Baseline训练
从已训练的模型继续训练，对抗更强的对手
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


def create_config(env_name: str, triangle_size: int = 4, num_workers: int = 8):
    """创建PPO配置"""
    rlm_class = TorchActionMaskRLM
    model_config = {"fcnet_hiddens": [256, 256, 128]}
    rlm_spec = SingleAgentRLModuleSpec(module_class=rlm_class, model_config_dict=model_config)

    action_space_dim = (4 * triangle_size + 1) ** 2 * 6 * 2 + 1
    observation_space_dim = (4 * triangle_size + 1) ** 2 * 4

    import torch
    num_gpus = 1 if torch.cuda.is_available() else 0
    if num_gpus > 0:
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
    print(f"使用 {num_workers} 个并行worker")

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
        .rollouts(
            num_rollout_workers=num_workers,
            num_envs_per_worker=2,
            rollout_fragment_length="auto",
        )
        .training(
            train_batch_size=4096,
            lr=1e-4,  # 降低学习率，细调
            gamma=0.995,
            lambda_=0.95,
            use_gae=True,
            clip_param=0.1,  # 降低clip，更保守更新
            grad_clip=0.5,
            vf_loss_coeff=0.5,
            sgd_minibatch_size=512,
            num_sgd_iter=10,
            entropy_coeff=0.01,  # 增加探索
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
    def env_creator(config):
        return chinese_checker_v0.env(**config)

    env_name = 'chinese_checker_v0'
    register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    config = create_config(env_name, args.triangle_size, args.num_workers)

    # 创建算法
    algo = config.build()
    
    # 如果有预训练模型，加载权重
    if args.checkpoint:
        print(f"从 {args.checkpoint} 加载预训练模型...")
        # 加载预训练的policy权重
        pretrained_policy = Policy.from_checkpoint(args.checkpoint)
        pretrained_policy = pretrained_policy['default_policy']
        weights = pretrained_policy.get_weights()
        
        # 设置到算法的所有组件
        algo.set_weights({"default_policy": weights})
        
        # 验证加载
        print("验证模型加载...")
        greedy = GreedyPolicy(args.triangle_size)
        policy = algo.get_policy("default_policy")
        test_wr = evaluate_vs_opponent(policy, greedy, args.triangle_size, num_trials=10)
        print(f"加载后 vs Greedy 胜率: {test_wr*100:.0f}%")

    # 加载RL Baseline作为对手
    rl_baseline = Policy.from_checkpoint(os.path.join(os.path.dirname(__file__), 'pretrained'))
    rl_baseline = rl_baseline['default_policy']
    
    # Greedy用于额外评估
    greedy = GreedyPolicy(args.triangle_size)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = f"logs/vs_rl_baseline_{timestamp}"
    os.makedirs(logdir, exist_ok=True)

    best_winrate_rl = 0.0
    best_winrate_greedy = 0.0
    
    print("="*60)
    print("开始对抗 RL Baseline 训练")
    print("="*60)

    for i in range(args.train_iters):
        result = algo.train()
        
        if i % args.eval_period == 0:
            policy = algo.get_policy("default_policy")
            
            # 评估 vs RL Baseline
            wr_rl = evaluate_vs_opponent(policy, rl_baseline, args.triangle_size, num_trials=20)
            # 评估 vs Greedy
            wr_greedy = evaluate_vs_opponent(policy, greedy, args.triangle_size, num_trials=20)
            
            reward_mean = result.get('episode_reward_mean', float('nan'))
            print(f"Iter {i}: reward={reward_mean:.1f}, vs_Greedy={wr_greedy*100:.0f}%, vs_RL={wr_rl*100:.0f}%")
            
            # 保存最好的模型（基于RL Baseline胜率）
            if wr_rl > best_winrate_rl:
                best_winrate_rl = wr_rl
                best_winrate_greedy = wr_greedy
                checkpoint_dir = f"{logdir}/best_checkpoint"
                algo.save(checkpoint_dir=checkpoint_dir)
                print(f"  -> 新最佳! vs_RL={wr_rl*100:.0f}%")
            
            # 早停：达到目标胜率
            if wr_rl >= args.target_winrate:
                print(f"\n达到目标胜率 {args.target_winrate*100:.0f}%，停止训练")
                break
        
        # 定期保存
        if i % 50 == 0:
            algo.save(checkpoint_dir=f"{logdir}/checkpoint_{i}")

    # 最终评估
    policy = algo.get_policy("default_policy")
    final_wr_rl = evaluate_vs_opponent(policy, rl_baseline, args.triangle_size, num_trials=50)
    final_wr_greedy = evaluate_vs_opponent(policy, greedy, args.triangle_size, num_trials=50)
    
    print("="*60)
    print(f"训练完成!")
    print(f"最终: vs_Greedy={final_wr_greedy*100:.1f}%, vs_RL={final_wr_rl*100:.1f}%")
    print(f"最佳: vs_Greedy={best_winrate_greedy*100:.1f}%, vs_RL={best_winrate_rl*100:.1f}%")
    print(f"模型保存在: {logdir}")
    
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Train vs RL Baseline')
    parser.add_argument('--train_iters', type=int, default=300)
    parser.add_argument('--triangle_size', type=int, default=2)
    parser.add_argument('--num_cpus', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_period', type=int, default=10)
    parser.add_argument('--target_winrate', type=float, default=0.7, help='目标胜率，达到后停止')
    parser.add_argument('--checkpoint', type=str, default=None, help='预训练模型路径')
    parser.add_argument('--local_mode', action='store_true')
    args = parser.parse_args()
    main(args)
