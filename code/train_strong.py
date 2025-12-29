"""
强化版训练脚本
使用更多迭代、更大批量、自我对弈等技术训练更强的RL模型
"""
import datetime
import os
import numpy as np
import glob
import argparse

from gymnasium.spaces import Box, Discrete

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import (
    PPO,
    PPOConfig,
)
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec

from ChineseChecker import chinese_checker_v0
from ChineseChecker.models.action_masking_rlm import TorchActionMaskRLM
from ChineseChecker.logger import custom_log_creator
from rllib_marl import train


def create_strong_config(env_name: str, triangle_size: int = 4, entropy_coeff: float = 0.01):
    """
    创建强化版PPO训练配置
    """
    rlm_class = TorchActionMaskRLM

    # 更大的网络
    model_config = {
        "fcnet_hiddens": [512, 512, 256, 128]  # 更深更宽的网络
    }

    rlm_spec = SingleAgentRLModuleSpec(
        module_class=rlm_class, 
        model_config_dict=model_config
    )

    action_space_dim = (4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1
    observation_space_dim = (4 * triangle_size + 1) * (4 * triangle_size + 1) * 4

    config = (
        PPOConfig()
        .environment(
            env=env_name, 
            clip_actions=True,
            env_config={
                "triangle_size": triangle_size,
                "action_space": Discrete(action_space_dim),
                "max_iters": 300,  # 更长的游戏
                "render_mode": None,
                "observation_space": Box(low=0, high=1, shape=(observation_space_dim,), dtype=np.int8),
            },
        )
        .training(
            # 强化训练参数
            train_batch_size=4096,  # 更大的批量
            lr=1e-4,  # 较低学习率，更稳定
            gamma=0.998,  # 更高折扣因子，重视长期
            lambda_=0.97,
            use_gae=True,
            clip_param=0.15,  # 更小的clip，更稳定
            grad_clip=0.5,
            vf_loss_coeff=0.5,
            sgd_minibatch_size=512,
            num_sgd_iter=20,  # 更多SGD迭代
            entropy_coeff=entropy_coeff,  # 更高熵鼓励探索
            _enable_learner_api=True
        )
        .experimental(
            _disable_preprocessor_api=True,
        )
        .framework("torch")
        .resources(
            num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0"))
        )
        .rl_module(rl_module_spec=rlm_spec)
    )
    return config


def main(args):
    """主函数"""
    def env_creator(config):
        return chinese_checker_v0.env(**config)

    env_name = 'chinese_checker_v0'
    model_name = f'strong_model_{args.entropy_coeff}'
    register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)
    
    config = create_strong_config(env_name, args.triangle_size, args.entropy_coeff)
    
    train_config = {
        "triangle_size": args.triangle_size,
        "train_iters": args.train_iters,
        "entropy_coeff": args.entropy_coeff,
        "eval_period": args.eval_period,
        "eval_num_trials": args.eval_num_trials,
        "eval_max_iters": args.eval_max_iters,
        "render_mode": args.render_mode,
    }
    
    train(config, model_name, train_config)
    print("Strong model training finished!")
    
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Strong RL Training Script')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('--train_iters', type=int, default=500)  # 更多迭代
    parser.add_argument('--triangle_size', type=int, required=True)
    parser.add_argument('--eval_period', type=int, default=10)
    parser.add_argument('--eval_num_trials', type=int, default=20)
    parser.add_argument('--eval_max_iters', type=int, default=400)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)  # 更高熵
    parser.add_argument('--render_mode', type=str, default=None)
    parser.add_argument('--num_cpus', type=int, default=4)
    parser.add_argument('--local_mode', action='store_true')
    args = parser.parse_args()
    main(args)
