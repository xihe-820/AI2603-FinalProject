"""
对抗外部对手训练 - 使用 multi-policy 配置
一个策略可训练，另一个策略是固定的外部对手
"""
import datetime
import os
import numpy as np
import argparse

from gymnasium.spaces import Box, Discrete

import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.policy.policy import Policy, PolicySpec

from ChineseChecker import chinese_checker_v0
from ChineseChecker.models.action_masking_rlm import TorchActionMaskRLM
from agents import GreedyPolicy


class GreedyPolicyWrapper(Policy):
    """包装 GreedyPolicy 为 RLlib Policy"""
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        triangle_size = config.get("triangle_size", 2)
        self.greedy = GreedyPolicy(triangle_size)
    
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None,
                       prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        actions = []
        for obs in obs_batch:
            action = self.greedy.compute_single_action(obs)[0]
            actions.append(action)
        return actions, [], {}
    
    def get_weights(self):
        return {}
    
    def set_weights(self, weights):
        pass


class RLBaselinePolicyWrapper(Policy):
    """包装 RL Baseline 为 RLlib Policy"""
    _baseline_policy = None  # 类变量，所有实例共享
    
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        if RLBaselinePolicyWrapper._baseline_policy is None:
            checkpoint_path = os.path.join(os.path.dirname(__file__), 'pretrained')
            loaded = Policy.from_checkpoint(checkpoint_path)
            RLBaselinePolicyWrapper._baseline_policy = loaded['default_policy']
        self.baseline = RLBaselinePolicyWrapper._baseline_policy
    
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None,
                       prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        actions = []
        for obs in obs_batch:
            action = self.baseline.compute_single_action(obs)[0]
            actions.append(action)
        return actions, [], {}
    
    def get_weights(self):
        return {}
    
    def set_weights(self, weights):
        pass


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """策略映射：player_0 用可训练策略，player_3 用对手策略"""
    if agent_id == "player_0":
        return "learned_policy"
    else:
        return "opponent_policy"


def create_config(env_name: str, triangle_size: int = 2, num_workers: int = 8, 
                  opponent_type: str = "greedy"):
    """创建 multi-policy PPO 配置"""
    
    action_space_dim = (4 * triangle_size + 1) ** 2 * 6 * 2 + 1
    observation_space_dim = (4 * triangle_size + 1) ** 2 * 4
    
    # 选择对手策略类
    if opponent_type == "rl_baseline":
        opponent_class = RLBaselinePolicyWrapper
    else:
        opponent_class = GreedyPolicyWrapper
    
    import torch
    num_gpus = 1 if torch.cuda.is_available() else 0
    
    # 为可训练策略配置 RLModule
    rlm_class = TorchActionMaskRLM
    model_config = {"fcnet_hiddens": [256, 256, 128]}
    rlm_spec = SingleAgentRLModuleSpec(module_class=rlm_class, model_config_dict=model_config)

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
            lr=3e-4,
            gamma=0.995,
            lambda_=0.95,
            use_gae=True,
            clip_param=0.2,
            grad_clip=0.5,
            vf_loss_coeff=0.5,
            sgd_minibatch_size=512,
            num_sgd_iter=10,
            entropy_coeff=0.005,
            _enable_learner_api=True
        )
        .multi_agent(
            policies={
                "learned_policy": PolicySpec(),  # 使用默认配置，可训练
                "opponent_policy": PolicySpec(
                    policy_class=opponent_class,
                    config={"triangle_size": triangle_size}
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["learned_policy"],  # 只训练 learned_policy
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

    # 阶段1：先对抗 Greedy
    print("="*60)
    print("阶段1: 对抗 Greedy 训练")
    print("="*60)
    
    config_greedy = create_config(env_name, args.triangle_size, args.num_workers, "greedy")
    algo = config_greedy.build()

    # 加载评估用的对手
    rl_baseline = Policy.from_checkpoint(os.path.join(os.path.dirname(__file__), 'pretrained'))
    rl_baseline = rl_baseline['default_policy']
    greedy = GreedyPolicy(args.triangle_size)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = f"logs/adversarial_{timestamp}"
    os.makedirs(logdir, exist_ok=True)

    best_winrate_rl = 0.0
    phase = 1
    greedy_mastered = False
    
    for i in range(args.train_iters):
        result = algo.train()
        
        if i % args.eval_period == 0:
            policy = algo.get_policy("learned_policy")
            
            wr_greedy = evaluate_vs_opponent(policy, greedy, args.triangle_size, num_trials=20)
            wr_rl = evaluate_vs_opponent(policy, rl_baseline, args.triangle_size, num_trials=20)
            
            reward_mean = result.get('episode_reward_mean', float('nan'))
            print(f"[Phase {phase}] Iter {i}: reward={reward_mean:.1f}, vs_Greedy={wr_greedy*100:.0f}%, vs_RL={wr_rl*100:.0f}%")
            
            # 保存最好的模型
            if wr_rl > best_winrate_rl:
                best_winrate_rl = wr_rl
                algo.save(checkpoint_dir=f"{logdir}/best_checkpoint")
                print(f"  -> 新最佳! vs_RL={wr_rl*100:.0f}%")
            
            # 切换到阶段2：当 Greedy 胜率 >= 80%
            if not greedy_mastered and wr_greedy >= 0.8 and phase == 1:
                greedy_mastered = True
                print("\n" + "="*60)
                print("阶段2: 切换到对抗 RL Baseline 训练")
                print("="*60 + "\n")
                phase = 2
                
                # 保存当前权重
                learned_weights = algo.get_policy("learned_policy").get_weights()
                
                # 重建算法，使用 RL Baseline 作为对手
                algo.stop()
                config_rl = create_config(env_name, args.triangle_size, args.num_workers, "rl_baseline")
                algo = config_rl.build()
                
                # 恢复权重
                algo.get_policy("learned_policy").set_weights(learned_weights)
            
            # 早停
            if wr_rl >= args.target_winrate:
                print(f"\n达到目标胜率 {args.target_winrate*100:.0f}%!")
                break
        
        if i % 50 == 0 and i > 0:
            algo.save(checkpoint_dir=f"{logdir}/checkpoint_{i}")

    # 最终评估
    policy = algo.get_policy("learned_policy")
    final_wr_rl = evaluate_vs_opponent(policy, rl_baseline, args.triangle_size, num_trials=50)
    final_wr_greedy = evaluate_vs_opponent(policy, greedy, args.triangle_size, num_trials=50)
    
    print("="*60)
    print(f"训练完成!")
    print(f"最终: vs_Greedy={final_wr_greedy*100:.1f}%, vs_RL={final_wr_rl*100:.1f}%")
    print(f"最佳 vs_RL: {best_winrate_rl*100:.1f}%")
    print(f"模型: {logdir}")
    
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Adversarial Training')
    parser.add_argument('--train_iters', type=int, default=500)
    parser.add_argument('--triangle_size', type=int, default=2)
    parser.add_argument('--num_cpus', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_period', type=int, default=10)
    parser.add_argument('--target_winrate', type=float, default=0.7)
    parser.add_argument('--local_mode', action='store_true')
    args = parser.parse_args()
    main(args)
