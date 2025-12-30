"""
对抗训练脚本
训练agent对抗Greedy策略，这样可以学习如何击败对手
"""
import datetime
import os
import numpy as np
import argparse
from tqdm import tqdm

from gymnasium.spaces import Box, Discrete, Dict as GymDict
import gymnasium as gym

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

class SingleAgentVsOpponent(gym.Env):
    """单agent环境包装器：agent对抗固定对手（Greedy或RL Baseline）"""
    def __init__(self, triangle_size=2, max_iters=200, opponent_type='greedy', rl_opponent_policy=None):
        super().__init__()
        self.triangle_size = triangle_size
        self.max_iters = max_iters
        self.opponent_type = opponent_type
        
        # 初始化对手
        if opponent_type == 'greedy':
            self.opponent = GreedyPolicy(triangle_size)
        elif opponent_type == 'rl_baseline':
            if rl_opponent_policy is None:
                self.opponent = Policy.from_checkpoint("pretrained/policies/default_policy")
            else:
                self.opponent = rl_opponent_policy
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")
            
        self.env = chinese_checker_v0.env(
            render_mode=None, 
            triangle_size=triangle_size, 
            max_iters=max_iters
        )
        
        # 定义观测和动作空间
        action_space_dim = (4 * triangle_size + 1) ** 2 * 6 * 2 + 1
        observation_space_dim = (4 * triangle_size + 1) ** 2 * 4
        
        # 使用Dict空间包含observation和action_mask
        self.observation_space = GymDict({
            "observation": Box(low=0, high=1, shape=(observation_space_dim,), dtype=np.int8),
            "action_mask": Box(low=0, high=1, shape=(action_space_dim,), dtype=np.int8)
        })
        self.action_space = Discrete(action_space_dim)
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        self.env.reset(seed=seed)
        
        # 如果第一个玩家是对手，让它先走
        if self.env.agent_selection == self.env.possible_agents[1]:
            obs, reward, termination, truncation, info = self.env.last()
            if not (termination or truncation):
                action = self.opponent.compute_single_action(obs)[0]
                self.env.step(int(action))
        
        # 返回学习agent的观测
        obs, reward, termination, truncation, info = self.env.last()
        return obs, info
    
    def step(self, action):
        """执行动作"""
        # 学习agent走一步
        self.env.step(int(action))
        obs, reward, terminated, truncated, info = self.env.last()
        
        done = terminated or truncated
        
        # 如果游戏结束，检查谁赢了
        if done:
            # agent刚走完游戏就结束，检查是否agent赢了
            winner = self.env.unwrapped.winner
            if winner == self.env.possible_agents[0]:
                # Agent是player_0，赢了
                reward = 1000
            else:
                # Agent输了
                reward = -1000
        # 如果游戏没结束，让对手走
        elif self.env.agent_selection == self.env.possible_agents[1]:
            opp_obs = obs
            opp_action = self.opponent.compute_single_action(opp_obs)[0]
            self.env.step(int(opp_action))
            obs, opp_reward, terminated, truncated, info = self.env.last()
            
            done = terminated or truncated
            # 如果对手走完后游戏结束，检查谁赢了
            if done:
                winner = self.env.unwrapped.winner
                if winner == self.env.possible_agents[0]:
                    reward = 1000
                else:
                    reward = -1000
            else:
                reward = 0
        
        return obs, reward, done, done, info


def create_config(env_name: str, triangle_size: int = 4, num_workers: int = 8):
    """创建PPO配置"""
    rlm_class = TorchActionMaskRLM
    model_config = {"fcnet_hiddens": [512, 512, 256]}  # 更大的网络
    rlm_spec = SingleAgentRLModuleSpec(module_class=rlm_class, model_config_dict=model_config)

    action_space_dim = (4 * triangle_size + 1) ** 2 * 6 * 2 + 1
    observation_space_dim = (4 * triangle_size + 1) ** 2 * 4

    # 自动检测GPU
    import torch
    num_gpus = 1 if torch.cuda.is_available() else 0
    if num_gpus > 0:
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}, 将使用GPU训练")
    else:
        print("未检测到GPU, 使用CPU训练")
    print(f"使用 {num_workers} 个并行worker进行环境采样")

    config = (
        PPOConfig()
        .environment(
            env=env_name,
            clip_actions=True,
            env_config={
                "triangle_size": triangle_size,
                "max_iters": 200,
            },
        )
        .rollouts(
            num_rollout_workers=num_workers,  # 增加并行worker
            num_envs_per_worker=2,            # 每个worker运行2个环境
            rollout_fragment_length="auto",
        )
        .training(
            train_batch_size=4096,
            lr=5e-5,                          # 降低学习率，更稳定
            gamma=0.995,
            lambda_=0.95,
            use_gae=True,
            clip_param=0.2,
            grad_clip=0.5,
            vf_loss_coeff=0.5,
            sgd_minibatch_size=512,
            num_sgd_iter=10,
            entropy_coeff=0.01,               # 增加熵，促进探索
            _enable_learner_api=True
        )
        .experimental(_disable_preprocessor_api=True)
        .framework("torch")
        .resources(num_gpus=num_gpus)
        .rl_module(rl_module_spec=rlm_spec)
    )
    return config


def evaluate_vs_greedy(policy, triangle_size, num_trials=20):
    """评估策略对抗Greedy"""
    env = chinese_checker_v0.env(render_mode=None, triangle_size=triangle_size, max_iters=100)
    greedy = GreedyPolicy(triangle_size)
    
    wins = 0
    for i in range(num_trials):
        env.reset(seed=i)
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                break
            
            # 检查observation格式并处理
            if agent == env.possible_agents[0]:
                # RL策略：可能需要处理dict格式
                try:
                    action = policy.compute_single_action(obs)[0]
                except Exception as e:
                    # 如果obs是dict，尝试展平
                    if isinstance(obs, dict) and "observation" in obs:
                        action = policy.compute_single_action(obs["observation"])[0]
                    else:
                        raise e
            else:
                action = greedy.compute_single_action(obs)[0]
            env.step(int(action))
        
        if env.unwrapped.winner == env.possible_agents[0]:
            wins += 1
    
    return wins / num_trials


def evaluate_vs_rl_baseline(policy, rl_baseline, triangle_size, num_trials=20):
    """评估策略对抗RL Baseline"""
    env = chinese_checker_v0.env(render_mode=None, triangle_size=triangle_size, max_iters=100)
    
    wins = 0
    for i in range(num_trials):
        env.reset(seed=i)
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                break
            
            # 检查observation格式并处理
            if agent == env.possible_agents[0]:
                try:
                    action = policy.compute_single_action(obs)[0]
                except Exception as e:
                    if isinstance(obs, dict) and "observation" in obs:
                        action = policy.compute_single_action(obs["observation"])[0]
                    else:
                        raise e
            else:
                try:
                    action = rl_baseline.compute_single_action(obs)[0]
                except Exception as e:
                    if isinstance(obs, dict) and "observation" in obs:
                        action = rl_baseline.compute_single_action(obs["observation"])[0]
                    else:
                        raise e
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
    """主函数 - 两阶段训练"""
    
    # 阶段1环境：对抗Greedy
    def env_creator_greedy(config):
        return SingleAgentVsOpponent(
            triangle_size=config.get("triangle_size", 2),
            max_iters=config.get("max_iters", 200),
            opponent_type='greedy'
        )
    
    # 阶段2环境：对抗RL Baseline
    def env_creator_rl(config):
        return SingleAgentVsOpponent(
            triangle_size=config.get("triangle_size", 2),
            max_iters=config.get("max_iters", 200),
            opponent_type='rl_baseline'
        )

    env_name = 'single_vs_opponent'
    # 先注册Greedy环境
    register_env(env_name, env_creator_greedy)

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)
    
    config = create_config(env_name, args.triangle_size, args.num_workers)
    
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = f"logs/two_stage_{timestr}"
    os.makedirs(logdir, exist_ok=True)
    
    algo = config.build(logger_creator=custom_log_creator(os.path.join(os.curdir, logdir), ''))
    
    # 从checkpoint恢复权重
    if args.restore_from:
        print(f"从checkpoint恢复权重: {args.restore_from}")
        try:
            # 尝试加载policy的权重
            restored_policy = Policy.from_checkpoint(os.path.join(args.restore_from, "policies", "default_policy"))
            current_policy = algo.get_policy("default_policy")
            current_policy.set_weights(restored_policy.get_weights())
            # 同步权重到所有worker
            algo.workers.sync_weights()
            print("成功恢复权重并同步到所有worker!")
        except Exception as e:
            print(f"无法从checkpoint恢复权重: {e}")
            print("将从头开始训练...")
    
    greedy = GreedyPolicy(args.triangle_size)
    
    # 加载RL Baseline用于评估
    rl_baseline = Policy.from_checkpoint("pretrained/policies/default_policy")
    
    best_winrate_greedy = 0.0
    best_winrate_rl = 0.0
    phase = 1  # 1=对抗Greedy, 2=对抗RL Baseline
    phase1_completed = False
    
    print("=" * 60)
    print("阶段1: 对抗Greedy训练 (目标: 90%+)")
    print("=" * 60)
    
    for i in range(args.train_iters):
        # 训练一次迭代
        result = algo.train()
        
        # 获取策略
        policy = algo.get_policy("default_policy")
        
        # 每N次评估一下
        if i % args.eval_period == 0:
            winrate_greedy = evaluate_vs_greedy(policy, args.triangle_size, num_trials=10)
            winrate_rl = evaluate_vs_rl_baseline(policy, rl_baseline, args.triangle_size, num_trials=10)
            
            print(f"[阶段{phase}] Iter {i}: reward={result['episode_reward_mean']:.1f}, "
                  f"vs_Greedy={winrate_greedy*100:.0f}%, vs_RL={winrate_rl*100:.0f}%")
            
            # 保存vs Greedy最好的模型
            if winrate_greedy > best_winrate_greedy:
                best_winrate_greedy = winrate_greedy
                checkpoint_dir = f"{logdir}/best_vs_greedy"
                algo.save(checkpoint_dir=checkpoint_dir)
                print(f"  -> 新最佳vs Greedy: {winrate_greedy*100:.0f}%")
            
            # 保存vs RL最好的模型
            if winrate_rl > best_winrate_rl:
                best_winrate_rl = winrate_rl
                checkpoint_dir = f"{logdir}/best_vs_rl"
                algo.save(checkpoint_dir=checkpoint_dir)
                print(f"  -> 新最佳vs RL: {winrate_rl*100:.0f}%")
            
            # 检查是否达到阶段1目标
            if phase == 1 and winrate_greedy >= 0.90 and not phase1_completed:
                phase1_completed = True
                print("\n" + "=" * 60)
                print(f"🎉 阶段1完成! vs Greedy达到 {winrate_greedy*100:.0f}%")
                print("现在切换到阶段2: 对抗RL Baseline (目标: 90%+)")
                print("=" * 60 + "\n")
                
                # 1. 保存阶段1完成的checkpoint（使用最佳vs_greedy的checkpoint）
                phase1_checkpoint = f"{logdir}/best_vs_greedy"
                if not os.path.exists(phase1_checkpoint):
                    print("警告: 最佳checkpoint不存在，保存当前状态...")
                    phase1_checkpoint = f"{logdir}/phase1_completed"
                    algo.save(checkpoint_dir=phase1_checkpoint)
                else:
                    print(f"使用最佳checkpoint: {phase1_checkpoint}")
                
                # 2. 验证checkpoint文件存在
                policy_checkpoint_path = os.path.join(phase1_checkpoint, "policies", "default_policy")
                if not os.path.exists(policy_checkpoint_path):
                    print(f"错误: Checkpoint路径不存在: {policy_checkpoint_path}")
                    print("继续训练而不切换阶段...")
                    continue
                
                # 3. 先加载权重，再切换环境
                print("正在加载阶段1权重...")
                try:
                    phase1_policy = Policy.from_checkpoint(policy_checkpoint_path)
                    phase1_weights = phase1_policy.get_weights()
                    print(f"成功加载权重，共 {len(phase1_weights)} 个参数")
                except Exception as e:
                    print(f"错误: 无法加载阶段1权重: {e}")
                    print("继续训练而不切换阶段...")
                    continue
                
                # 4. 停止当前算法
                algo.stop()
                
                # 5. 重新注册环境为RL Baseline
                print("重新注册环境为对抗RL Baseline...")
                register_env(env_name, env_creator_rl)
                config = create_config(env_name, args.triangle_size, args.num_workers)
                algo = config.build(logger_creator=custom_log_creator(os.path.join(os.curdir, logdir), ''))
                
                # 6. 设置权重并同步
                print("将权重设置到新算法...")
                try:
                    current_policy = algo.get_policy("default_policy")
                    current_policy.set_weights(phase1_weights)
                    algo.workers.sync_weights()
                    print("✅ 成功加载阶段1权重并同步到所有worker!")
                    
                    # 7. 验证权重是否正确加载（测试vs Greedy）
                    verify_winrate = evaluate_vs_greedy(current_policy, args.triangle_size, num_trials=10)
                    print(f"验证: vs Greedy = {verify_winrate*100:.0f}% (应该接近 {winrate_greedy*100:.0f}%)")
                    if verify_winrate < 0.80:
                        print("⚠️ 警告: 权重可能未正确加载，胜率下降明显!")
                    
                except Exception as e:
                    print(f"错误: 设置权重失败: {e}")
                    print("将从头开始阶段2...")
                
                # 8. 切换阶段
                phase = 2
                print("开始阶段2训练...")
            
            # 检查是否达到最终目标
            # 检查是否两个目标都达成
            if winrate_greedy >= 0.90 and winrate_rl >= 0.90:
                print("\n" + "=" * 60)
                print(f"🎊 训练完成! vs Greedy={winrate_greedy*100:.0f}%, vs RL={winrate_rl*100:.0f}%")
                print("=" * 60)
                break
        
        # 定期保存
        if i % 50 == 0:
            checkpoint_dir = f"{logdir}/checkpoint_{i}"
            algo.save(checkpoint_dir=checkpoint_dir)
    
    # 最终评估
    policy = algo.get_policy("default_policy")
    final_greedy = evaluate_vs_greedy(policy, args.triangle_size, num_trials=50)
    final_rl = evaluate_vs_rl_baseline(policy, rl_baseline, args.triangle_size, num_trials=50)
    print("="*60)
    print(f"训练完成!")
    print(f"最终 vs Greedy: {final_greedy*100:.1f}%")
    print(f"最终 vs RL Baseline: {final_rl*100:.1f}%")
    print(f"最佳 vs RL: {best_rl_winrate*100:.1f}%")
    print(f"模型保存在: {logdir}")
    
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Train vs Greedy')
    parser.add_argument('--train_iters', type=int, default=200)
    parser.add_argument('--triangle_size', type=int, default=2)
    parser.add_argument('--num_cpus', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8, help='并行采样worker数量')
    parser.add_argument('--eval_period', type=int, default=20, help='评估间隔')
    parser.add_argument('--restore_from', type=str, default=None, help='从checkpoint恢复训练')
    parser.add_argument('--local_mode', action='store_true')
    args = parser.parse_args()
    main(args)
