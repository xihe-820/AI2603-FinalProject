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
        elif opponent_type == 'random':
            self.opponent = None  # 随机对手
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
        my_agent = self.env.possible_agents[0]  # player_0
        
        # 学习agent走一步
        self.env.step(int(action))
        obs, env_reward, terminated, truncated, info = self.env.last()
        
        done = terminated or truncated
        reward = 0  # 默认没有奖励
        
        # 如果游戏结束，检查谁赢了
        if done:
            winner = self.env.unwrapped.winner
            if winner == my_agent:
                reward = 100  # 赢了
            elif winner is None:
                reward = -10  # 平局/超时
            else:
                reward = -100  # 输了
        else:
            # 让对手走
            if self.env.agent_selection == self.env.possible_agents[1]:
                opp_obs = obs
                # 随机对手：随机选择合法动作
                if self.opponent is None:
                    action_mask = opp_obs["action_mask"]
                    legal_actions = np.where(action_mask == 1)[0]
                    opp_action = np.random.choice(legal_actions) if len(legal_actions) > 0 else 0
                else:
                    opp_action = self.opponent.compute_single_action(opp_obs)[0]
                self.env.step(int(opp_action))
                obs, opp_reward, terminated, truncated, info = self.env.last()
                
                done = terminated or truncated
                if done:
                    winner = self.env.unwrapped.winner
                    if winner == my_agent:
                        reward = 100
                    elif winner is None:
                        reward = -10
                    else:
                        reward = -100
        
        return obs, reward, done, done, info
    
    def _compute_progress_from_obs(self, obs):
        """从observation计算棋子向目标区域的进度（越大越好）"""
        n = self.triangle_size
        board_size = 4 * n + 1
        
        # 解析observation获取棋子位置
        if isinstance(obs, dict):
            observation = obs["observation"].reshape(board_size, board_size, 4)
        else:
            observation = obs.reshape(board_size, board_size, 4)
        
        # 目标区域位置（与agents.py中MinimaxPolicy一致）
        target_positions = set()
        for i in range(n):
            for j in range(0, n - i):
                q = -n + j
                r = n + 1 + i
                target_positions.add((q, r))
        
        total_progress = 0
        # 通道0是当前玩家（player_0）的棋子
        for qi in range(board_size):
            for ri in range(board_size):
                if observation[qi, ri, 0] == 1:  # 我的棋子
                    q = qi - 2 * n
                    r = ri - 2 * n
                    
                    # 计算到目标区域的最小距离的负值（越近越大）
                    if target_positions:
                        min_dist = min(abs(q - t[0]) + abs(r - t[1]) for t in target_positions)
                        total_progress -= min_dist
        
        return total_progress


def create_config(env_name: str, triangle_size: int = 4, num_workers: int = 8, use_large_network: bool = False):
    """创建PPO配置"""
    rlm_class = TorchActionMaskRLM
    # 如果要从pretrained加载，必须用相同的网络结构 [256, 128]
    if use_large_network:
        model_config = {"fcnet_hiddens": [512, 512, 256]}  # 更大的网络
    else:
        model_config = {"fcnet_hiddens": [256, 128]}  # 与pretrained一致
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
            num_envs_per_worker=4,            # 每个worker运行4个环境
            rollout_fragment_length="auto",
        )
        .training(
            train_batch_size=2048,            # 减小batch size加快迭代
            lr=1e-5,                          # 更低学习率，防止破坏预训练权重
            gamma=0.995,
            lambda_=0.95,
            use_gae=True,
            clip_param=0.1,                   # 更小的clip，更保守更新
            grad_clip=0.5,
            vf_loss_coeff=0.5,
            sgd_minibatch_size=256,           # 对应调整minibatch
            num_sgd_iter=3,                   # 更少SGD迭代
            entropy_coeff=0.005,              # 降低熵，更确定性
            _enable_learner_api=True
        )
        .experimental(_disable_preprocessor_api=True)
        .framework("torch")
        .resources(num_gpus=num_gpus)
        .rl_module(rl_module_spec=rlm_spec)
    )
    return config


def evaluate_vs_greedy(policy, triangle_size, num_trials=20, verbose=False):
    """评估策略对抗Greedy"""
    env = chinese_checker_v0.env(render_mode=None, triangle_size=triangle_size, max_iters=100)
    greedy = GreedyPolicy(triangle_size)
    
    wins = 0
    for i in range(num_trials):
        env.reset(seed=i)
        step_count = 0
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                break
            step_count += 1
            
            if agent == env.possible_agents[0]:
                # RL策略：obs已经是dict格式，直接传入
                action = policy.compute_single_action(obs)[0]
            else:
                action = greedy.compute_single_action(obs)[0]
            env.step(int(action))
        
        winner = env.unwrapped.winner
        if verbose and i == 0:
            print(f"  [Debug] Game {i}: steps={step_count}, winner={winner}, possible_agents={env.possible_agents}")
        if winner == env.possible_agents[0]:
            wins += 1
    
    return wins / num_trials


def evaluate_vs_random(policy, triangle_size, num_trials=20):
    """评估策略对抗随机对手"""
    env = chinese_checker_v0.env(render_mode=None, triangle_size=triangle_size, max_iters=100)
    
    wins = 0
    for i in range(num_trials):
        env.reset(seed=i)
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                break
            
            if agent == env.possible_agents[0]:
                # RL策略：obs已经是dict格式，直接传入
                action = policy.compute_single_action(obs)[0]
            else:
                # 随机对手
                action_mask = obs["action_mask"]
                legal_actions = np.where(action_mask == 1)[0]
                action = np.random.choice(legal_actions) if len(legal_actions) > 0 else 0
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
            
            if agent == env.possible_agents[0]:
                # RL策略：obs已经是dict格式，直接传入
                action = policy.compute_single_action(obs)[0]
            else:
                # RL Baseline
                action = rl_baseline.compute_single_action(obs)[0]
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
    """主函数 - 三阶段训练"""
    
    # 阶段0环境：对抗随机（预训练）
    def env_creator_random(config):
        return SingleAgentVsOpponent(
            triangle_size=config.get("triangle_size", 2),
            max_iters=config.get("max_iters", 100),
            opponent_type='random'
        )
    
    # 阶段1环境：对抗Greedy
    def env_creator_greedy(config):
        return SingleAgentVsOpponent(
            triangle_size=config.get("triangle_size", 2),
            max_iters=config.get("max_iters", 100),
            opponent_type='greedy'
        )
    
    # 阶段2环境：对抗RL Baseline
    def env_creator_rl(config):
        return SingleAgentVsOpponent(
            triangle_size=config.get("triangle_size", 2),
            max_iters=config.get("max_iters", 100),
            opponent_type='rl_baseline'
        )

    env_name = 'single_vs_opponent'
    
    # 根据是否从pretrained开始，选择初始环境
    if args.start_from_pretrained:
        # 从pretrained开始，直接注册Greedy环境（阶段1）
        register_env(env_name, env_creator_greedy)
        phase = 1
        phase0_completed = True
    else:
        # 从头开始，注册Random环境（阶段0）
        register_env(env_name, env_creator_random)
        phase = 0
        phase0_completed = False

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)
    
    # 如果从pretrained开始，用小网络；否则用大网络
    use_large = not args.start_from_pretrained
    config = create_config(env_name, args.triangle_size, args.num_workers, use_large_network=use_large)
    
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = f"logs/three_stage_{timestr}"
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
            weights_to_sync = {"default_policy": restored_policy.get_weights()}
            algo.workers.foreach_worker(lambda w: w.set_weights(weights_to_sync))
            print("成功恢复权重并同步到所有worker!")
        except Exception as e:
            print(f"无法从checkpoint恢复权重: {e}")
            print("将从头开始训练...")
    elif args.start_from_pretrained:
        # 从pretrained RL Baseline开始，跳过阶段0
        print("从pretrained RL Baseline开始训练...")
        try:
            restored_policy = Policy.from_checkpoint("pretrained/policies/default_policy")
            restored_weights = restored_policy.get_weights()
            
            # 设置main policy权重
            current_policy = algo.get_policy("default_policy")
            current_policy.set_weights(restored_weights)
            
            # 方法1: 使用local_worker同步
            algo.workers.local_worker().set_weights({"default_policy": restored_weights})
            
            # 方法2: 逐个同步到remote workers
            def set_weights_fn(worker):
                worker.set_weights({"default_policy": restored_weights})
            algo.workers.foreach_worker(set_weights_fn, local_worker=False)
            
            # ★ 关键：同步到Learner模块 ★
            # 新RLlib API中，Learner有独立的权重副本
            if hasattr(algo, 'learner_group') and algo.learner_group is not None:
                # 获取RLModule的state dict格式
                rl_module = current_policy.model
                learner_weights = {"default_policy": rl_module.state_dict()}
                algo.learner_group.set_weights(learner_weights)
                print("已同步权重到Learner模块!")
            
            print("成功从pretrained加载权重!")
            
            # 验证local worker的policy
            print("验证权重同步...")
            local_policy = algo.workers.local_worker().get_policy("default_policy")
            test_winrate = evaluate_vs_random(local_policy, args.triangle_size, num_trials=10)
            print(f"  local worker policy vs Random: {test_winrate*100:.0f}%")
            test_winrate_greedy = evaluate_vs_greedy(local_policy, args.triangle_size, num_trials=10)
            print(f"  local worker policy vs Greedy: {test_winrate_greedy*100:.0f}%")
            
            # 验证remote worker (抽查一个)
            def get_remote_winrate(worker):
                p = worker.get_policy("default_policy")
                # 简单测试：返回权重的一个值来确认是否同步
                w = p.get_weights()
                first_key = list(w.keys())[0]
                return float(w[first_key].flat[0])
            
            remote_check = algo.workers.foreach_worker(get_remote_winrate, local_worker=False)
            local_check = get_remote_winrate(algo.workers.local_worker())
            print(f"  权重一致性检查: local={local_check:.6f}, remote[0]={remote_check[0]:.6f}")
            if abs(local_check - remote_check[0]) < 1e-6:
                print("  ✓ 权重已同步到所有worker!")
            else:
                print("  ✗ 警告: 权重可能未正确同步!")
            
            # 跳过阶段0
            phase0_completed = True
            phase = 1  # 直接进入阶段1
        except Exception as e:
            print(f"无法从pretrained加载权重: {e}")
            import traceback
            traceback.print_exc()
            print("将从阶段0开始训练...")
    
    greedy = GreedyPolicy(args.triangle_size)
    
    # 加载RL Baseline用于评估
    rl_baseline = Policy.from_checkpoint("pretrained/policies/default_policy")
    
    best_winrate_random = 0.0
    best_winrate_greedy = 0.0
    best_winrate_rl = 0.0
    # phase和phase0_completed已在上面根据args.start_from_pretrained设置
    phase1_completed = False
    
    if phase == 0:
        print("=" * 60)
        print("阶段0: 对抗Random预训练 (目标: 90%+)")
        print("=" * 60)
    else:
        print("=" * 60)
        print("阶段1: 对抗Greedy训练 (目标: 90%+) - 从pretrained开始")
        print("=" * 60)
    
    # 保存训练前的权重用于对比
    if args.start_from_pretrained:
        pre_train_weights = algo.get_policy("default_policy").get_weights()
        first_key = list(pre_train_weights.keys())[0]
        pre_train_sample = pre_train_weights[first_key].copy()
    
    for i in range(args.train_iters):
        # 训练一次迭代
        result = algo.train()
        
        # 获取策略
        policy = algo.get_policy("default_policy")
        
        # 第一次迭代：检查权重变化
        if i == 0 and args.start_from_pretrained:
            post_train_weights = policy.get_weights()
            post_train_sample = post_train_weights[first_key]
            weight_diff = np.abs(post_train_sample - pre_train_sample).mean()
            weight_max_diff = np.abs(post_train_sample - pre_train_sample).max()
            print(f"[权重变化诊断] mean_diff={weight_diff:.6f}, max_diff={weight_max_diff:.6f}")
            print(f"  原始权重范围: [{pre_train_sample.min():.4f}, {pre_train_sample.max():.4f}]")
            print(f"  训练后权重范围: [{post_train_sample.min():.4f}, {post_train_sample.max():.4f}]")
        
        # 每N次评估一下
        if i % args.eval_period == 0:
            # 第一次评估加调试信息
            verbose = (i == 0)
            winrate_random = evaluate_vs_random(policy, args.triangle_size, num_trials=10)
            winrate_greedy = evaluate_vs_greedy(policy, args.triangle_size, num_trials=10, verbose=verbose)
            winrate_rl = evaluate_vs_rl_baseline(policy, rl_baseline, args.triangle_size, num_trials=10)
            
            print(f"[阶段{phase}] Iter {i}: reward={result['episode_reward_mean']:.1f}, "
                  f"vs_Random={winrate_random*100:.0f}%, vs_Greedy={winrate_greedy*100:.0f}%, vs_RL={winrate_rl*100:.0f}%")
            
            # 保存vs Random最好的模型
            if winrate_random > best_winrate_random:
                best_winrate_random = winrate_random
            
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
            
            # 检查是否达到阶段0目标（vs Random 90%+）
            if phase == 0 and winrate_random >= 0.90 and not phase0_completed:
                phase0_completed = True
                print("\n" + "=" * 60)
                print(f"🎉 阶段0完成! vs Random达到 {winrate_random*100:.0f}%")
                print("现在切换到阶段1: 对抗Greedy (目标: 90%+)")
                print("=" * 60 + "\n")
                
                # 保存checkpoint到文件（更可靠）
                phase0_checkpoint = f"{logdir}/phase0_completed"
                algo.save(checkpoint_dir=phase0_checkpoint)
                print(f"已保存阶段0 checkpoint到: {phase0_checkpoint}")
                
                # 停止当前算法
                algo.stop()
                
                # 重新注册环境为Greedy
                register_env(env_name, env_creator_greedy)
                config = create_config(env_name, args.triangle_size, args.num_workers, use_large_network=use_large)
                algo = config.build(logger_creator=custom_log_creator(os.path.join(os.curdir, logdir), ''))
                
                # 从checkpoint恢复权重
                policy_path = os.path.join(phase0_checkpoint, "policies", "default_policy")
                print(f"从checkpoint加载权重: {policy_path}")
                restored_policy = Policy.from_checkpoint(policy_path)
                restored_weights = restored_policy.get_weights()
                
                current_policy = algo.get_policy("default_policy")
                current_policy.set_weights(restored_weights)
                
                # 同步到所有worker
                algo.workers.local_worker().set_weights({"default_policy": restored_weights})
                def set_weights_fn(worker):
                    worker.set_weights({"default_policy": restored_weights})
                algo.workers.foreach_worker(set_weights_fn, local_worker=False)
                
                # 同步到Learner模块（关键！）
                learner_group = algo.learner_group
                def update_learner_weights(learner):
                    learner._module["default_policy"].load_state_dict(
                        current_policy.model.state_dict()
                    )
                learner_group.foreach_learner(update_learner_weights)
                
                # 验证权重是否正确加载
                verify_winrate = evaluate_vs_random(current_policy, args.triangle_size, num_trials=10)
                print(f"验证: vs Random = {verify_winrate*100:.0f}% (应该接近 {winrate_random*100:.0f}%)")
                
                if verify_winrate < 0.80:
                    print("⚠️ 警告: 权重可能未正确加载!")
                else:
                    print("✅ 成功切换到阶段1!")
                
                phase = 1
            
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
                config = create_config(env_name, args.triangle_size, args.num_workers, use_large_network=use_large)
                algo = config.build(logger_creator=custom_log_creator(os.path.join(os.curdir, logdir), ''))
                
                # 6. 设置权重并同步
                print("将权重设置到新算法...")
                try:
                    current_policy = algo.get_policy("default_policy")
                    current_policy.set_weights(phase1_weights)
                    
                    # 同步到所有worker
                    algo.workers.local_worker().set_weights({"default_policy": phase1_weights})
                    def set_weights_fn(worker):
                        worker.set_weights({"default_policy": phase1_weights})
                    algo.workers.foreach_worker(set_weights_fn, local_worker=False)
                    
                    # 同步到Learner模块（关键！）
                    learner_group = algo.learner_group
                    def update_learner_weights(learner):
                        learner._module["default_policy"].load_state_dict(
                            current_policy.model.state_dict()
                        )
                    learner_group.foreach_learner(update_learner_weights)
                    
                    print("✅ 成功加载阶段1权重并同步到所有worker和Learner!")
                    
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
    print(f"最佳 vs RL: {best_winrate_rl*100:.1f}%")
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
    parser.add_argument('--start_from_pretrained', action='store_true', help='从pretrained RL Baseline开始，跳过阶段0')
    parser.add_argument('--local_mode', action='store_true')
    args = parser.parse_args()
    main(args)
