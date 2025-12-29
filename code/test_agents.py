"""
交互式测试脚本
可以选择测试不同的算法：MinimaxPolicy, MCTSPolicy, AdaptiveStrategyPolicy, 或自定义RL模型
"""
import os
import argparse
from tqdm import tqdm

import ray
from ray.rllib.policy.policy import Policy

from ChineseChecker import chinese_checker_v0
from agents import GreedyPolicy, MinimaxPolicy, MCTSPolicy, AdaptiveStrategyPolicy, DeepJumpChainPolicy, UltimatePolicy


def load_policy(checkpoint_path, policy_name='default_policy'):
    """加载训练好的RL策略"""
    policy = Policy.from_checkpoint(checkpoint_path)
    policy = policy[policy_name]
    return policy


def evaluate_policy(env, your_policy, baseline_policy, num_trials=20, render_mode=None):
    """评估策略胜率"""
    won = 0
    
    for i in tqdm(range(num_trials)):
        env.reset(seed=i)
        for a in range(len(env.possible_agents)):
            env.action_space(env.possible_agents[a]).seed(i)
        
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    action = your_policy.compute_single_action(obs)
                else:
                    action = baseline_policy.compute_single_action(obs)
            act = int(action[0])
            if render_mode:
                env.render()
            env.step(act)
            
        if env.unwrapped.winner == env.possible_agents[0]:
            won += 1
    
    return won / num_trials


def main():
    parser = argparse.ArgumentParser(description='测试不同算法的胜率')
    parser.add_argument('--triangle_size', type=int, default=2, help='棋盘大小')
    parser.add_argument('--num_trials', type=int, default=20, help='测试局数')
    parser.add_argument('--render_mode', type=str, default=None, help='渲染模式')
    args = parser.parse_args()
    
    # 创建环境
    env = chinese_checker_v0.env(
        render_mode=args.render_mode, 
        triangle_size=args.triangle_size, 
        max_iters=200
    )
    
    # 加载基准策略
    greedy_policy = GreedyPolicy(args.triangle_size)
    rl_baseline_policy = load_policy(
        os.path.join(os.path.dirname(__file__), 'pretrained')
    )
    
    # 可选算法
    algorithms = {
        '1': ('MinimaxPolicy', MinimaxPolicy(args.triangle_size)),
        '2': ('MCTSPolicy', MCTSPolicy(args.triangle_size, num_simulations=50)),
        '3': ('AdaptiveStrategyPolicy', AdaptiveStrategyPolicy(args.triangle_size)),
        '4': ('DeepJumpChainPolicy', DeepJumpChainPolicy(args.triangle_size)),
        '5': ('UltimatePolicy', UltimatePolicy(args.triangle_size)),
    }
    
    while True:
        print("\n" + "="*50)
        print("选择要测试的算法:")
        print("  1. MinimaxPolicy (Alpha-Beta剪枝)")
        print("  2. MCTSPolicy (蒙特卡洛树搜索)")
        print("  3. AdaptiveStrategyPolicy (自适应策略)")
        print("  4. DeepJumpChainPolicy (深度跳跃链策略)")
        print("  5. UltimatePolicy (终极策略)")
        print("  6. 自定义RL模型 (输入checkpoint路径)")
        print("  7. 测试所有算法")
        print("  q. 退出")
        print("="*50)
        
        choice = input("请输入选项: ").strip()
        
        if choice == 'q':
            break
        
        policies_to_test = []
        
        if choice == '1':
            policies_to_test = [algorithms['1']]
        elif choice == '2':
            policies_to_test = [algorithms['2']]
        elif choice == '3':
            policies_to_test = [algorithms['3']]
        elif choice == '4':
            policies_to_test = [algorithms['4']]
        elif choice == '5':
            policies_to_test = [algorithms['5']]
        elif choice == '6':
            checkpoint_path = input("请输入checkpoint路径: ").strip()
            if os.path.exists(checkpoint_path):
                try:
                    rl_policy = load_policy(checkpoint_path)
                    policies_to_test = [('CustomRL', rl_policy)]
                except Exception as e:
                    print(f"加载失败: {e}")
                    continue
            else:
                print("路径不存在!")
                continue
        elif choice == '7':
            policies_to_test = list(algorithms.values())
        else:
            print("无效选项!")
            continue
        
        # 测试选中的算法
        print(f"\n开始测试 (每个算法 {args.num_trials} 局)...\n")
        
        results = []
        for name, policy in policies_to_test:
            print(f"\n{'='*40}")
            print(f"测试: {name}")
            print(f"{'='*40}")
            
            # vs Greedy
            print(f"\nvs Greedy:")
            wr_greedy = evaluate_policy(env, policy, greedy_policy, args.num_trials, args.render_mode)
            print(f"胜率: {wr_greedy*100:.1f}%")
            
            # vs RL Baseline
            print(f"\nvs RL Baseline:")
            wr_rl = evaluate_policy(env, policy, rl_baseline_policy, args.num_trials, args.render_mode)
            print(f"胜率: {wr_rl*100:.1f}%")
            
            results.append((name, wr_greedy, wr_rl))
        
        # 打印汇总
        print("\n" + "="*60)
        print("测试结果汇总")
        print("="*60)
        print(f"{'算法':<25} {'vs Greedy':<15} {'vs RL Baseline':<15}")
        print("-"*60)
        for name, wr_greedy, wr_rl in results:
            print(f"{name:<25} {wr_greedy*100:>6.1f}%{'':<8} {wr_rl*100:>6.1f}%")
        print("="*60)
    
    env.close()
    print("测试结束!")


if __name__ == "__main__":
    main()
