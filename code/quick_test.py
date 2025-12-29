"""
快速测试脚本 - 测试所有算法对战Greedy和RL Baseline
"""
import os
import argparse
from tqdm import tqdm

from ray.rllib.policy.policy import Policy

from ChineseChecker import chinese_checker_v0
from agents import GreedyPolicy, MinimaxPolicy, AdaptiveStrategyPolicy, EnhancedPolicy


def load_policy(checkpoint_path, policy_name='default_policy'):
    policy = Policy.from_checkpoint(checkpoint_path)
    policy = policy[policy_name]
    return policy


def evaluate_policy(env, your_policy, baseline_policy, num_trials=100):
    """评估策略胜率"""
    won = 0
    
    for i in tqdm(range(num_trials), desc="Testing"):
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
            env.step(act)
            
        if env.unwrapped.winner == env.possible_agents[0]:
            won += 1
    
    return won / num_trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--triangle_size', type=int, default=2)
    parser.add_argument('--num_trials', type=int, default=100)
    args = parser.parse_args()
    
    env = chinese_checker_v0.env(render_mode=None, triangle_size=args.triangle_size, max_iters=200)
    
    # 基准策略
    greedy_policy = GreedyPolicy(args.triangle_size)
    rl_baseline_policy = load_policy(os.path.join(os.path.dirname(__file__), 'pretrained'))
    
    # 测试算法
    algorithms = [
        ('MinimaxPolicy', MinimaxPolicy(args.triangle_size)),
        ('AdaptiveStrategyPolicy', AdaptiveStrategyPolicy(args.triangle_size)),
        ('EnhancedPolicy', EnhancedPolicy(args.triangle_size)),
    ]
    
    print(f"\n测试 {args.num_trials} 局...\n")
    print("="*70)
    print(f"{'算法':<25} {'vs Greedy':<20} {'vs RL Baseline':<20}")
    print("="*70)
    
    for name, policy in algorithms:
        print(f"\n测试 {name}...")
        
        wr_greedy = evaluate_policy(env, policy, greedy_policy, args.num_trials)
        wr_rl = evaluate_policy(env, policy, rl_baseline_policy, args.num_trials)
        
        print(f"{name:<25} {wr_greedy*100:>6.1f}%{'':<13} {wr_rl*100:>6.1f}%")
    
    print("="*70)
    env.close()


if __name__ == "__main__":
    main()
