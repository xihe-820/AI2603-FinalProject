"""
评估最终训练模型的性能
对抗Greedy和RL Baseline进行多次对局
"""
import os
import argparse
from tqdm import tqdm

from ray.rllib.policy.policy import Policy

from ChineseChecker import chinese_checker_v0
from agents import GreedyPolicy


def evaluate_vs_greedy(policy, triangle_size, num_trials=100):
    """评估策略对抗Greedy"""
    env = chinese_checker_v0.env(render_mode=None, triangle_size=triangle_size, max_iters=100)
    greedy = GreedyPolicy(triangle_size)
    
    wins = 0
    for i in tqdm(range(num_trials), desc="vs Greedy"):
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


def evaluate_vs_rl_baseline(policy, rl_baseline, triangle_size, num_trials=100):
    """评估策略对抗RL Baseline"""
    env = chinese_checker_v0.env(render_mode=None, triangle_size=triangle_size, max_iters=100)
    
    wins = 0
    for i in tqdm(range(num_trials), desc="vs RL Baseline"):
        env.reset(seed=i)
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                break
            
            if agent == env.possible_agents[0]:
                action = policy.compute_single_action(obs)[0]
            else:
                action = rl_baseline.compute_single_action(obs)[0]
            env.step(int(action))
        
        if env.unwrapped.winner == env.possible_agents[0]:
            wins += 1
    
    return wins / num_trials


def evaluate_vs_random(policy, triangle_size, num_trials=100):
    """评估策略对抗Random"""
    env = chinese_checker_v0.env(render_mode=None, triangle_size=triangle_size, max_iters=100)
    import random
    
    wins = 0
    for i in tqdm(range(num_trials), desc="vs Random"):
        env.reset(seed=i)
        random.seed(i)
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                break
            
            if agent == env.possible_agents[0]:
                action = policy.compute_single_action(obs)[0]
            else:
                # Random选择合法动作
                action_mask = obs["action_mask"]
                valid_actions = [j for j in range(len(action_mask)) if action_mask[j] == 1]
                action = random.choice(valid_actions) if valid_actions else 0
            env.step(int(action))
        
        if env.unwrapped.winner == env.possible_agents[0]:
            wins += 1
    
    return wins / num_trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, 
                       default="logs/three_stage_2025-12-31_19-49-51/best_vs_rl",
                       help="训练好的模型checkpoint路径")
    parser.add_argument("--triangle_size", type=int, default=2)
    parser.add_argument("--num_trials", type=int, default=100, help="评估对局数")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"评估最终训练模型: {args.checkpoint}")
    print(f"对局数: {args.num_trials}")
    print("=" * 60)
    
    # 加载训练好的模型
    policy_path = os.path.join(args.checkpoint, "policies", "default_policy")
    if not os.path.exists(policy_path):
        policy_path = args.checkpoint  # 可能直接是policy路径
    
    print(f"\n加载模型: {policy_path}")
    trained_policy = Policy.from_checkpoint(policy_path)
    
    # 加载RL Baseline
    print("加载RL Baseline: pretrained/policies/default_policy")
    rl_baseline = Policy.from_checkpoint("pretrained/policies/default_policy")
    
    # 评估
    print("\n" + "=" * 60)
    print("开始评估...")
    print("=" * 60)
    
    # vs Random
    winrate_random = evaluate_vs_random(trained_policy, args.triangle_size, args.num_trials)
    print(f"\n✓ vs Random: {winrate_random*100:.1f}%")
    
    # vs Greedy
    winrate_greedy = evaluate_vs_greedy(trained_policy, args.triangle_size, args.num_trials)
    print(f"✓ vs Greedy: {winrate_greedy*100:.1f}%")
    
    # vs RL Baseline
    winrate_rl = evaluate_vs_rl_baseline(trained_policy, rl_baseline, args.triangle_size, args.num_trials)
    print(f"✓ vs RL Baseline: {winrate_rl*100:.1f}%")
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 最终评估结果")
    print("=" * 60)
    print(f"  vs Random:      {winrate_random*100:.1f}%")
    print(f"  vs Greedy:      {winrate_greedy*100:.1f}%")
    print(f"  vs RL Baseline: {winrate_rl*100:.1f}%")
    print("=" * 60)
    
    # 判断是否达标
    if winrate_greedy >= 0.90 and winrate_rl >= 0.90:
        print("🎊 恭喜！模型达到目标 (90%+ vs Greedy & RL Baseline)")
    elif winrate_greedy >= 0.90:
        print(f"✓ vs Greedy达标，vs RL还需提升 {(0.90 - winrate_rl)*100:.1f}%")
    else:
        print(f"需要继续训练...")


if __name__ == "__main__":
    main()
