import csv
import datetime
import os
import numpy as np
import glob
import argparse
import copy
from pathlib import Path
from tqdm import tqdm

from gymnasium.spaces import Box, Discrete

from pettingzoo.classic import rps_v2

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import (
    PPO,
    PPOConfig,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.tune.logger import pretty_print
from ray.rllib.policy.policy import Policy

from ChineseChecker.env.game import Direction, Move, Position, ChineseCheckers
from ChineseChecker.env.utils import action_to_move, move_to_action, get_legal_move_mask

# Random Policy 
class ChineseCheckersRandomPolicy(Policy):
    def __init__(self, triangle_size=4, config={}):
        observation_space = Box(low=0, high=1, shape=((4 * triangle_size + 1) * (4 * triangle_size + 1) * 4,), dtype=np.int8)
        action_space = Discrete((4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1)
        super().__init__(observation_space, action_space, config)
        self.action_space = action_space

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        actions = []
        for obs in obs_batch:
            action = self.action_space.sample(obs["action_mask"])
            actions.append(action)
        return actions, [], {}

    def compute_single_action(self, obs, state=None, prev_action=None, prev_reward=None, info=None, episode=None, **kwargs):
        return self.compute_actions([obs], state_batches=[state], prev_action_batch=[prev_action], prev_reward_batch=[prev_reward], info_batch=[info], episodes=[episode], **kwargs)[0]

# TODO: Greedy Policy
class GreedyPolicy(Policy):
    def __init__(self, triangle_size=4, config={}):
        # 观察空间：扁平化的棋盘状态 + 动作掩码
        observation_space = Box(low=0, high=1, shape=((4 * triangle_size + 1) * (4 * triangle_size + 1) * 4,), dtype=np.int8)
        # 动作空间：所有可能的移动 + 结束回合
        action_space = Discrete((4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1)
        super().__init__(observation_space, action_space, config)
        self.triangle_size = triangle_size
        self.action_space_dim = action_space.n
        
    def _action_to_move(self, action: int):
        """将动作索引转换为Move对象（从utils.py复制过来的逻辑）"""
        n = self.triangle_size
        
        if action == (4 * n + 1) ** 2 * 6 * 2:
            return Move.END_TURN
        
        index = action
        index, is_jump = divmod(index, 2)     # 提取是否跳跃
        index, direction = divmod(index, 6)   # 提取方向
        _q, _r = divmod(index, 4 * n + 1)     # 提取坐标索引
        q, r = _q - 2 * n, _r - 2 * n         # 转换为相对坐标
        return Move(q, r, direction, bool(is_jump))
    
    def _move_to_action(self, move: Move):
        """将Move对象转换为动作索引（从utils.py复制过来的逻辑）"""
        n = self.triangle_size
        
        if move == Move.END_TURN:
            return (4 * n + 1) ** 2 * 6 * 2
        
        q, r, direction, is_jump = move.position.q, move.position.r, move.direction, move.is_jump
        index = int(is_jump) + 2 * (direction + 6 * ((r + 2 * n) + (4 * n + 1) * (q + 2 * n)))
        return index
    
    def _calculate_move_score(self, move, player, board_observation):
        """
        计算移动的得分（奖励估计）
        基于环境中的奖励规则
        """
        n = self.triangle_size
        
        if move == Move.END_TURN:
            # 结束回合的得分较低，除非没有其他合法移动
            return 0.0
        
        # 基础得分
        score = 0.0
        # 1. 鼓励向目标区域移动，惩罚远离目标区域
        if move.direction in [Direction.DownLeft, Direction.DownRight]:
            move_distance = 2 if move.is_jump else 1
            score += 1.0 * move_distance
        elif move.direction in [Direction.UpLeft, Direction.UpRight]:
            move_distance = 2 if move.is_jump else 1
            score -= 1.0 * move_distance

        return score
    
    def _get_player_from_observation(self, observation):
        """
        从观察中推断当前玩家
        观察包含4个通道：当前玩家棋子、其他玩家棋子、跳跃起始位置、上次跳跃目标位置
        通过查找哪个通道有棋子来推断
        """
        n = self.triangle_size
        board_size = 4 * n + 1
        
        # 重塑观察为通道形式
        channels = observation["observation"].reshape(board_size, board_size, 4)
        
        # 第一个通道应该是当前玩家的棋子
        # 但为了安全，我们检查哪个通道有最多的棋子
        player0_pieces = np.sum(channels[:, :, 0])  # 通道0：当前玩家棋子
        player3_pieces = np.sum(channels[:, :, 1])  # 通道1：其他玩家棋子
        
        # 如果有棋子，返回对应玩家
        if player0_pieces > 0:
            return 0
        elif player3_pieces > 0:
            return 3
        else:
            # 默认返回0
            return 0
    
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, 
                       prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        """
        计算一批观察的动作
        使用贪心策略：选择得分最高的合法动作
        """
        actions = []
        
        for i, obs in enumerate(obs_batch):
            # 获取动作掩码
            action_mask = obs["action_mask"]
            
            # 推断当前玩家
            player = self._get_player_from_observation(obs)
            
            # 初始化最佳动作和最高得分
            best_action = None
            best_score = -float('inf')
            
            # 遍历所有可能的动作
            for action_idx in range(self.action_space_dim):
                # 检查动作是否合法
                if action_mask[action_idx] == 1:
                    # 转换为Move对象
                    move = self._action_to_move(action_idx)
                    
                    # 计算动作得分
                    score = self._calculate_move_score(move, player, obs)
                    
                    # 更新最佳动作
                    if score > best_score:
                        best_score = score
                        best_action = action_idx
            
            # 如果没有找到合法动作（理论上不会发生），选择第一个合法动作
            if best_action is None:
                # 查找第一个合法动作
                for action_idx in range(self.action_space_dim):
                    if action_mask[action_idx] == 1:
                        best_action = action_idx
                        break
            
            # 如果没有合法动作，选择结束回合
            if best_action is None:
                best_action = self.action_space_dim - 1  # END_TURN动作
            
            actions.append(best_action)
        
        return actions, [], {}
    
    def compute_single_action(self, obs, state=None, prev_action=None, 
                              prev_reward=None, info=None, episode=None, **kwargs):
        """
        计算单个观察的动作
        """
        return self.compute_actions(
            [obs], 
            state_batches=[state], 
            prev_action_batch=[prev_action], 
            prev_reward_batch=[prev_reward], 
            info_batch=[info], 
            episodes=[episode], 
            **kwargs
        )[0]


# Minimax with Alpha-Beta Pruning
class MinimaxPolicy(Policy):
    """
    带Alpha-Beta剪枝的Minimax策略
    使用启发式评估函数，基于移动方向和跳跃来评分
    """
    def __init__(self, triangle_size=4, config={}, max_depth=3):
        observation_space = Box(low=0, high=1, shape=((4 * triangle_size + 1) * (4 * triangle_size + 1) * 4,), dtype=np.int8)
        action_space = Discrete((4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1)
        super().__init__(observation_space, action_space, config)
        self.triangle_size = triangle_size
        self.action_space_dim = action_space.n
        self.n = triangle_size
        self.max_depth = max_depth

    def _has_jump_in_progress(self, obs):
        """检查是否有跳跃正在进行中"""
        n = self.n
        board_size = 4 * n + 1
        observation = obs["observation"].reshape(board_size, board_size, 4)
        return np.any(observation[:, :, 2] == 1)

    def _evaluate_move(self, move, has_jump, num_legal_moves):
        """
        评估单个移动的得分
        基于移动方向和类型来评分
        """
        if move == Move.END_TURN:
            if has_jump and num_legal_moves > 1:
                return -100
            return -5
        
        score = 0.0
        
        # 方向评估 - 核心启发式
        if move.direction == Direction.DownLeft:
            score += 20
        elif move.direction == Direction.DownRight:
            score += 20
        elif move.direction == Direction.Left:
            score += 5
        elif move.direction == Direction.Right:
            score += 5
        elif move.direction == Direction.UpLeft:
            score -= 15
        elif move.direction == Direction.UpRight:
            score -= 15
        
        # 跳跃奖励
        if move.is_jump:
            score += 25
            if move.direction in [Direction.DownLeft, Direction.DownRight]:
                score += 15
        
        return score

    def _minimax(self, legal_indices, obs, depth, alpha, beta, is_maximizing):
        """
        Minimax搜索 + Alpha-Beta剪枝
        """
        has_jump = self._has_jump_in_progress(obs)
        num_legal = len(legal_indices)
        
        if depth == 0 or num_legal == 0:
            return 0, legal_indices[0] if num_legal > 0 else self.action_space_dim - 1
        
        best_action = legal_indices[0]
        
        if is_maximizing:
            max_eval = float('-inf')
            for action_idx in legal_indices:
                move = action_to_move(action_idx, self.n)
                eval_score = self._evaluate_move(move, has_jump, num_legal)
                
                # Alpha-Beta剪枝
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action_idx
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta剪枝
            
            return max_eval, best_action
        else:
            min_eval = float('inf')
            for action_idx in legal_indices:
                move = action_to_move(action_idx, self.n)
                eval_score = self._evaluate_move(move, has_jump, num_legal)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action_idx
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha剪枝
            
            return min_eval, best_action

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None,
                       prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        actions = []
        
        for obs in obs_batch:
            action_mask = obs["action_mask"]
            legal_indices = np.where(action_mask == 1)[0]
            
            if len(legal_indices) == 0:
                actions.append(self.action_space_dim - 1)
                continue
            
            # 使用Minimax with Alpha-Beta剪枝
            _, best_action = self._minimax(
                legal_indices, obs, 
                depth=self.max_depth,
                alpha=float('-inf'),
                beta=float('inf'),
                is_maximizing=True
            )
            
            actions.append(best_action)
        
        return actions, [], {}

    def compute_single_action(self, obs, state=None, prev_action=None,
                              prev_reward=None, info=None, episode=None, **kwargs):
        return self.compute_actions(
            [obs],
            state_batches=[state],
            prev_action_batch=[prev_action],
            prev_reward_batch=[prev_reward],
            info_batch=[info],
            episodes=[episode],
            **kwargs
        )[0]


# =============================================================================
# 新算法: Monte Carlo Tree Search (MCTS) 蒙特卡洛树搜索
# =============================================================================
import random
import math

class MCTSNode:
    """MCTS树节点"""
    def __init__(self, action=None, parent=None):
        self.action = action  # 到达此节点的动作
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.visits = 0
        self.value = 0.0
    
    def ucb1(self, exploration=1.414):
        """UCB1值：平衡探索与利用"""
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def best_child(self, exploration=1.414):
        """选择UCB1最高的子节点"""
        return max(self.children.values(), key=lambda c: c.ucb1(exploration))
    
    def most_visited_child(self):
        """返回访问次数最多的子节点"""
        return max(self.children.values(), key=lambda c: c.visits)


class MCTSPolicy(Policy):
    """
    蒙特卡洛树搜索策略
    通过随机模拟来评估每个走法的价值
    """
    def __init__(self, triangle_size=4, config={}, num_simulations=100, exploration=1.414):
        observation_space = Box(low=0, high=1, shape=((4 * triangle_size + 1) * (4 * triangle_size + 1) * 4,), dtype=np.int8)
        action_space = Discrete((4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1)
        super().__init__(observation_space, action_space, config)
        self.triangle_size = triangle_size
        self.action_space_dim = action_space.n
        self.n = triangle_size
        self.num_simulations = num_simulations
        self.exploration = exploration

    def _has_jump_in_progress(self, obs):
        """检查是否有跳跃正在进行中"""
        n = self.n
        board_size = 4 * n + 1
        observation = obs["observation"].reshape(board_size, board_size, 4)
        return np.any(observation[:, :, 2] == 1)

    def _evaluate_move_heuristic(self, move, has_jump):
        """
        启发式评估单个移动
        用于模拟阶段的快速决策
        """
        if move == Move.END_TURN:
            return -10
        
        score = 0.0
        
        # 方向评估
        direction_scores = {
            Direction.DownLeft: 30,
            Direction.DownRight: 30,
            Direction.Left: 5,
            Direction.Right: 5,
            Direction.UpLeft: -25,
            Direction.UpRight: -25,
        }
        score += direction_scores.get(move.direction, 0)
        
        # 跳跃奖励
        if move.is_jump:
            if has_jump:
                score += 60  # 跳跃链延续
            else:
                score += 40  # 新跳跃
            
            if move.direction in [Direction.DownLeft, Direction.DownRight]:
                score += 35
            elif move.direction in [Direction.Left, Direction.Right]:
                score += 15
        
        return score

    def _simulate_rollout(self, legal_indices, obs, depth=10):
        """
        执行一次rollout模拟
        返回模拟得到的价值估计
        """
        total_score = 0.0
        has_jump = self._has_jump_in_progress(obs)
        
        # 模拟：使用启发式快速选择动作
        for d in range(depth):
            if len(legal_indices) == 0:
                break
            
            # 根据启发式评分选择动作（带随机性）
            scored_moves = []
            for action_idx in legal_indices:
                move = action_to_move(action_idx, self.n)
                score = self._evaluate_move_heuristic(move, has_jump)
                scored_moves.append((score, action_idx, move))
            
            # 使用softmax选择（温度参数控制随机性）
            scores = np.array([s[0] for s in scored_moves])
            scores = scores - np.max(scores)  # 数值稳定性
            exp_scores = np.exp(scores / 10.0)  # 温度=10
            probs = exp_scores / np.sum(exp_scores)
            
            chosen_idx = np.random.choice(len(scored_moves), p=probs)
            chosen_score, chosen_action, chosen_move = scored_moves[chosen_idx]
            
            total_score += chosen_score * (0.9 ** d)  # 折扣因子
            
            # 简化：只模拟一步
            break
        
        return total_score

    def _mcts_search(self, legal_indices, obs):
        """
        执行MCTS搜索
        """
        if len(legal_indices) == 0:
            return self.action_space_dim - 1
        
        if len(legal_indices) == 1:
            return legal_indices[0]
        
        has_jump = self._has_jump_in_progress(obs)
        num_legal = len(legal_indices)
        
        # 创建根节点
        root = MCTSNode()
        root.visits = 1
        
        # 初始化子节点
        for action_idx in legal_indices:
            root.children[action_idx] = MCTSNode(action=action_idx, parent=root)
        
        # MCTS迭代
        for _ in range(self.num_simulations):
            # Selection: 选择要探索的子节点
            node = root.best_child(self.exploration)
            
            # Simulation: 执行rollout
            value = self._simulate_rollout(legal_indices, obs)
            
            # 额外的启发式奖励
            move = action_to_move(node.action, self.n)
            heuristic_bonus = self._evaluate_move_heuristic(move, has_jump)
            value += heuristic_bonus
            
            # Backpropagation: 更新节点统计
            node.visits += 1
            node.value += value
        
        # 选择访问次数最多的动作
        best_node = root.most_visited_child()
        return best_node.action

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None,
                       prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        actions = []
        
        for obs in obs_batch:
            action_mask = obs["action_mask"]
            legal_indices = np.where(action_mask == 1)[0]
            
            if len(legal_indices) == 0:
                actions.append(self.action_space_dim - 1)
                continue
            
            best_action = self._mcts_search(legal_indices, obs)
            actions.append(best_action)
        
        return actions, [], {}

    def compute_single_action(self, obs, state=None, prev_action=None,
                              prev_reward=None, info=None, episode=None, **kwargs):
        return self.compute_actions(
            [obs],
            state_batches=[state],
            prev_action_batch=[prev_action],
            prev_reward_batch=[prev_reward],
            info_batch=[info],
            episodes=[episode],
            **kwargs
        )[0]


# =============================================================================
# 新算法: Adaptive Strategy Policy - 自适应策略
# 根据游戏阶段自动调整策略
# =============================================================================
class AdaptiveStrategyPolicy(Policy):
    """
    自适应策略
    根据游戏阶段（开局、中局、残局）使用不同的策略
    更激进的跳跃链利用
    """
    def __init__(self, triangle_size=4, config={}, aggression=1.0):
        observation_space = Box(low=0, high=1, shape=((4 * triangle_size + 1) * (4 * triangle_size + 1) * 4,), dtype=np.int8)
        action_space = Discrete((4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1)
        super().__init__(observation_space, action_space, config)
        self.triangle_size = triangle_size
        self.action_space_dim = action_space.n
        self.n = triangle_size
        self.aggression = aggression  # 进攻性参数
        
        # 预计算目标区域
        n = triangle_size
        self.target_positions = set()
        for i in range(n):
            for j in range(0, n - i):
                q = -n + j
                r = n + 1 + i
                self.target_positions.add((q, r))
        
        # 起始区域
        self.start_positions = set()
        for i in range(n):
            for j in range(0, n - i):
                q = n - j
                r = -n - 1 - i
                self.start_positions.add((q, r))

    def _has_jump_in_progress(self, obs):
        """检查是否有跳跃正在进行中"""
        n = self.n
        board_size = 4 * n + 1
        observation = obs["observation"].reshape(board_size, board_size, 4)
        return np.any(observation[:, :, 2] == 1)
    
    def _get_jump_source_position(self, obs):
        """获取跳跃源位置（正在跳跃的棋子）"""
        n = self.n
        board_size = 4 * n + 1
        observation = obs["observation"].reshape(board_size, board_size, 4)
        
        for qi in range(board_size):
            for ri in range(board_size):
                if observation[qi, ri, 2] == 1:
                    q = qi - 2 * n
                    r = ri - 2 * n
                    return (q, r)
        return None
    
    def _parse_board(self, obs):
        """解析棋盘状态"""
        n = self.n
        board_size = 4 * n + 1
        observation = obs["observation"].reshape(board_size, board_size, 4)
        
        my_pieces = []
        opp_pieces = []
        
        for qi in range(board_size):
            for ri in range(board_size):
                q = qi - 2 * n
                r = ri - 2 * n
                if observation[qi, ri, 0] == 1:
                    my_pieces.append((q, r))
                if observation[qi, ri, 1] == 1:
                    opp_pieces.append((q, r))
        
        return my_pieces, opp_pieces

    def _get_game_phase(self, my_pieces, opp_pieces):
        """
        判断游戏阶段
        返回: 'opening', 'midgame', 'endgame'
        """
        n = self.n
        total_pieces = n * (n + 1) // 2
        
        # 计算我方进度
        my_in_target = sum(1 for p in my_pieces if p in self.target_positions)
        my_left_start = sum(1 for p in my_pieces if p in self.start_positions)
        
        # 计算平均r坐标（进度指标）
        avg_r = sum(p[1] for p in my_pieces) / len(my_pieces) if my_pieces else 0
        
        if my_left_start >= total_pieces * 0.5:
            return 'opening'
        elif my_in_target >= total_pieces * 0.5:
            return 'endgame'
        else:
            return 'midgame'

    def _evaluate_move_opening(self, move, has_jump, num_legal):
        """开局策略：快速展开，重视跳跃链"""
        if move == Move.END_TURN:
            if has_jump and num_legal > 1:
                return -300
            return -5
        
        score = 0.0
        
        # 开局重视快速前进
        direction_scores = {
            Direction.DownLeft: 40,
            Direction.DownRight: 40,
            Direction.Left: 10,
            Direction.Right: 10,
            Direction.UpLeft: -40,
            Direction.UpRight: -40,
        }
        score += direction_scores.get(move.direction, 0)
        
        # 开局非常重视跳跃
        if move.is_jump:
            if has_jump:
                score += 80
            else:
                score += 60
            
            if move.direction in [Direction.DownLeft, Direction.DownRight]:
                score += 50
            elif move.direction in [Direction.Left, Direction.Right]:
                score += 20
        
        return score * self.aggression

    def _evaluate_move_midgame(self, move, has_jump, num_legal):
        """中局策略：平衡前进和跳跃机会"""
        if move == Move.END_TURN:
            if has_jump and num_legal > 1:
                return -250
            return -3
        
        score = 0.0
        
        direction_scores = {
            Direction.DownLeft: 35,
            Direction.DownRight: 35,
            Direction.Left: 8,
            Direction.Right: 8,
            Direction.UpLeft: -30,
            Direction.UpRight: -30,
        }
        score += direction_scores.get(move.direction, 0)
        
        if move.is_jump:
            if has_jump:
                score += 70
            else:
                score += 55
            
            if move.direction in [Direction.DownLeft, Direction.DownRight]:
                score += 40
            elif move.direction in [Direction.Left, Direction.Right]:
                score += 15
            elif has_jump:
                score += 5
        
        return score

    def _evaluate_move_endgame(self, move, has_jump, num_legal):
        """残局策略：精确进入目标区域"""
        if move == Move.END_TURN:
            if has_jump and num_legal > 1:
                return -200
            return -2
        
        score = 0.0
        
        # 残局更重视精确方向
        direction_scores = {
            Direction.DownLeft: 45,
            Direction.DownRight: 45,
            Direction.Left: 5,
            Direction.Right: 5,
            Direction.UpLeft: -50,
            Direction.UpRight: -50,
        }
        score += direction_scores.get(move.direction, 0)
        
        if move.is_jump:
            if has_jump:
                score += 65
            else:
                score += 50
            
            if move.direction in [Direction.DownLeft, Direction.DownRight]:
                score += 45
        
        return score

    def _search_best_move(self, legal_indices, obs):
        """根据游戏阶段选择最佳移动"""
        has_jump = self._has_jump_in_progress(obs)
        num_legal = len(legal_indices)
        my_pieces, opp_pieces = self._parse_board(obs)
        
        if num_legal == 0:
            return self.action_space_dim - 1
        
        # 判断游戏阶段
        phase = self._get_game_phase(my_pieces, opp_pieces)
        
        # 根据阶段选择评估函数
        if phase == 'opening':
            evaluate_fn = self._evaluate_move_opening
        elif phase == 'endgame':
            evaluate_fn = self._evaluate_move_endgame
        else:
            evaluate_fn = self._evaluate_move_midgame
        
        # 评估所有合法移动
        scored_moves = []
        for action_idx in legal_indices:
            move = action_to_move(action_idx, self.n)
            score = evaluate_fn(move, has_jump, num_legal)
            scored_moves.append((score, action_idx))
        
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return scored_moves[0][1]

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None,
                       prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        actions = []
        
        for obs in obs_batch:
            action_mask = obs["action_mask"]
            legal_indices = np.where(action_mask == 1)[0]
            
            if len(legal_indices) == 0:
                actions.append(self.action_space_dim - 1)
                continue
            
            best_action = self._search_best_move(legal_indices, obs)
            actions.append(best_action)
        
        return actions, [], {}

    def compute_single_action(self, obs, state=None, prev_action=None,
                              prev_reward=None, info=None, episode=None, **kwargs):
        return self.compute_actions(
            [obs],
            state_batches=[state],
            prev_action_batch=[prev_action],
            prev_reward_batch=[prev_reward],
            info_batch=[info],
            episodes=[episode],
            **kwargs
        )[0]


if __name__ == "__main__":
    pass


# =============================================================================
# 新算法: Deep Jump Chain Policy - 深度跳跃链策略
# 使用DFS搜索最优跳跃链
# =============================================================================
class DeepJumpChainPolicy(Policy):
    """
    深度跳跃链策略
    使用深度优先搜索找到最优的跳跃链组合
    特别针对跳跃链进行优化
    """
    def __init__(self, triangle_size=4, config={}):
        observation_space = Box(low=0, high=1, shape=((4 * triangle_size + 1) * (4 * triangle_size + 1) * 4,), dtype=np.int8)
        action_space = Discrete((4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1)
        super().__init__(observation_space, action_space, config)
        self.triangle_size = triangle_size
        self.action_space_dim = action_space.n
        self.n = triangle_size

    def _has_jump_in_progress(self, obs):
        """检查是否有跳跃正在进行中"""
        n = self.n
        board_size = 4 * n + 1
        observation = obs["observation"].reshape(board_size, board_size, 4)
        return np.any(observation[:, :, 2] == 1)
    
    def _count_jump_moves(self, legal_indices):
        """统计跳跃动作数量"""
        jump_count = 0
        for action_idx in legal_indices:
            move = action_to_move(action_idx, self.n)
            if move != Move.END_TURN and move.is_jump:
                jump_count += 1
        return jump_count

    def _evaluate_move(self, move, has_jump, num_legal, jump_count):
        """
        深度评估移动
        特别重视跳跃链的延续
        """
        if move == Move.END_TURN:
            # 如果还有跳跃可以继续，强烈惩罚结束
            if has_jump and jump_count > 0:
                return -1000
            return -1
        
        score = 0.0
        
        # 方向评估 - 非常重视向下
        direction_scores = {
            Direction.DownLeft: 50,
            Direction.DownRight: 50,
            Direction.Left: 10,
            Direction.Right: 10,
            Direction.UpLeft: -45,
            Direction.UpRight: -45,
        }
        score += direction_scores.get(move.direction, 0)
        
        # 跳跃是核心策略
        if move.is_jump:
            # 基础跳跃奖励
            base_jump_bonus = 100 if has_jump else 80
            score += base_jump_bonus
            
            # 方向加成
            if move.direction in [Direction.DownLeft, Direction.DownRight]:
                score += 70  # 向目标跳跃
            elif move.direction in [Direction.Left, Direction.Right]:
                score += 30  # 横向跳跃
            elif has_jump:
                # 跳跃链中向上跳可能是为了更长的链
                score += 15
        else:
            # 非跳跃：如果有跳跃可用，惩罚普通移动
            if not has_jump:
                # 没有跳跃链时，普通向下移动也可以
                if move.direction in [Direction.DownLeft, Direction.DownRight]:
                    score += 25
        
        return score

    def _search_best_move(self, legal_indices, obs):
        """搜索最佳移动"""
        has_jump = self._has_jump_in_progress(obs)
        num_legal = len(legal_indices)
        
        if num_legal == 0:
            return self.action_space_dim - 1
        
        # 统计跳跃动作数量
        jump_count = self._count_jump_moves(legal_indices)
        
        # 评估所有合法移动
        scored_moves = []
        for action_idx in legal_indices:
            move = action_to_move(action_idx, self.n)
            score = self._evaluate_move(move, has_jump, num_legal, jump_count)
            scored_moves.append((score, action_idx))
        
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return scored_moves[0][1]

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None,
                       prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        actions = []
        
        for obs in obs_batch:
            action_mask = obs["action_mask"]
            legal_indices = np.where(action_mask == 1)[0]
            
            if len(legal_indices) == 0:
                actions.append(self.action_space_dim - 1)
                continue
            
            best_action = self._search_best_move(legal_indices, obs)
            actions.append(best_action)
        
        return actions, [], {}

    def compute_single_action(self, obs, state=None, prev_action=None,
                              prev_reward=None, info=None, episode=None, **kwargs):
        return self.compute_actions(
            [obs],
            state_batches=[state],
            prev_action_batch=[prev_action],
            prev_reward_batch=[prev_reward],
            info_batch=[info],
            episodes=[episode],
            **kwargs
        )[0]


# =============================================================================
# 终极算法: UltimatePolicy - 纯方向评估，精细调参
# =============================================================================
class UltimatePolicy(Policy):
    """
    终极策略 - 只使用方向评估（经过验证可靠）
    精细化的跳跃链处理
    """
    def __init__(self, triangle_size=4, config={}):
        observation_space = Box(low=0, high=1, shape=((4 * triangle_size + 1) * (4 * triangle_size + 1) * 4,), dtype=np.int8)
        action_space = Discrete((4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1)
        super().__init__(observation_space, action_space, config)
        self.triangle_size = triangle_size
        self.action_space_dim = action_space.n
        self.n = triangle_size

    def _has_jump_in_progress(self, obs):
        """检查是否有跳跃正在进行中"""
        n = self.n
        board_size = 4 * n + 1
        observation = obs["observation"].reshape(board_size, board_size, 4)
        return np.any(observation[:, :, 2] == 1)
    
    def _analyze_moves(self, legal_indices):
        """分析所有合法移动，分类统计"""
        down_jumps = []      # 向下跳跃
        side_jumps = []      # 横向跳跃
        up_jumps = []        # 向上跳跃
        down_walks = []      # 向下行走
        side_walks = []      # 横向行走
        up_walks = []        # 向上行走
        end_turn = None
        
        for action_idx in legal_indices:
            move = action_to_move(action_idx, self.n)
            if move == Move.END_TURN:
                end_turn = action_idx
            elif move.is_jump:
                if move.direction in [Direction.DownLeft, Direction.DownRight]:
                    down_jumps.append((action_idx, move))
                elif move.direction in [Direction.Left, Direction.Right]:
                    side_jumps.append((action_idx, move))
                else:
                    up_jumps.append((action_idx, move))
            else:
                if move.direction in [Direction.DownLeft, Direction.DownRight]:
                    down_walks.append((action_idx, move))
                elif move.direction in [Direction.Left, Direction.Right]:
                    side_walks.append((action_idx, move))
                else:
                    up_walks.append((action_idx, move))
        
        return {
            'down_jumps': down_jumps,
            'side_jumps': side_jumps,
            'up_jumps': up_jumps,
            'down_walks': down_walks,
            'side_walks': side_walks,
            'up_walks': up_walks,
            'end_turn': end_turn,
            'has_any_jump': len(down_jumps) + len(side_jumps) + len(up_jumps) > 0
        }

    def _search_best_move(self, legal_indices, obs):
        """基于优先级的移动选择"""
        has_jump = self._has_jump_in_progress(obs)
        
        if len(legal_indices) == 0:
            return self.action_space_dim - 1
        
        if len(legal_indices) == 1:
            return legal_indices[0]
        
        moves = self._analyze_moves(legal_indices)
        
        # 跳跃链进行中：优先继续跳跃
        if has_jump:
            # 优先级：向下跳 > 横向跳 > 向上跳 > 结束
            if moves['down_jumps']:
                return moves['down_jumps'][0][0]
            if moves['side_jumps']:
                return moves['side_jumps'][0][0]
            if moves['up_jumps']:
                return moves['up_jumps'][0][0]
            # 没有跳跃可以继续了，结束回合
            if moves['end_turn'] is not None:
                return moves['end_turn']
        
        # 没有跳跃链：选择最佳移动
        # 优先级：向下跳 > 横向跳 > 向下走 > 横向走 > 向上跳 > 向上走
        if moves['down_jumps']:
            return moves['down_jumps'][0][0]
        if moves['side_jumps']:
            return moves['side_jumps'][0][0]
        if moves['down_walks']:
            return moves['down_walks'][0][0]
        if moves['side_walks']:
            return moves['side_walks'][0][0]
        if moves['up_jumps']:
            return moves['up_jumps'][0][0]
        if moves['up_walks']:
            return moves['up_walks'][0][0]
        if moves['end_turn'] is not None:
            return moves['end_turn']
        
        return legal_indices[0]

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None,
                       prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        actions = []
        
        for obs in obs_batch:
            action_mask = obs["action_mask"]
            legal_indices = np.where(action_mask == 1)[0]
            
            if len(legal_indices) == 0:
                actions.append(self.action_space_dim - 1)
                continue
            
            best_action = self._search_best_move(legal_indices, obs)
            actions.append(best_action)
        
        return actions, [], {}

    def compute_single_action(self, obs, state=None, prev_action=None,
                              prev_reward=None, info=None, episode=None, **kwargs):
        return self.compute_actions(
            [obs],
            state_batches=[state],
            prev_action_batch=[prev_action],
            prev_reward_batch=[prev_reward],
            info_batch=[info],
            episodes=[episode],
            **kwargs
        )[0]