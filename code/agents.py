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

# 方向到坐标增量的映射（全局常量，避免重复定义）
DIRECTION_DELTAS = {
    Direction.Right: (1, 0),
    Direction.UpRight: (1, -1),
    Direction.UpLeft: (0, -1),
    Direction.Left: (-1, 0),
    Direction.DownLeft: (-1, 1),
    Direction.DownRight: (0, 1),
}

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


# =============================================================================
# 强化算法: EnhancedPolicy - 综合位置与跳跃链评估
# =============================================================================
class EnhancedPolicy(Policy):
    """
    增强策略 - 综合多种因素进行评估
    
    核心思想：
    1. 利用observation中的棋子位置信息
    2. 评估每个移动对整体进度的贡献
    3. 特别优化跳跃链的利用
    4. 考虑棋子的分布（避免棋子堆积）
    """
    def __init__(self, triangle_size=4, config={}):
        observation_space = Box(low=0, high=1, shape=((4 * triangle_size + 1) * (4 * triangle_size + 1) * 4,), dtype=np.int8)
        action_space = Discrete((4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1)
        super().__init__(observation_space, action_space, config)
        self.triangle_size = triangle_size
        self.action_space_dim = action_space.n
        self.n = triangle_size
        self.board_size = 4 * self.n + 1

    def _parse_board(self, obs):
        """解析棋盘状态"""
        observation = obs["observation"].reshape(self.board_size, self.board_size, 4)
        
        my_pieces = []
        opp_pieces = []
        has_jump_in_progress = False
        jump_source = None
        
        for q_idx in range(self.board_size):
            for r_idx in range(self.board_size):
                q = q_idx - 2 * self.n
                r = r_idx - 2 * self.n
                
                if observation[q_idx, r_idx, 0] == 1:
                    my_pieces.append((q, r))
                if observation[q_idx, r_idx, 1] == 1:
                    opp_pieces.append((q, r))
                if observation[q_idx, r_idx, 2] == 1:
                    has_jump_in_progress = True
                    jump_source = (q, r)
        
        return my_pieces, opp_pieces, has_jump_in_progress, jump_source

    def _get_r_delta(self, direction, is_jump):
        """获取r坐标变化量"""
        delta = {
            Direction.DownLeft: 1,
            Direction.DownRight: 1,
            Direction.Left: 0,
            Direction.Right: 0,
            Direction.UpLeft: -1,
            Direction.UpRight: -1,
        }
        base = delta.get(direction, 0)
        return base * 2 if is_jump else base

    def _count_potential_jumps(self, from_q, from_r, all_pieces, direction_taken=None):
        """
        计算从某个位置可能的跳跃数量
        用于评估跳跃链的潜力
        """
        all_occupied = set(all_pieces)
        potential = 0
        best_delta_r = -10
        
        for direction in Direction:
            dq, dr = DIRECTION_DELTAS[direction]
            mid_q, mid_r = from_q + dq, from_r + dr
            land_q, land_r = from_q + 2*dq, from_r + 2*dr
            
            # 检查是否可以跳跃
            if (mid_q, mid_r) in all_occupied:
                if (land_q, land_r) not in all_occupied:
                    # 检查落点在棋盘内
                    if abs(land_q) <= 2*self.n and abs(land_r) <= 2*self.n:
                        potential += 1
                        if dr > best_delta_r:
                            best_delta_r = dr
        
        return potential, best_delta_r

    def _evaluate_move(self, move, my_pieces, opp_pieces, has_jump, jump_count):
        """综合评估移动"""
        if move == Move.END_TURN:
            # 如果在跳跃链中且还有跳跃可继续，严重惩罚
            if has_jump and jump_count > 0:
                return -5000
            return -1
        
        score = 0.0
        all_pieces = my_pieces + opp_pieces
        
        # 1. 基础方向分数
        r_delta = self._get_r_delta(move.direction, move.is_jump)
        score += r_delta * 100  # 进度最重要
        
        # 2. 跳跃奖励
        if move.is_jump:
            score += 80  # 基础跳跃奖励
            
            # 计算跳跃后的位置
            q, r = move.position.q, move.position.r
            dq, dr = DIRECTION_DELTAS[move.direction]
            new_q, new_r = q + 2*dq, r + 2*dr
            
            # 评估跳跃后的继续潜力
            # 更新棋子集合
            updated_pieces = [p for p in all_pieces if p != (q, r)]
            updated_pieces.append((new_q, new_r))
            
            potential, best_dr = self._count_potential_jumps(new_q, new_r, updated_pieces)
            score += potential * 40  # 每个潜在跳跃加分
            if best_dr > 0:
                score += 50  # 如果有向下跳跃的可能
            
            # 跳跃链延续奖励
            if has_jump:
                score += 60
        else:
            # 普通移动
            if move.direction in [Direction.DownLeft, Direction.DownRight]:
                score += 40
            elif move.direction in [Direction.Left, Direction.Right]:
                score += 10
        
        return score

    def _search_best_move(self, legal_indices, obs):
        """搜索最佳移动"""
        if len(legal_indices) == 0:
            return self.action_space_dim - 1
        
        if len(legal_indices) == 1:
            return legal_indices[0]
        
        my_pieces, opp_pieces, has_jump, jump_source = self._parse_board(obs)
        
        # 统计可用跳跃数
        jump_count = 0
        for idx in legal_indices:
            m = action_to_move(idx, self.n)
            if m != Move.END_TURN and m.is_jump:
                jump_count += 1
        
        scored_moves = []
        for action_idx in legal_indices:
            move = action_to_move(action_idx, self.n)
            score = self._evaluate_move(move, my_pieces, opp_pieces, has_jump, jump_count)
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