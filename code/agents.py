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
        self.board_size = 4 * triangle_size + 1

    def _has_jump_in_progress(self, obs):
        """检查是否有跳跃正在进行中"""
        observation = obs["observation"].reshape(self.board_size, self.board_size, 4)
        return np.any(observation[:, :, 2] == 1)

    def _evaluate_move(self, move, has_jump, num_legal_moves):
        """
        评估单个移动的得分
        基于移动方向和类型来评分
        """
        if move == Move.END_TURN:
            # 跳跃中不应该结束回合（除非只有这一个选项）
            if has_jump and num_legal_moves > 1:
                return -100
            return -5
        
        score = 0.0
        
        # 方向评估 - 核心启发式
        # 向下移动（向目标区域）是好的
        if move.direction == Direction.DownLeft:
            score += 25
        elif move.direction == Direction.DownRight:
            score += 25
        elif move.direction == Direction.Left:
            score += 8
        elif move.direction == Direction.Right:
            score += 8
        # 向上移动（远离目标）是不好的
        elif move.direction == Direction.UpLeft:
            score -= 20
        elif move.direction == Direction.UpRight:
            score -= 20
        
        # 跳跃奖励 - 跳跃可以移动更远
        if move.is_jump:
            score += 35
            # 向下跳跃更好
            if move.direction in [Direction.DownLeft, Direction.DownRight]:
                score += 20

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
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action_idx
                
                # Alpha-Beta剪枝
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
            
            if len(legal_indices) == 1:
                actions.append(legal_indices[0])
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


if __name__ == "__main__":
    pass