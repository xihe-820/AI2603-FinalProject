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
    def __init__(self, triangle_size=4, config={}, max_depth=3):
        observation_space = Box(low=0, high=1, shape=((4 * triangle_size + 1) * (4 * triangle_size + 1) * 4,), dtype=np.int8)
        action_space = Discrete((4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1)
        super().__init__(observation_space, action_space, config)
        self.triangle_size = triangle_size
        self.action_space_dim = action_space.n
        self.n = triangle_size
        
        # 预计算目标区域坐标 (从当前玩家视角，目标在下方)
        n = triangle_size
        self.target_coords = []
        offset = np.array([-n, n + 1, -1])
        for i in range(n):
            for j in range(0, n - i):
                coord = offset + np.array([j, i, -i - j])
                self.target_coords.append((coord[0], coord[1]))
        self.target_set = set(self.target_coords)
        
        # 预计算起始区域坐标
        self.home_coords = []
        offset = np.array([1, -n - 1, n])
        for i in range(n):
            for j in range(i, n):
                coord = offset + np.array([j, -i, i - j])
                self.home_coords.append((coord[0], coord[1]))
        self.home_set = set(self.home_coords)

    def _obs_to_pieces(self, obs):
        n = self.n
        board_size = 4 * n + 1
        observation = obs["observation"].reshape(board_size, board_size, 4)
        
        # 通道0是我的棋子，通道1是对手棋子（从当前玩家视角）
        my_pieces = []
        opp_pieces = []
        
        for q in range(board_size):
            for r in range(board_size):
                if observation[q, r, 0] == 1:
                    my_pieces.append((q - 2*n, r - 2*n))
                if observation[q, r, 1] == 1:
                    opp_pieces.append((q - 2*n, r - 2*n))
        
        has_jump = np.any(observation[:, :, 2] == 1)
        
        return my_pieces, opp_pieces, has_jump

    def _evaluate_state(self, my_pieces, opp_pieces):
        n = self.n
        total_pieces = n * (n + 1) // 2
        score = 0.0
        
        in_target = 0
        for piece in my_pieces:
            if piece in self.target_set:
                in_target += 1
                score += 100
            else:
                # 距离评估
                min_dist = min(abs(piece[0] - t[0]) + abs(piece[1] - t[1]) for t in self.target_coords)
                score -= min_dist * 2
                # r坐标进度
                score += piece[1] * 5
        
        if in_target == total_pieces:
            return 10000
        
        # 对手到达目标的惩罚
        opp_in_target = 0
        for piece in opp_pieces:
            if piece in self.home_set:
                opp_in_target += 1
                score -= 80
        
        if opp_in_target == total_pieces:
            return -10000
        
        return score
    
    def _simulate_move(self, my_pieces, move):
        if move == Move.END_TURN:
            return my_pieces
        
        new_pieces = list(my_pieces)
        src = (move.position.q, move.position.r)
        dst = move.moved_position()
        dst_tuple = (dst.q, dst.r)
        
        if src in new_pieces:
            new_pieces.remove(src)
            new_pieces.append(dst_tuple)
        
        return new_pieces

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None,
                       prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        actions = []
        
        for obs in obs_batch:
            action_mask = obs["action_mask"]
            my_pieces, opp_pieces, has_jump = self._obs_to_pieces(obs)
            
            legal_indices = np.where(action_mask == 1)[0]
            
            if len(legal_indices) == 0:
                actions.append(self.action_space_dim - 1)
                continue
            
            best_action_idx = legal_indices[0]
            best_score = float('-inf')
            
            for action_idx in legal_indices:
                move = action_to_move(action_idx, self.n)
                
                new_pieces = self._simulate_move(my_pieces, move)
                score = self._evaluate_state(new_pieces, opp_pieces)
                
                # 跳跃奖励
                if move != Move.END_TURN and move.is_jump:
                    score += 10
                
                # END_TURN惩罚
                if move == Move.END_TURN:
                    if has_jump and len(legal_indices) > 1:
                        score -= 50
                    else:
                        score -= 5
                
                # Alpha-Beta剪枝思想：选最大分
                if score > best_score:
                    best_score = score
                    best_action_idx = action_idx
            
            actions.append(best_action_idx)
        
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


# 增强版Minimax - 更智能的评估函数
class EnhancedMinimaxPolicy(Policy):
    def __init__(self, triangle_size=4, config={}, max_depth=2):
        observation_space = Box(low=0, high=1, shape=((4 * triangle_size + 1) * (4 * triangle_size + 1) * 4,), dtype=np.int8)
        action_space = Discrete((4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1)
        super().__init__(observation_space, action_space, config)
        self.triangle_size = triangle_size
        self.action_space_dim = action_space.n
        self.n = triangle_size
        
        # 预计算目标区域坐标 (相对坐标，当前玩家视角)
        n = triangle_size
        self.target_coords = []
        offset = np.array([-n, n + 1, -1])
        for i in range(n):
            for j in range(0, n - i):
                coord = offset + np.array([j, i, -i - j])
                self.target_coords.append((coord[0], coord[1]))
        self.target_set = set(self.target_coords)
        
        # 预计算起始区域坐标
        self.home_coords = []
        offset = np.array([1, -n - 1, n])
        for i in range(n):
            for j in range(i, n):
                coord = offset + np.array([j, -i, i - j])
                self.home_coords.append((coord[0], coord[1]))
        self.home_set = set(self.home_coords)
    
    def _obs_to_pieces(self, obs):
        n = self.n
        board_size = 4 * n + 1
        observation = obs["observation"].reshape(board_size, board_size, 4)
        
        my_pieces = []
        opp_pieces = []
        
        for q in range(board_size):
            for r in range(board_size):
                if observation[q, r, 0] == 1:
                    my_pieces.append((q - 2*n, r - 2*n))
                if observation[q, r, 1] == 1:
                    opp_pieces.append((q - 2*n, r - 2*n))
        
        has_jump = np.any(observation[:, :, 2] == 1)
        
        return my_pieces, opp_pieces, has_jump

    def _evaluate_state(self, my_pieces, opp_pieces):
        n = self.n
        total_pieces = n * (n + 1) // 2
        score = 0.0
        
        in_target = 0
        for piece in my_pieces:
            if piece in self.target_set:
                in_target += 1
                score += 200
            else:
                # 到目标的最小距离
                min_dist = min(abs(piece[0] - t[0]) + abs(piece[1] - t[1]) for t in self.target_coords)
                score -= min_dist * 5
                # r坐标进度奖励
                score += piece[1] * 10
        
        if in_target == total_pieces:
            return 100000
        
        # 对手进度惩罚
        for piece in opp_pieces:
            if piece in self.home_set:
                score -= 50
        
        return score
    
    def _simulate_move(self, my_pieces, move):
        if move == Move.END_TURN:
            return my_pieces
        
        new_pieces = list(my_pieces)
        src = (move.position.q, move.position.r)
        dst = move.moved_position()
        dst_tuple = (dst.q, dst.r)
        
        if src in new_pieces:
            new_pieces.remove(src)
            new_pieces.append(dst_tuple)
        
        return new_pieces

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None,
                       prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        actions = []
        
        for obs in obs_batch:
            action_mask = obs["action_mask"]
            my_pieces, opp_pieces, has_jump = self._obs_to_pieces(obs)
            
            legal_indices = np.where(action_mask == 1)[0]
            
            if len(legal_indices) == 0:
                actions.append(self.action_space_dim - 1)
                continue
            
            best_action_idx = legal_indices[0]
            best_score = float('-inf')
            
            for action_idx in legal_indices:
                move = action_to_move(action_idx, self.n)
                
                new_pieces = self._simulate_move(my_pieces, move)
                score = self._evaluate_state(new_pieces, opp_pieces)
                
                if move != Move.END_TURN:
                    # 跳跃奖励
                    if move.is_jump:
                        score += 15
                    # 方向奖励
                    if move.direction in [Direction.DownLeft, Direction.DownRight]:
                        score += 10
                    elif move.direction in [Direction.UpLeft, Direction.UpRight]:
                        score -= 20
                else:
                    # END_TURN惩罚
                    if has_jump and len(legal_indices) > 1:
                        score -= 50
                    else:
                        score -= 5
                
                if score > best_score:
                    best_score = score
                    best_action_idx = action_idx
            
            actions.append(best_action_idx)
        
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