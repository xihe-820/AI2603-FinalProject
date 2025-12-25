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
        self.max_depth = max_depth
        
        # 预计算目标区域坐标
        self._target_coords_cache = {}
        self._home_coords_cache = {}
        
    def _get_target_coords(self, player):
        if player not in self._target_coords_cache:
            n = self.triangle_size
            if player == 0:
                offset = np.array([-n, n + 1, -1])
                coords = []
                for i in range(n):
                    for j in range(0, n - i):
                        q, r, s = j, i, -i - j
                        coord = offset + np.array([q, r, s])
                        coords.append((coord[0], coord[1]))
                self._target_coords_cache[player] = coords
            else:  # player == 3
                offset = np.array([1, -n - 1, n])
                coords = []
                for i in range(n):
                    for j in range(i, n):
                        q, r, s = j, -i, i - j
                        coord = offset + np.array([q, r, s])
                        coords.append((coord[0], coord[1]))
                self._target_coords_cache[player] = coords
        return self._target_coords_cache[player]
    
    def _get_home_coords(self, player):
        if player not in self._home_coords_cache:
            n = self.triangle_size
            if player == 0:
                offset = np.array([1, -n - 1, n])
                coords = []
                for i in range(n):
                    for j in range(i, n):
                        q, r, s = j, -i, i - j
                        coord = offset + np.array([q, r, s])
                        coords.append((coord[0], coord[1]))
                self._home_coords_cache[player] = coords
            else:  # player == 3
                offset = np.array([-n, n + 1, -1])
                coords = []
                for i in range(n):
                    for j in range(0, n - i):
                        q, r, s = j, i, -i - j
                        coord = offset + np.array([q, r, s])
                        coords.append((coord[0], coord[1]))
                self._home_coords_cache[player] = coords
        return self._home_coords_cache[player]

    def _obs_to_board_state(self, obs):
        # 从观察重建棋盘状态
        n = self.triangle_size
        board_size = 4 * n + 1
        observation = obs["observation"].reshape(board_size, board_size, 4)
        
        # 提取棋子位置
        player0_pieces = []
        player3_pieces = []
        
        for q in range(board_size):
            for r in range(board_size):
                if observation[q, r, 0] == 1:
                    player0_pieces.append((q - 2*n, r - 2*n))
                if observation[q, r, 1] == 1:
                    player3_pieces.append((q - 2*n, r - 2*n))
        
        # 提取跳跃状态
        jump_sources = []
        last_jump_dest = None
        for q in range(board_size):
            for r in range(board_size):
                if observation[q, r, 2] == 1:
                    jump_sources.append((q - 2*n, r - 2*n))
                if observation[q, r, 3] == 1:
                    last_jump_dest = (q - 2*n, r - 2*n)
        
        return {
            'player0': player0_pieces,
            'player3': player3_pieces,
            'jump_sources': jump_sources,
            'last_jump_dest': last_jump_dest
        }

    def _evaluate_state(self, state, maximizing_player):
        # 评估函数
        n = self.triangle_size
        my_pieces = state['player0'] if maximizing_player == 0 else state['player3']
        opp_pieces = state['player3'] if maximizing_player == 0 else state['player0']
        
        my_target = self._get_target_coords(maximizing_player)
        opp_target = self._get_target_coords(3 - maximizing_player if maximizing_player == 0 else 0)
        
        score = 0.0
        
        # 计算自己的进度
        my_in_target = 0
        my_total_dist = 0
        for piece in my_pieces:
            if piece in my_target:
                my_in_target += 1
                score += 100  # 大奖励 - 已到达目标
            else:
                # 距离评估 - 越接近目标越好
                min_dist = float('inf')
                for t in my_target:
                    dist = abs(piece[0] - t[0]) + abs(piece[1] - t[1])
                    min_dist = min(min_dist, dist)
                my_total_dist += min_dist
                
                # 向前移动奖励 (player 0向下, player 3向上)
                if maximizing_player == 0:
                    score += piece[1] * 5  # r坐标越大越好
                else:
                    score -= piece[1] * 5  # r坐标越小越好
        
        score -= my_total_dist * 2
        
        # 计算对手进度作为惩罚
        opp_in_target = 0
        for piece in opp_pieces:
            if piece in opp_target:
                opp_in_target += 1
                score -= 80
        
        # 胜负判断
        total_pieces = n * (n + 1) // 2
        if my_in_target == total_pieces:
            return 10000
        if opp_in_target == total_pieces:
            return -10000
        
        return score
    
    def _simulate_move(self, state, move, current_player):
        # 模拟移动
        new_state = {
            'player0': list(state['player0']),
            'player3': list(state['player3']),
            'jump_sources': list(state['jump_sources']),
            'last_jump_dest': state['last_jump_dest']
        }
        
        if move == Move.END_TURN:
            new_state['jump_sources'] = []
            new_state['last_jump_dest'] = None
            return new_state
        
        pieces_key = 'player0' if current_player == 0 else 'player3'
        src = (move.position.q, move.position.r)
        dst = move.moved_position()
        dst = (dst.q, dst.r)
        
        if src in new_state[pieces_key]:
            new_state[pieces_key].remove(src)
            new_state[pieces_key].append(dst)
        
        if move.is_jump:
            new_state['jump_sources'].append(src)
            new_state['last_jump_dest'] = dst
        else:
            new_state['jump_sources'] = []
            new_state['last_jump_dest'] = None
        
        return new_state

    def _get_legal_moves_from_mask(self, action_mask):
        moves = []
        for i, valid in enumerate(action_mask):
            if valid == 1:
                moves.append(action_to_move(i, self.triangle_size))
        return moves

    def _minimax(self, state, depth, alpha, beta, maximizing, current_player, action_mask, my_player):
        if depth == 0:
            return self._evaluate_state(state, my_player), None
        
        legal_moves = self._get_legal_moves_from_mask(action_mask)
        
        if not legal_moves:
            return self._evaluate_state(state, my_player), None
        
        # 简化: 深度较浅时只评估当前状态
        if depth < self.max_depth:
            return self._evaluate_state(state, my_player), None
        
        best_move = legal_moves[0]
        
        if maximizing:
            max_eval = float('-inf')
            for move in legal_moves:
                new_state = self._simulate_move(state, move, current_player)
                eval_score = self._evaluate_state(new_state, my_player)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in legal_moves:
                new_state = self._simulate_move(state, move, current_player)
                eval_score = self._evaluate_state(new_state, my_player)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None,
                       prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        actions = []
        
        for obs in obs_batch:
            action_mask = obs["action_mask"]
            state = self._obs_to_board_state(obs)
            
            # 获取合法动作
            legal_moves = self._get_legal_moves_from_mask(action_mask)
            
            if not legal_moves:
                actions.append(self.action_space_dim - 1)
                continue
            
            # 对每个合法动作评分
            best_action = None
            best_score = float('-inf')
            
            for move in legal_moves:
                new_state = self._simulate_move(state, move, 0)
                score = self._evaluate_state(new_state, 0)
                
                # 优先选择跳跃动作 (可能连跳)
                if move != Move.END_TURN and move.is_jump:
                    score += 10
                
                # 惩罚END_TURN除非必要
                if move == Move.END_TURN and len(legal_moves) > 1:
                    score -= 5
                
                if score > best_score:
                    best_score = score
                    best_action = move
            
            if best_action is None:
                best_action = legal_moves[0]
            
            actions.append(move_to_action(best_action, self.triangle_size))
        
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
        self.max_depth = max_depth
        
        # 预计算目标区域
        n = triangle_size
        self.target_coords_p0 = set()
        offset = np.array([-n, n + 1, -1])
        for i in range(n):
            for j in range(0, n - i):
                coord = offset + np.array([j, i, -i - j])
                self.target_coords_p0.add((coord[0], coord[1]))
        
        self.home_coords_p0 = set()
        offset = np.array([1, -n - 1, n])
        for i in range(n):
            for j in range(i, n):
                coord = offset + np.array([j, -i, i - j])
                self.home_coords_p0.add((coord[0], coord[1]))
    
    def _obs_to_pieces(self, obs):
        n = self.triangle_size
        board_size = 4 * n + 1
        observation = obs["observation"].reshape(board_size, board_size, 4)
        
        player0_pieces = []
        player3_pieces = []
        
        for q in range(board_size):
            for r in range(board_size):
                if observation[q, r, 0] == 1:
                    player0_pieces.append((q - 2*n, r - 2*n))
                if observation[q, r, 1] == 1:
                    player3_pieces.append((q - 2*n, r - 2*n))
        
        has_jump = np.any(observation[:, :, 2] == 1)
        
        return player0_pieces, player3_pieces, has_jump

    def _evaluate_position(self, my_pieces, opp_pieces):
        n = self.triangle_size
        score = 0.0
        
        in_target = 0
        for piece in my_pieces:
            q, r = piece
            
            # 已到达目标区域
            if piece in self.target_coords_p0:
                in_target += 1
                score += 200
            else:
                # 向下移动进度 (r越大越好)
                score += r * 20
                
                # 离开起始区惩罚减轻
                if piece in self.home_coords_p0:
                    score -= 30
                
                # 到目标区域的曼哈顿距离
                min_dist = min(abs(q - t[0]) + abs(r - t[1]) for t in self.target_coords_p0)
                score -= min_dist * 3
        
        # 胜利检测
        total_pieces = n * (n + 1) // 2
        if in_target == total_pieces:
            return 100000
        
        # 轻微考虑对手位置
        for piece in opp_pieces:
            if piece in self.home_coords_p0:  # 对手到达其目标(我方home)
                score -= 15
        
        return score

    def _simulate_move_simple(self, my_pieces, move):
        if move == Move.END_TURN:
            return my_pieces
        
        new_pieces = list(my_pieces)
        src = (move.position.q, move.position.r)
        dst = move.moved_position()
        dst = (dst.q, dst.r)
        
        if src in new_pieces:
            new_pieces.remove(src)
            new_pieces.append(dst)
        
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
                move = action_to_move(action_idx, self.triangle_size)
                
                new_my_pieces = self._simulate_move_simple(my_pieces, move)
                score = self._evaluate_position(new_my_pieces, opp_pieces)
                
                if move != Move.END_TURN:
                    dst = move.moved_position()
                    src = move.position
                    
                    # 向前移动奖励
                    delta_r = dst.r - src.r
                    if delta_r > 0:
                        score += delta_r * 15
                        if move.is_jump:
                            score += 20  # 跳跃额外奖励
                    elif delta_r < 0:
                        score += delta_r * 25  # 后退惩罚更重
                    
                    # 水平跳跃也给一点奖励（可能是为了绕障碍）
                    if move.is_jump and delta_r == 0:
                        score += 5
                else:
                    # END_TURN
                    if has_jump and len(legal_indices) > 1:
                        score -= 30
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