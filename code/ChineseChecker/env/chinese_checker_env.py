import functools
import re

import gymnasium
import numpy as np
from matplotlib import pyplot as plt

from gymnasium.spaces import Box, Discrete, Dict
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
import pygame

# 导入自定义的游戏类和工具函数
from .game import ChineseCheckers, Direction, Move, Position
from .utils import action_to_move, get_legal_move_mask, rotate_observation

def env(**kwargs):
    """
    创建环境包装器
    Returns:
        AECEnv: 包装后的环境实例
    """
    env = raw_env(**kwargs)
    # 可选包装器：非法动作终止（当前被注释掉）
    # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)    # 断言越界检查
    env = wrappers.OrderEnforcingWrapper(env)       # 强制动作顺序
    return env
        
class raw_env(AECEnv):
    """
    跳棋游戏环境的主要实现类
    继承自PettingZoo的AECEnv（异步环境）
    """
    metadata = {
        "render_modes": ["rgb_array", "human"],  # 渲染模式
        "name": "chinese_checkers"               # 环境名称
    }

    def __init__(self, render_mode: str = "rgb_array", triangle_size: int = 4, max_iters: int = 1000, **kwargs):
        """
        初始化环境
        
        Args:
            render_mode: 渲染模式，"rgb_array"或"human"
            triangle_size: 三角区域大小，决定棋盘规模
            max_iters: 最大迭代次数，用于终止条件
        """
        self.max_iters = max_iters

        # 定义2个玩家        
        self.agents = [f"player_{r}" for r in [0, 3]]
        self.possible_agents = self.agents[:]   # 可能的玩家列表
        self._agent_selector = agent_selector(self.agents)  # 玩家选择器
        self.agent_name_mapping = dict(
            # zip(self.possible_agents, list(range(len(self.possible_agents))))
            zip(self.possible_agents, [0, 3])
        )
        
        # 游戏参数
        self.n = triangle_size       # 三角区域大小
        self.iters = 0               # 当前迭代次数
        self.num_moves = 0           # 移动步数
        self.winner = None           # 获胜者

        # 游戏状态记录
        self.rewards = None          # 奖励字典
        self.infos = {agent: {} for agent in self.agents}        # 信息字典
        self.truncations = {agent: False for agent in self.agents}  # 截断标志
        self.terminations = {agent: False for agent in self.agents} # 终止标志

        # 动作和观察空间维度计算
        # 动作空间：(棋盘大小)^2 × 6个方向 × 2种移动类型(跳/不跳) + 1(结束回合)
        self.action_space_dim = (4 * self.n + 1) * (4 * self.n + 1) * 6 * 2 + 1
        # 观察空间：(棋盘大小)^2 × 4个通道
        self.observation_space_dim = (4 * self.n + 1) * (4 * self.n + 1) * 4
        
        # 定义每个玩家的动作和观察空间
        self.action_spaces = {agent: Discrete(self.action_space_dim) for agent in self.agents}
        self.observation_spaces = {
            agent: Dict({
                # 原始观察形状被注释掉，现使用扁平化版本
                # "observation": Box(low=0, high=1, shape=(4 * self.n + 1, 4 * self.n + 1, 8)),
                "observation": Box(low=0, high=1, shape=(self.observation_space_dim,)),  # 扁平化观察
                "action_mask": Box(low=0, high=1, shape=(self.action_space_dim,), dtype=np.int8)  # 动作掩码
            })
            for agent in self.agents
        }

        self.agent_selection = None  # 当前选择的玩家

        # 渲染相关
        self.window_size = 512       # PyGame窗口大小
        self.render_mode = render_mode

        self.rotation = 0            # 棋盘旋转角度
        self.game = ChineseCheckers(triangle_size, render_mode=render_mode)  # 游戏实例

    def reset(self, seed=None, return_info=None, options=None):
        """
        重置环境到初始状态
        
        Returns:
            dict: 初始观察
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.game.init_game()  # 初始化游戏
        self.iters = 0
        self.num_moves = 0
        self.winner = None

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()  # 选择第一个玩家
        return self.observe(self.agent_selection)

    def _calculate_progress(self, player):
        """
        计算玩家的进度分数
        基于棋子到目标区域的距离
        
        Args:
            player: 玩家编号
            
        Returns:
            float: 进度分数（越高越好）
        """
        progress = 0.0
        target_coords = list(self.game.get_target_coordinates(player))
        
        # 获取玩家的棋子位置
        board = self.game.get_axial_board(player)
        for q in range(-2 * self.n, 2 * self.n + 1):
            for r in range(-2 * self.n, 2 * self.n + 1):
                q_idx = q + 2 * self.n
                r_idx = r + 2 * self.n
                if 0 <= q_idx < board.shape[0] and 0 <= r_idx < board.shape[1]:
                    if board[q_idx, r_idx] == 0:  # 当前玩家的棋子（旋转后为0）
                        # 检查是否在目标区域
                        in_target = any(t[0] == q and t[1] == r for t in target_coords)
                        if in_target:
                            progress += 10.0  # 在目标区域内
                        else:
                            # 计算到目标区域的r坐标进度（目标在下方，r越大越好）
                            progress += r * 0.5
        
        return progress

    def step(self, action):
        """
        执行一步动作
        
        Args:
            action: 选择的动作索引
        """
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)  # 处理终止或截断状态
            return
        
        agent: str = self.agent_selection
        player: int = self.agent_name_mapping[agent]

        action = int(action)

        # 将动作索引转换为Move对象
        move = action_to_move(action, self.n)
        
        # 记录移动前的状态用于计算奖励
        old_progress = self._calculate_progress(player)
        
        # 执行移动
        move = self.game.move(player, move)
        
        # 计算移动后的进度
        new_progress = self._calculate_progress(player)

        # 奖励分配逻辑
        if self.game.did_player_win(self.agent_name_mapping[agent]):
            # 玩家获胜 - 大奖励
            self.terminations = {
                agent: self.game.is_game_over() for agent in self.agents
            }
            for a in self.agents:
                self.rewards[a] = 100 if a == agent else -10  # 增大胜负奖励差距
            self.winner = agent
        elif move is None:
            # 非法移动
            self.rewards[agent] = -1000
        else:
            # 基于进度变化的奖励（核心改进）
            progress_reward = (new_progress - old_progress) * 2.0  # 进度奖励
            self.rewards[agent] = progress_reward
            
            # 方向相关的奖励/惩罚（增强）
            if isinstance(move, Move) and move != Move.END_TURN:
                if move.direction in [Direction.DownLeft, Direction.DownRight]:
                    self.rewards[agent] += 0.1  # 向下移动奖励
                    if move.is_jump:
                        self.rewards[agent] += 0.2  # 向下跳跃额外奖励
                elif move.direction in [Direction.UpLeft, Direction.UpRight]:
                    self.rewards[agent] -= 0.15  # 向上移动惩罚
            
            # 目标区域进出奖励（增强）
            if move and move != Move.END_TURN:
                src_pos = move.position
                dst_pos = move.moved_position()
                target = [Position(q, r) for q, r, s in self.game.get_target_coordinates(player)]
                if src_pos not in target and dst_pos in target:
                    self.rewards[agent] += 1.0  # 进入目标区域（增大10倍）
                if src_pos in target and dst_pos not in target:
                    self.rewards[agent] -= 1.5  # 离开目标区域（惩罚更重）

        self._accumulate_rewards()  # 累积奖励
        self._clear_rewards()       # 清除当前奖励

        # 检查是否达到最大迭代次数
        if self._agent_selector.is_last():
            self.truncations = {
                agent: self.iters >= self.max_iters for agent in self.agents
            }
        self.num_moves += 1

        # 如果是跳跃且不是结束回合，则当前玩家继续行动
        if move == Move.END_TURN or not (move and move.is_jump):
            self.agent_selection = self._agent_selector.next()  # 切换到下一个玩家
            if self._agent_selector.is_last():
                self.iters += 1  # 完成一轮后增加迭代计数

    def render(self):
        """渲染当前游戏状态"""
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        else:
            return self.game.render()

    def observation_space(self, agent):
        """返回指定玩家的观察空间"""
        return self.observation_spaces[agent]
        
    def observe(self, agent):
        """
        获取指定玩家的观察
        
        Args:
            agent: 玩家名称
            
        Returns:
            dict: 包含观察和动作掩码的字典
            
        观察包含4个通道：
        - 通道0: 当前玩家的棋子
        - 通道1: 其他玩家的棋子
        - 通道2: 当前玩家所有跳跃的起始位置
        - 通道3: 上一次跳跃的目标位置
        """
        player = self.agent_name_mapping[agent]
        board = self.game.get_axial_board(player)  # 获取轴向坐标系下的棋盘

        # 初始化跳跃相关通道
        jump_sources_channel = -2 * np.zeros((4 * self.game.n + 1, 4 * self.game.n + 1), dtype=np.int8)
        last_jump_destination_channel = -2 * np.zeros((4 * self.game.n + 1, 4 * self.game.n + 1), dtype=np.int8)
        
        # 获取跳跃信息
        last_jump = self.game.get_last_jump(player)
        if last_jump:
            jumps = self.game.get_jumps(player)
            for jump in jumps:
                jump_sources_channel[jump.position.q, jump.position.r] = 1
            last_jump_destination = last_jump.moved_position()
            last_jump_destination_channel[last_jump_destination.q, last_jump_destination.r] = 1

        # 堆叠所有通道
        observation = np.stack(
            [(board == player).astype(np.int8) for player in [0, 3]] + 
            [jump_sources_channel, last_jump_destination_channel],
            axis=-1
        )

        observation = observation.flatten()  # 扁平化处理

        return {
            "observation": observation,
            "action_mask": get_legal_move_mask(self.game, player)  # 合法动作掩码
        }
    
    def action_space(self, agent):
        """返回指定玩家的动作空间"""
        return self.action_spaces[agent]
    
    def close(self):
        """关闭环境"""
        pass
