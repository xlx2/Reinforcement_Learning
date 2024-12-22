#  -*- coding: utf-8 -*-
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random


class SnakeGameEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 15}

    def __init__(self, render_mode=None, window_size=360, cell_size=10, speed=15):
        super().__init__()

        # 游戏参数
        self.WINDOW_SIZE = window_size
        self.CELL_SIZE = cell_size
        self.grid_size = self.WINDOW_SIZE // self.CELL_SIZE
        self.speed = speed
        self.render_mode = render_mode

        # 动作空间：上下左右
        self.action_space = spaces.Discrete(4)  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT

        # 状态空间：3D 表示（网格 x 网格 x 3 通道）
        # self.observation_space = spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(self.grid_size, self.grid_size, 3),
        #     dtype=np.uint8
        # )
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.grid_size * self.grid_size * 3,),  # 展平后的状态维度
            dtype=np.uint8
        )

        # 初始化 pygame（仅在需要渲染时）
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
            pygame.display.set_caption("Snake RL Game")
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake_position = [[100, 100], [90, 100], [80, 100]]
        self.snake_head = [100, 100]
        self.apple_position = self._get_new_apple_position()
        self.score = 0
        self.direction = "RIGHT"
        self.done = False

        return self._get_observation(), {}

    def step(self, action):
        """执行一步动作"""
        assert self.action_space.contains(action), "Invalid Action"

        # 更新方向
        if action == 0 and self.direction != "DOWN":
            self.direction = "UP"
        elif action == 1 and self.direction != "UP":
            self.direction = "DOWN"
        elif action == 2 and self.direction != "RIGHT":
            self.direction = "LEFT"
        elif action == 3 and self.direction != "LEFT":
            self.direction = "RIGHT"

        # 更新蛇头位置
        if self.direction == "UP":
            self.snake_head[1] -= self.CELL_SIZE
        elif self.direction == "DOWN":
            self.snake_head[1] += self.CELL_SIZE
        elif self.direction == "LEFT":
            self.snake_head[0] -= self.CELL_SIZE
        elif self.direction == "RIGHT":
            self.snake_head[0] += self.CELL_SIZE

        # 增长蛇的身体
        self.snake_position.insert(0, list(self.snake_head))

        # 初始化奖励
        reward = 0

        # 检测碰撞（游戏结束）
        self.done = self._is_collision()
        if self.done:
            reward -= 10
            observation = self._get_observation()
            return observation, reward, self.done, False, {}

        # 检测是否吃到苹果
        if self.snake_head == self.apple_position:
            reward += 1  # 吃到苹果奖励
            self.apple_position = self._get_new_apple_position()
        else:
            self.snake_position.pop()  # 如果没吃到苹果，去掉尾部

        # 存活奖励
        reward += 0.0001

        # 曼哈顿距离奖励（鼓励靠近苹果）
        distance = (abs(self.snake_head[0] - self.apple_position[0]) +
                    abs(self.snake_head[1] - self.apple_position[1]))
        reward -= distance * 0.001

        # 获取观察值
        observation = self._get_observation()
        return observation, reward, self.done, False, {}

    def render(self):
        """渲染游戏画面"""
        if self.render_mode == "human":
            self.screen.fill((0, 0, 0))

            # 绘制苹果
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),
                pygame.Rect(self.apple_position[0], self.apple_position[1], self.CELL_SIZE, self.CELL_SIZE)
            )
            # 绘制蛇
            for block in self.snake_position:
                pygame.draw.rect(
                    self.screen,
                    (0, 255, 0),
                    pygame.Rect(block[0], block[1], self.CELL_SIZE, self.CELL_SIZE)
                )
            pygame.display.flip()
            self.clock.tick(self.speed)

    def close(self):
        """关闭游戏"""
        if self.render_mode == "human":
            pygame.quit()

    def _get_new_apple_position(self):
        """生成新的苹果位置"""
        while True:
            position = [
                random.randrange(1, self.WINDOW_SIZE // self.CELL_SIZE) * self.CELL_SIZE,
                random.randrange(1, self.WINDOW_SIZE // self.CELL_SIZE) * self.CELL_SIZE
            ]
            if position not in self.snake_position:
                return position

    def _is_collision(self):
        """检测碰撞"""
        # 碰撞边界
        if (self.snake_head[0] >= self.WINDOW_SIZE or self.snake_head[0] < 0 or
                self.snake_head[1] >= self.WINDOW_SIZE or self.snake_head[1] < 0):
            return True
        # 碰撞自己
        if self.snake_head in self.snake_position[1:]:
            return True
        return False

    def _get_observation(self):
        """获取当前环境状态"""
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        for block in self.snake_position:
            grid_y, grid_x = block[1] // self.CELL_SIZE, block[0] // self.CELL_SIZE
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                obs[grid_y, grid_x, 0] = 1  # 蛇
        obs[self.apple_position[1] // self.CELL_SIZE, self.apple_position[0] // self.CELL_SIZE, 1] = 1  # 苹果
        return obs.flatten()

