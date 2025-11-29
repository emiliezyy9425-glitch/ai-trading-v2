# ppo.py — FINAL VERSION (uses price_1h instead of close)
# Updated: 2025-11-17
# Works 100% with your current historical_data.csv

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import logging
import os
from self_learn import FEATURE_NAMES

# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/ppo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

EXPECTED_FEATURE_COUNT = len(FEATURE_NAMES)


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data=None):
        super().__init__()

        if data is not None:
            self.data = data.copy().reset_index(drop=True)
            logger.info(f"Using in-memory data with {len(self.data)} rows")
        else:
            raise ValueError("PPO env requires data passed directly")

        self.current_step = 0
        self.max_steps = len(self.data) - 1

        self.transaction_cost = 0.001
        self.initial_balance = 100_000.0
        self.cash = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance

        self.position = 0        # -1, 0, 1
        self.entry_price = 0.0
        self.position_size = 100

        self.returns = []

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(EXPECTED_FEATURE_COUNT,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=Buy, 1=Sell, 2=Hold

    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        return row[FEATURE_NAMES].values.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 200
        self.cash = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.returns = []
        return self._get_observation(), {}

    def _update_portfolio_value(self, price):
        self.portfolio_value = self.cash + (self.position * self.position_size * price)

    def _calculate_reward(self):
        current_value = self.portfolio_value
        prev_value = self.previous_portfolio_value or current_value
        if prev_value <= 0:
            return 0.0

        pct_return = (current_value - prev_value) / prev_value
        reward = pct_return * 100

        # TD9 BONUS
        td9_1h = self.data.iloc[self.current_step].get("td9_1h", 0)
        td9_4h = self.data.iloc[self.current_step].get("td9_4h", 0)

        if td9_1h in [9, 13] and self.position == 1:
            reward += 4.0
        if td9_4h in [9, 13] and self.position == 1:
            reward += 8.0
        if td9_1h in [-9, -13] and self.position == -1:
            reward += 4.0
        if td9_4h in [-9, -13] and self.position == -1:
            reward += 8.0
        if abs(td9_1h) in [9, 13] and self.position == 0:
            reward -= 0.8

        # VIX filter
        vix = self.data.iloc[self.current_step].get("vix", 20)
        if vix > 35 and abs(self.position) > 0:
            reward *= 0.5

        # Sharpe bonus
        if len(self.returns) >= 10:
            recent = np.array(self.returns[-10:])
            sharpe = recent.mean() / (recent.std() + 1e-8)
            if sharpe > 0.5:
                reward += sharpe * 0.5

        if self.position == 0:
            reward -= 0.01

        return float(reward)

    def step(self, action):
        # CRITICAL FIX: use price_1h instead of close
        current_price = float(self.data.iloc[self.current_step]["price_1h"])

        # Execute action
        if action == 0 and self.position != 1:  # Buy
            if self.position == -1:
                profit = self.position_size * (self.entry_price - current_price)
                self.cash += profit * (1 - self.transaction_cost)
            cost = self.position_size * current_price * (1 + self.transaction_cost)
            if self.cash >= cost:
                self.cash -= cost
                self.position = 1
                self.entry_price = current_price

        elif action == 1 and self.position != -1:  # Sell
            if self.position == 1:
                profit = self.position_size * (current_price - self.entry_price)
                self.cash += profit * (1 - self.transaction_cost)
            proceeds = self.position_size * current_price * (1 - self.transaction_cost)
            self.cash += proceeds
            self.position = -1
            self.entry_price = current_price

        self._update_portfolio_value(current_price)

        reward = self._calculate_reward()
        pct_return = (self.portfolio_value - self.previous_portfolio_value) / (self.previous_portfolio_value or 1)
        self.returns.append(pct_return)
        if len(self.returns) > 100:
            self.returns.pop(0)

        self.previous_portfolio_value = self.portfolio_value
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False
        obs = self._get_observation() if not terminated else np.zeros(EXPECTED_FEATURE_COUNT)

        info = {"portfolio_value": self.portfolio_value, "position": self.position}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass
