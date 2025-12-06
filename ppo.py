import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import logging
import os
from self_learn import FEATURE_NAMES

CURRENT_FEATURE_DIM = len(FEATURE_NAMES)

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/ppo.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data=None):
        super().__init__()
        if data is None or "ret_1h" not in data.columns:
            raise ValueError("Data with 'ret_1h' required")
        self.data = data.reset_index(drop=True)
        logger.info(f"PPO Env: {len(self.data)} rows")

        self.current_step = 0
        self.max_steps = len(self.data) - 1
        self.initial_balance = 100_000.0
        self.cash = self.initial_balance
        self.position = 0  # -1, 0, +1
        self.entry_price = 100.0
        self.position_size = 100
        self.transaction_cost = 0.001
        self.portfolio_value = self.initial_balance
        self.previous_value = self.initial_balance
        self.returns = []

        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(CURRENT_FEATURE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0=Buy, 1=Sell, 2=Hold

    def _get_price(self):
        """Reconstruct price from cumulative log returns — mathematically perfect"""
        cum_ret = self.data["ret_1h"].iloc[: self.current_step + 1].sum()
        return np.exp(cum_ret) * 100.0

    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        return row.reindex(FEATURE_NAMES).fillna(0.0).astype(np.float32).values

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 200
        self.cash = self.initial_balance
        self.position = 0
        self.entry_price = 100.0
        self.portfolio_value = self.initial_balance
        self.previous_value = self.initial_balance
        self.returns = []
        return self._get_observation(), {}

    def _update_portfolio(self, price):
        shares = self.position * self.position_size
        self.portfolio_value = self.cash + shares * price
        # CRITICAL: NEVER LET previous_value BE ZERO OR NEGATIVE
        if self.portfolio_value <= 0:
            self.portfolio_value = 1e-8

    def _calculate_reward(self):
        # This is now 100% safe — no log(0) possible
        current = max(self.portfolio_value, 1e-8)
        prev = max(self.previous_value, 1e-8)
        log_ret = np.log(current / prev)

        market_ret = (
            self.data["ret_1h"].iloc[self.current_step]
            if self.current_step < len(self.data)
            else 0.0
        )
        right_direction = (self.position == 1 and market_ret > 0) or (
            self.position == -1 and market_ret < 0
        )
        direction_bonus = 1.0 if right_direction else 0.9

        if len(self.returns) >= 20:
            sharpe = np.mean(self.returns) / (np.std(self.returns) + 1e-8)
            reward = log_ret * 80 + sharpe * 4
        else:
            reward = log_ret * 80

        return float(np.clip(reward * direction_bonus, -3.0, 3.0))

    def step(self, action):
        price = self._get_price()

        if action == 0 and self.position != 1:  # Buy
            cost = self.position_size * price * (1 + self.transaction_cost)
            if self.position == -1:  # close short first
                profit = self.position_size * (self.entry_price - price)
                self.cash += profit * (1 - self.transaction_cost)
            if self.cash >= cost:
                self.cash -= cost
                self.position = 1
                self.entry_price = price

        elif action == 1 and self.position != -1:  # Sell
            proceeds = self.position_size * price * (1 - self.transaction_cost)
            if self.position == 1:  # close long first
                profit = self.position_size * (price - self.entry_price)
                self.cash += profit * (1 - self.transaction_cost)
            self.cash += proceeds
            self.position = -1
            self.entry_price = price

        self._update_portfolio(price)
        reward = self._calculate_reward()

        pct_change = (self.portfolio_value - self.previous_value) / (self.previous_value + 1e-8)
        self.returns.append(pct_change)
        if len(self.returns) > 100:
            self.returns.pop(0)

        self.previous_value = self.portfolio_value
        self.current_step += 1

        done = self.current_step >= self.max_steps
        obs = (
            self._get_observation()
            if not done
            else np.zeros(CURRENT_FEATURE_DIM, dtype=np.float32)
        )

        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "price": price
        }

        return obs, reward, done, False, info

    def render(self):
        pass
