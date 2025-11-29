import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import logging
import os
from self_learn import FEATURE_NAMES

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


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data=None):
        super().__init__()
        if data is None:
            raise ValueError("Data required")
        self.data = data.reset_index(drop=True)
        logger.info(f"PPO Env loaded with {len(self.data)} rows")

        self.current_step = 0
        self.max_steps = len(self.data) - 1
        self.initial_balance = 100_000.0
        self.cash = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.position_size = 100
        self.transaction_cost = 0.001
        self.portfolio_value = self.initial_balance
        self.previous_value = self.initial_balance
        self.returns = []

        self.observation_space = spaces.Box(low=-10, high=10, shape=(len(FEATURE_NAMES),), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=Buy, 1=Sell, 2=Hold

    def _get_price(self):
        """Reconstruct price from cumulative log returns — mathematically perfect"""
        if "ret_1h" not in self.data.columns:
            return 100.0
        cum_ret = self.data["ret_1h"].iloc[:self.current_step+1].sum()
        return np.exp(cum_ret) * 100.0  # start from $100

    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        row_filled = row.reindex(FEATURE_NAMES).fillna(0.0)
        return row_filled.values.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 200
        self.cash = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.portfolio_value = self.initial_balance
        self.previous_value = self.initial_balance
        self.returns = []
        return self._get_observation(), {}

    def _update_portfolio(self, price):
        shares = self.position * self.position_size
        self.portfolio_value = self.cash + shares * price

    def _calculate_reward(self):
        """Log-return-based reward — stable and scale-invariant"""
        if self.previous_value <= 0:
            return 0.0
        # Use log portfolio growth (same scale as ret_1h)
        log_return = np.log(self.portfolio_value / self.previous_value)
        # Bonus for being in the market during up moves
        market_move = self.data["ret_1h"].iloc[self.current_step] if self.current_step < len(self.data) else 0
        direction_bonus = 1.0 if (self.position == 1 and market_move > 0) or (self.position == -1 and market_move < 0) else 0.8

        # Sharpe-like smoothing
        if len(self.returns) > 20:
            sharpe = np.mean(self.returns) / (np.std(self.returns) + 1e-6)
            reward = log_return * 50 + sharpe * 2
        else:
            reward = log_return * 50

        return float(reward * direction_bonus)

    def step(self, action):
        current_price = self._get_price()

        if action == 0 and self.position != 1:  # Buy
            cost = self.position_size * current_price * (1 + self.transaction_cost)
            if self.position == -1:
                profit = self.position_size * (self.entry_price - current_price)
                self.cash += profit * (1 - self.transaction_cost)
            if self.cash >= cost:
                self.cash -= cost
                self.position = 1
                self.entry_price = current_price

        elif action == 1 and self.position != -1:  # Sell
            proceeds = self.position_size * current_price * (1 - self.transaction_cost)
            if self.position == 1:
                profit = self.position_size * (current_price - self.entry_price)
                self.cash += profit * (1 - self.transaction_cost)
            self.cash += proceeds
            self.position = -1
            self.entry_price = current_price

        self._update_portfolio(current_price)
        reward = self._calculate_reward()

        pct = (self.portfolio_value - self.previous_value) / (self.previous_value or 1)
        self.returns.append(pct)
        if len(self.returns) > 100:
            self.returns.pop(0)

        self.previous_value = self.portfolio_value
        self.current_step += 1

        done = self.current_step >= self.max_steps
        obs = self._get_observation() if not done else np.zeros(len(FEATURE_NAMES))

        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "price": current_price
        }

        return obs, reward, done, False, info

    def render(self):
        pass
