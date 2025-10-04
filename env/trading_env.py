import gymnasium as gym
import numpy as np

class TradingEnv(gym.Env):
    """
    A simple trading environment with discrete actions: Hold, Buy, Sell.
    Compatible with Gymnasium and Stable-Baselines3.
    """
    metadata = {"render_modes": []}

    def __init__(self, data, window_size=10, initial_balance=1000):
        super().__init__()
        self.df = data.reset_index(drop=True)
        self.window_size = int(window_size)
        self.initial_balance = float(initial_balance)

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = gym.spaces.Discrete(3)

        # Observation space: (window_size, num_features)
        self.num_features = self.df.shape[1]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32,
        )

        # Internal state
        self.balance = None
        self.position = None            # -1 = short, 0 = flat, 1 = long
        self.position_price = None
        self.current_step = None
        self.done = None

    # -------------------------
    # Gymnasium API
    # -------------------------
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment and return (obs, info).
        info includes: step, balance, position.
        """
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0.0
        self.current_step = self.window_size
        self.done = False

        obs = self._get_observation()
        info = {
            "step": int(self.current_step),
            "balance": float(self.balance),
            "position": int(self.position),
        }
        return obs, info

    def step(self, action: int):
        """
        Take an action (0=Hold, 1=Buy, 2=Sell) and return
        (obs, reward, terminated, truncated, info).
        """
        # Safety: bound action
        if isinstance(action, (np.ndarray, list)):
            action = int(action)
        action = int(np.clip(action, 0, 2))

        # Price at current time
        price = float(self.df.iloc[self.current_step]["Close"])
        reward = 0.0

        # --- Action logic ---
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.position_price = price
            elif self.position == -1:
                # Close short
                pnl = self.position_price - price
                reward += pnl
                self.balance += pnl
                self.position = 0
                self.position_price = 0.0

        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.position_price = price
            elif self.position == 1:
                # Close long
                pnl = price - self.position_price
                reward += pnl
                self.balance += pnl
                self.position = 0
                self.position_price = 0.0

        # --- Reward shaping: discourage frequent trading ---
        if action != 0:
            reward += -0.01

        # Advance time
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        self.done = terminated

        obs = self._get_observation()

        info = {
            "step": int(self.current_step),
            "balance": float(self.balance),
            "position": int(self.position),
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    # -------------------------
    # Helpers
    # -------------------------
    def _get_observation(self) -> np.ndarray:
        """
        Return observation window (window_size, num_features) as float32.
        """
        start = self.current_step - self.window_size
        end = self.current_step
        window = self.df.iloc[start:end].to_numpy(dtype=np.float32, copy=False)
        # 某些data源可能不足 window_size，做个保护（极少见）
        if window.shape != (self.window_size, self.num_features):
            pad_rows = self.window_size - window.shape[0]
            if pad_rows > 0:
                pad = np.repeat(window[:1], pad_rows, axis=0) if window.size else np.zeros(
                    (pad_rows, self.num_features), dtype=np.float32
                )
                window = np.concatenate([pad, window], axis=0)
        return window


