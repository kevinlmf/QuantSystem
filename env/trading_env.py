import gymnasium as gym
import numpy as np

class TradingEnv(gym.Env):
    """
    A simple trading environment with discrete actions: Hold, Buy, Sell.
    Compatible with Gymnasium and Stable-Baselines3.
    """
    def __init__(self, data, window_size=10, initial_balance=1000):
        super().__init__()
        self.df = data.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = gym.spaces.Discrete(3)

        # Observation space: (window_size, number of features [e.g., OHLCV])
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, self.df.shape[1]),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.

        Returns:
            obs (np.ndarray): Initial observation
            info (dict): Additional info (empty in this case)
        """
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0  # -1 = short, 0 = flat, 1 = long
        self.position_price = 0
        self.current_step = self.window_size
        self.done = False

        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        """
        Get the current observation window.

        Returns:
            np.ndarray: Observation array of shape (window_size, num_features)
        """
        return self.df.iloc[self.current_step - self.window_size:self.current_step].values

    def step(self, action):
        """
        Take an action in the environment.

        Args:
            action (int): Action to take (0=Hold, 1=Buy, 2=Sell)

        Returns:
            obs (np.ndarray): Next observation
            reward (float): Reward signal
            done (bool): Whether the episode has ended
            truncated (bool): Whether truncated (not used here)
            info (dict): Additional info (empty in this case)
        """
        price = self.df.iloc[self.current_step]['Close']
        reward = 0

        # --- Action logic ---
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.position_price = price
            elif self.position == -1:
                reward = self.position_price - price
                self.balance += reward
                self.position = 0

        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.position_price = price
            elif self.position == 1:
                reward = price - self.position_price
                self.balance += reward
                self.position = 0

        # --- Reward shaping: discourage frequent trading ---
        shaping_penalty = -0.01 if action != 0 else 0
        reward += shaping_penalty

        # --- Advance time step ---
        self.current_step += 1
        self.done = self.current_step >= len(self.df) - 1

        obs = self._get_observation()
        return obs, reward, self.done, False, {}

