"""
trading_env.py

Purpose:
  Implements a comprehensive futures trading simulation framework for MNQ (Micro Nasdaq).
  This environment processes 1‑minute OHLC data (1379 candles per day), computes technical indicators, 
  and builds a 42-dimensional observation vector that includes:
    A. Market Data & Indicators (21 features)
    B. Current Trade Metrics (4 features)
    C. Most Recent Closed Trade Metrics (10 features)
    D. Episode Account Metrics (7 features)
       - Contracts Traded (scaled)
       - Winning Trades Contracts (scaled)
       - Losing Trades Contracts (scaled)
       - Episode Commissions (normalized)
       - Total Trades (scaled)
       - Current Episode Total $ Winnings
       - Current Episode Total $ Losses
       
  The framework uses the following specifications for MNQ:
    - Tick Size: 0.25 points; Tick Value: $0.50 per tick.
    - Commission: $1.00 per contract per side (entry and exit).
    - Margin Requirement: $100.00 per contract.
    - Equity Reserve: 20% (only 80% of net equity is available for margin).
    
  Calculation formulas:
    • Maximum Position Size = floor((net_equity * 0.8) / margin_requirement)
    • p&l_ticks = (current_price - entry_price) * direction / tick_size
    • Open P&L = p&l_ticks * (position_size * tick_value)
    • Realized P&L = (p&l_ticks * (position_size * tick_value)) - (position_size * commission * 2)
    • Total P&L = Realized P&L + Open P&L
    • Net Equity = initial_balance + Total P&L
  
  Trading rules:
    - Only one open trade is allowed.
    - If a trade is already open, a new trade is only allowed if it reverses the position.
    - Commissions are deducted at both trade entry and exit.
    - Reversal orders close the existing trade with a "Reversal" closing method.
  
  Dashboard displays include Account Metrics, Current Trade Metrics, Most Recent Closed Trade Metrics, and Market Data.
  Automated Training Metrics (e.g., episodes ended by Bankruptcy, Profit Target, or End-of-Day) are used solely for day management.
  
Author: Damian Moore
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces
import yaml
import logging
import sys
import random

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = logging.FileHandler('trading_env.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.ERROR)
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console_handler)

plt.ion()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def compute_trading_day(dt):
    """Return trading day based on time (if before 16:00, previous day)."""
    return (dt - timedelta(days=1)).date() if dt.hour < 16 else dt.date()

def compute_minutes_from_start(dt, trading_day_start):
    """Return minutes elapsed since the trading day start."""
    delta = dt - trading_day_start
    return delta.total_seconds() / 60.0

# -----------------------------------------------------------------------------
# TradingDataEnv Class Definition
# -----------------------------------------------------------------------------
class TradingDataEnv(gym.Env):
    """
    Custom Gym environment for MNQ futures trading simulation.
    
    Observation Space (42 features):
      A. Market Data & Indicators (21 features)
      B. Current Trade Metrics (4 features)
      C. Most Recent Closed Trade Metrics (10 features)
      D. Episode Account Metrics (7 features)
      
    Action Space (Discrete 4):
      0: Buy Max, 1: Sell Max, 2: Hold, 3: Close All.
    """
    metadata = {"render_modes": ["human", "fast", None], "render_fps": 30}
    
    def __init__(self, csv_path, config_path="config.yaml", render_mode="fast", initial_balance=None):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
        
        self.render_mode = render_mode
        self._figure = None
        self.ax = None
        self.dashboard_ax = None
        
        self.action_space = spaces.Discrete(4)
        self.obs_dim = 42
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        trading_config = config.get('trading', {})
        self.initial_balance = float(initial_balance if initial_balance is not None else trading_config.get('initial_balance', 500.0))
        self.target_balance = float(trading_config.get('target_balance', 750.0))
        self.min_balance = float(trading_config.get('min_balance', 100.0))
        self.margin_requirement = float(trading_config.get('margin_requirement', 100.0))
        self.commission = float(trading_config.get('commission', 1.0))
        self.atr_lower_bound = float(trading_config.get('atr_lower_bound', 0.1))
        self.atr_upper_bound = float(trading_config.get('atr_upper_bound', 5.0))
        
        market_hours = trading_config.get('market_hours', {})
        self.market_start_hour = int(market_hours.get('start_hour', 17))
        self.market_end_hour = int(market_hours.get('end_hour', 16))
        self.market_end_minute = int(market_hours.get('end_minute', 0))
        
        indicator_config = trading_config.get('indicators', {})
        self.atr_window = int(indicator_config.get('atr_window', 14))
        self.ema_window = int(indicator_config.get('ema_window', 10))
        self.ma_window = int(indicator_config.get('ma_window', 20))
        self.volatility_window = int(indicator_config.get('volatility_window', 20))
        
        success_config = config.get('success_metrics', {})
        self.MIN_EPISODES = int(success_config.get('min_episodes', 10))
        self.required_success_rate = float(success_config.get('required_success_rate', 0.90))
        self.plateau_threshold = float(success_config.get('plateau_threshold', 0.20))
        self.improvement_threshold = float(success_config.get('improvement_threshold', 0.05))
        
        # Account variables and metrics
        self.balance = self.initial_balance
        self.position = 0
        self.is_long = None
        self.entry_price = None
        self.entry_time = None      # Numeric: minutes from trading day start
        self.entry_time_str = None  # Formatted: "HH:MM"
        self.time_in_position = 0
        self.trade_history = []     # Trades in the current episode
        self.all_trade_log = []     # Full trade log
        self.current_episode_reward = 0.0
        
        # Episode metrics (for dashboard display only)
        self.episode_commissions = 0.0
        self.episode_trade_count = 0
        self.current_episode_winnings = 0.0
        self.current_episode_losses = 0.0
        
        # Day management metrics (displayed in dashboard only)
        self.day_total_episodes = 0
        self.cumulative_episode_count = 0
        self.day_success_count = 0
        self.no_improvement_count = 0
        self.termination_counts = {"profit_target": 0, "bankrupt": 0, "end_of_day": 0}
        
        self.days_mastered = 0
        self.last_action_result = ""
        
        # Contract constants
        self.tick_size = 0.25       # points
        self.tick_value = 0.50      # dollars per tick (per contract)
        self.lookback_window = 20
        
        self.data = self._load_data(csv_path)
        self._validate_data_quality(self.data)
        self.data = self._calculate_indicators(self.data)
        self.feature_means, self.feature_stds = self._calculate_feature_scaling(self.data)
        
        self.trading_days = sorted(self.data['TradingDay'].unique())
        if not self.trading_days:
            raise ValueError("No trading days found in dataset.")
        self.trading_day_ordinals = [d.toordinal() for d in self.trading_days]
        self.min_trading_day_ordinal = min(self.trading_day_ordinals)
        self.max_trading_day_ordinal = max(self.trading_day_ordinals)
        
        self.current_day = random.choice(self.trading_days)
        self.day_data = self._get_day_data(self.current_day)
        self.day_data['ATR_14'] = np.clip(self.day_data['ATR_14'], self.atr_lower_bound, self.atr_upper_bound)
        self.current_step = 0
        
        if self.render_mode == "human":
            self._init_rendering()
    
    # ------------------------ Data Loading & Processing ------------------------
    def _load_data(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            for col in ['Date', 'Open', 'High', 'Low', 'Close']:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y %H:%M:%S")
            df.sort_values('Date', inplace=True)
            df.reset_index(drop=True, inplace=True)
            df['TradingDay'] = df['Date'].apply(compute_trading_day)
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _validate_data_quality(self, df):
        for col in ['Open', 'High', 'Low', 'Close']:
            if df[col].isnull().any():
                logger.warning(f"Missing values in {col}; filling with zeros.")
                df[col].fillna(0, inplace=True)
            if (df[col] <= 0).any():
                raise ValueError(f"Non-positive values detected in {col}.")
    
    def _calculate_indicators(self, df):
        try:
            df = df.copy()
            df['returns'] = df['Close'].pct_change().fillna(0)
            df['volatility'] = df['returns'].rolling(window=self.volatility_window, min_periods=1).std().fillna(0)
            df['EMA_10'] = df['Close'].ewm(span=self.ema_window, adjust=False, min_periods=1).mean().fillna(0)
            df['MA_20'] = df['Close'].rolling(window=self.ma_window, min_periods=1).mean().fillna(0)
            df['STD_20'] = df['Close'].rolling(window=self.ma_window, min_periods=1).std().fillna(0)
            df['BB_upper'] = df['MA_20'] + 2 * df['STD_20']
            df['BB_lower'] = df['MA_20'] - 2 * df['STD_20']
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR_14'] = tr.rolling(window=self.atr_window, min_periods=1).mean().fillna(0)
            df['momentum'] = df['returns'].rolling(window=self.lookback_window, min_periods=1).mean().fillna(0)
            df.drop(columns=['returns'], inplace=True)
            return df
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            raise
    
    def _calculate_feature_scaling(self, df):
        features = ['Close', 'EMA_10', 'BB_upper', 'MA_20', 'BB_lower', 'ATR_14', 'Open', 'High', 'Low']
        means = {}
        stds = {}
        for feature in features:
            means[feature] = float(df[feature].mean())
            std_val = df[feature].std()
            stds[feature] = float(std_val) if std_val > 0 else 1.0
        return means, stds
    
    def _get_day_data(self, trading_day):
        day_data = self.data[self.data['TradingDay'] == trading_day].copy()
        if day_data.empty:
            raise ValueError(f"No data for trading day: {trading_day}")
        day_data.reset_index(drop=True, inplace=True)
        return day_data
    
    def _get_current_candle_time(self):
        current_dt = self.day_data.iloc[self.current_step]['Date']
        trading_day = self.current_day
        trading_day_start = datetime.combine(trading_day, datetime.min.time()) - timedelta(hours=7)
        minutes = compute_minutes_from_start(current_dt, trading_day_start)
        return minutes  # For observation, normalized later by dividing by 1379
    
    # ------------------------ Observation Construction ------------------------
    def _get_observation(self):
        try:
            idx = min(self.current_step, len(self.day_data) - 1)
            window_data = self.day_data.iloc[max(0, idx - self.lookback_window + 1): idx + 1]
            current_data = window_data.iloc[-1]
            
            obs = []
            # A. Market Data & Indicators (21 features)
            obs.append(self.position / 10.0)
            val = float(current_data.get('EMA_10', 0))
            obs.append(np.clip((val - self.feature_means['EMA_10']) / self.feature_stds['EMA_10'], -10, 10))
            direction = 1 if self.is_long else (-1 if self.is_long is False else 0)
            obs.append(direction)
            val = float(current_data.get('BB_upper', 0))
            obs.append(np.clip((val - self.feature_means['BB_upper']) / self.feature_stds['BB_upper'], -10, 10))
            pos_value = (self.position * self.tick_value) if self.position else 0.0
            obs.append(pos_value)
            val = float(current_data.get('MA_20', 0))
            obs.append(np.clip((val - self.feature_means['MA_20']) / self.feature_stds['MA_20'], -10, 10))
            total_steps = len(self.day_data)
            obs.append(self.current_step / total_steps if total_steps > 0 else 0)
            val = float(current_data.get('BB_lower', 0))
            obs.append(np.clip((val - self.feature_means['BB_lower']) / self.feature_stds['BB_lower'], -10, 10))
            total_pl = self._calculate_realized_pnl() + self._calculate_open_pnl()
            net_equity = self.initial_balance + total_pl
            obs.append(net_equity / self.initial_balance)
            val = float(current_data.get('Close', 0))
            obs.append(np.clip((val - self.feature_means['Close']) / self.feature_stds['Close'], -10, 10))
            obs.append(window_data['volatility'].mean())
            obs.append(window_data['momentum'].mean())
            obs.append(float(current_data.get('ATR_14', 0)))
            obs.append(total_pl / self.initial_balance)
            realized = self._calculate_realized_pnl()
            obs.append(realized / self.initial_balance)
            open_pl = self._calculate_open_pnl()
            obs.append(open_pl / self.initial_balance)
            pl_list = [trade.get('profit', 0) for trade in self.trade_history if trade.get('action') == 'Close']
            avg_pl = np.mean(pl_list) if pl_list else 0.0
            obs.append(avg_pl / self.initial_balance)
            val = float(current_data.get('High', 0))
            obs.append(np.clip((val - self.feature_means['High']) / self.feature_stds['High'], -10, 10))
            val = float(current_data.get('Low', 0))
            obs.append(np.clip((val - self.feature_means['Low']) / self.feature_stds['Low'], -10, 10))
            val = float(current_data.get('Open', 0))
            obs.append(np.clip((val - self.feature_means['Open']) / self.feature_stds['Open'], -10, 10))
            current_candle_time = self._get_current_candle_time()
            obs.append(current_candle_time / 1379.0)
            
            # B. Current Trade Metrics (4 features)
            if self.entry_price is not None:
                norm_entry = (self.entry_price - self.feature_means['Close']) / self.feature_stds['Close']
                obs.append(np.clip(norm_entry, -10, 10))
                current_entry_time = getattr(self, 'entry_time', 0)
                obs.append(current_entry_time / 1379.0)
                obs.append(self.position / 10.0)
                # Open P&L is calculated as:
                # ( (current_close - entry_price)/tick_size * (position * tick_value) )
                current_close = float(current_data.get('Close', 0))
                if self.is_long:
                    tick_diff = (current_close - self.entry_price) / self.tick_size
                else:
                    tick_diff = (self.entry_price - current_close) / self.tick_size
                open_pnl = tick_diff * (self.position * self.tick_value)
                obs.append(open_pnl / self.initial_balance)
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])
            
            # C. Most Recent Closed Trade Metrics (10 features)
            recent_trade = None
            for trade in reversed(self.all_trade_log):
                if trade.get('action') == 'Close':
                    recent_trade = trade
                    break
            if recent_trade:
                norm_entry = (recent_trade.get('entry_price', 0) - self.feature_means['Close']) / self.feature_stds['Close']
                norm_exit = (recent_trade.get('exit_price', 0) - self.feature_means['Close']) / self.feature_stds['Close']
                obs.append(np.clip(norm_entry, -10, 10))
                obs.append(np.clip(norm_exit, -10, 10))
                ticks = recent_trade.get('ticks', 0)
                obs.append((ticks * self.tick_value) / self.initial_balance)
                obs.append(recent_trade.get('profit', 0) / self.initial_balance)
                obs.append(recent_trade.get('size', 0) / 10.0)
                pos_val = recent_trade.get('size', 0) * self.tick_value
                obs.append(pos_val)
                trade_direction = 1 if recent_trade.get('trade_type', "Buy") == "Buy" else -1
                obs.append(trade_direction)
                obs.append(recent_trade.get('commission', 0) / self.initial_balance)
                entry_time = recent_trade.get('entry_time', 0)
                exit_time = recent_trade.get('exit_time', 0)
                obs.append(entry_time / 1379.0)
                obs.append(exit_time / 1379.0)
            else:
                obs.extend([0.0]*10)
            
            # D. Episode Account Metrics (7 features)
            total_contracts = sum(trade.get('size', 0) for trade in self.trade_history if trade.get('action') in ['Buy', 'Sell'])
            obs.append(total_contracts / 10.0)
            winning_contracts = sum(trade.get('size', 0) for trade in self.trade_history if trade.get('action')=='Close' and trade.get('profit', 0) > 0)
            obs.append(winning_contracts / 10.0)
            losing_contracts = sum(trade.get('size', 0) for trade in self.trade_history if trade.get('action')=='Close' and trade.get('profit', 0) <= 0)
            obs.append(losing_contracts / 10.0)
            obs.append(self.episode_commissions / self.initial_balance)
            obs.append(self.episode_trade_count / 10.0)
            total_winnings = sum(trade.get('profit', 0) for trade in self.trade_history if trade.get('action')=='Close' and trade.get('profit', 0) > 0)
            total_losses = sum(trade.get('profit', 0) for trade in self.trade_history if trade.get('action')=='Close' and trade.get('profit', 0) <= 0)
            obs.append(total_winnings / self.initial_balance)
            obs.append(total_losses / self.initial_balance)
            
            if len(obs) != self.obs_dim:
                logger.error(f"Observation vector length mismatch: expected {self.obs_dim}, got {len(obs)}")
                if len(obs) > self.obs_dim:
                    obs = obs[:self.obs_dim]
                else:
                    obs += [0.0] * (self.obs_dim - len(obs))
            
            obs_array = np.array(obs, dtype=np.float32)
            return np.nan_to_num(obs_array)
        except Exception as e:
            logger.error(f"Error constructing observation: {e}")
            return np.zeros(self.obs_dim, dtype=np.float32)
    
    # ------------------------ P&L Calculations ------------------------
    def _calculate_open_pnl(self):
        if self.position and self.entry_price is not None:
            current_row = self.day_data.iloc[self.current_step]
            current_close = float(current_row.get('Close', 0))
            if self.is_long:
                tick_diff = (current_close - self.entry_price) / self.tick_size
            else:
                tick_diff = (self.entry_price - current_close) / self.tick_size
            return tick_diff * (self.position * self.tick_value)
        return 0.0

    def _calculate_realized_pnl(self):
        realized = 0.0
        for trade in self.trade_history:
            if trade.get('action') == 'Close':
                realized += trade.get('profit', 0)
        return realized
    
    # ------------------------ Episode & Day Management ------------------------
    def update_episode_counters(self, episode_steps, trade_count):
        self.cumulative_episode_count += 1
        self.day_total_episodes += 1
        if self.balance >= self.target_balance:
            self.day_success_count += 1
        else:
            self.no_improvement_count += 1
    
    def check_day_mastery(self):
        if self.day_total_episodes < self.MIN_EPISODES:
            return False
        success_rate = self.day_success_count / self.day_total_episodes
        if success_rate >= self.required_success_rate and self.no_improvement_count >= int(self.day_total_episodes * self.plateau_threshold):
            return True
        return False
    
    def reset_episode(self):
        if self.check_day_mastery():
            self.days_mastered += 1
            self.day_total_episodes = 0
            self.day_success_count = 0
            self.no_improvement_count = 0
            self.termination_counts = {"profit_target": 0, "bankrupt": 0, "end_of_day": 0}
            available_days = [d for d in self.trading_days if d != self.current_day]
            if available_days:
                self.current_day = random.choice(available_days)
            else:
                logger.info("Only one trading day available; continuing with same day.")
        self.episode_commissions = 0.0
        self.episode_trade_count = 0
        self.current_episode_winnings = 0.0
        self.current_episode_losses = 0.0
    
    # ------------------------ Environment Reset ------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.balance = self.initial_balance
        self.position = 0
        self.is_long = None
        self.entry_price = None
        self.time_in_position = 0
        self.trade_history = []
        self.current_episode_reward = 0.0
        self.current_step = 0
        self.day_data = self._get_day_data(self.current_day)
        self.day_data['ATR_14'] = np.clip(self.day_data['ATR_14'], self.atr_lower_bound, self.atr_upper_bound)
        return self._get_observation(), {}
    
    # ------------------------ Step Function ------------------------
    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        current_data = self.day_data.iloc[self.current_step]
        current_price = float(current_data['Close'])
        reward = 0.0
        done = False
        
        total_pl = self._calculate_realized_pnl() + self._calculate_open_pnl()
        net_equity = self.initial_balance + total_pl
        max_contracts = int((net_equity * 0.8) // self.margin_requirement)
        
        self.last_action_result = ""
        
        # If a trade is open, allow only reversal if the action is in the opposite direction.
        if action in [0, 1]:
            if self.position != 0:
                if (action == 0 and not self.is_long) or (action == 1 and self.is_long):
                    raw_ticks = int(abs(current_price - self.entry_price) / self.tick_size)
                    if self.is_long:
                        profit = raw_ticks * (self.position * self.tick_value)
                    else:
                        profit = raw_ticks * (self.position * self.tick_value)
                    if (self.is_long and current_price < self.entry_price) or (not self.is_long and current_price > self.entry_price):
                        profit = -profit
                    total_comm = self.position * self.commission * 2
                    exit_time = self._get_current_candle_time()
                    exit_time_str = current_data['Date'].strftime("%H:%M")
                    self.balance += profit - (self.position * self.commission)
                    reversal_trade = {
                        'time': current_data['Date'],
                        'action': 'Close',
                        'exit_price': current_price,
                        'size': self.position,
                        'profit': profit,
                        'commission': total_comm,
                        'ticks': raw_ticks * 4,
                        'time_in_trade': self.time_in_position,
                        'closing_method': "Reversal",
                        'entry_time': getattr(self, 'entry_time', 0),
                        'exit_time': exit_time,
                        'entry_time_str': getattr(self, 'entry_time_str', "N/A"),
                        'exit_time_str': exit_time_str,
                        'trade_type': "Buy" if not self.is_long else "Sell"
                    }
                    self.trade_history.append(reversal_trade)
                    self.all_trade_log.append(reversal_trade)
                    if profit > 0:
                        self.current_episode_winnings += profit
                    else:
                        self.current_episode_losses += profit
                    self.last_action_result = f"{'Buy' if action==0 else 'Sell'} Max: Reversal Close Executed"
                    self.position = 0
                    self.is_long = None
                    self.entry_price = None
                    self.time_in_position = 0
                    reward = -0.1
                    done = self.current_step >= len(self.day_data) - 1
                    obs = self._get_observation()
                    info = {
                        'balance': net_equity,
                        'position': 0,
                        'position_direction': "Flat",
                        'trades': len(self.trade_history),
                        'current_price': current_price,
                        'step': self.current_step,
                        'unrealized_pnl': 0.0,
                        'last_action': self.last_action_result
                    }
                    return obs, reward, done, False, info
                else:
                    self.last_action_result = f"{'Buy' if action==0 else 'Sell'} Max: Order Rejected – Trade Already Open"
                    reward = -0.1
                    obs = self._get_observation()
                    info = {
                        'balance': net_equity,
                        'position': self.position,
                        'position_direction': "Long" if self.is_long else ("Short" if self.is_long is False else "Flat"),
                        'trades': len(self.trade_history),
                        'current_price': current_price,
                        'step': self.current_step,
                        'unrealized_pnl': self._calculate_open_pnl(),
                        'last_action': self.last_action_result
                    }
                    return obs, reward, done, False, info
        
        if action in [0, 1] and max_contracts < 1:
            self.last_action_result = f"{'Buy' if action==0 else 'Sell'} Max: Order Rejected – Insufficient Margin"
            reward = -0.1
            obs = self._get_observation()
            info = {
                'balance': net_equity,
                'position': 0,
                'position_direction': "Flat",
                'trades': len(self.trade_history),
                'current_price': current_price,
                'step': self.current_step,
                'unrealized_pnl': 0.0,
                'last_action': self.last_action_result
            }
            return obs, reward, done, False, info
        
        if action == 0:  # Buy Max
            self.position = max_contracts
            self.is_long = True
            self.entry_price = current_price
            self.entry_time = self._get_current_candle_time()
            self.entry_time_str = current_data['Date'].strftime("%H:%M")
            self.time_in_position = 0
            cost = max_contracts * self.commission
            self.balance -= cost
            self.episode_commissions += cost
            self.episode_trade_count += 1
            trade = {
                'time': current_data['Date'],
                'action': 'Buy',
                'entry_price': current_price,
                'size': max_contracts,
                'commission': cost,
                'closing_method': "N/A",
                'entry_time': self.entry_time,
                'entry_time_str': self.entry_time_str,
                'trade_type': "Buy"
            }
            self.trade_history.append(trade)
            self.all_trade_log.append(trade)
            self.last_action_result = "Buy Max: Entry Executed"
        elif action == 1:  # Sell Max
            self.position = max_contracts
            self.is_long = False
            self.entry_price = current_price
            self.entry_time = self._get_current_candle_time()
            self.entry_time_str = current_data['Date'].strftime("%H:%M")
            self.time_in_position = 0
            cost = max_contracts * self.commission
            self.balance -= cost
            self.episode_commissions += cost
            self.episode_trade_count += 1
            trade = {
                'time': current_data['Date'],
                'action': 'Sell',
                'entry_price': current_price,
                'size': max_contracts,
                'commission': cost,
                'closing_method': "N/A",
                'entry_time': self.entry_time,
                'entry_time_str': self.entry_time_str,
                'trade_type': "Sell"
            }
            self.trade_history.append(trade)
            self.all_trade_log.append(trade)
            self.last_action_result = "Sell Max: Entry Executed"
        elif action == 3:  # Close All
            if self.position == 0:
                self.last_action_result = "Close All: Order Rejected – No Open Trade"
                reward = -0.1
                obs = self._get_observation()
                info = {
                    'balance': net_equity,
                    'position': 0,
                    'position_direction': "Flat",
                    'trades': len(self.trade_history),
                    'current_price': current_price,
                    'step': self.current_step,
                    'unrealized_pnl': 0.0,
                    'last_action': self.last_action_result
                }
                return obs, reward, done, False, info
            else:
                if self.is_long:
                    raw_ticks = int(abs(current_price - self.entry_price) / self.tick_size)
                    profit = raw_ticks * (self.position * self.tick_value)
                    if current_price < self.entry_price:
                        profit = -profit
                else:
                    raw_ticks = int(abs(current_price - self.entry_price) / self.tick_size)
                    profit = raw_ticks * (self.position * self.tick_value)
                    if current_price > self.entry_price:
                        profit = -profit
                total_comm = self.position * self.commission * 2
                tick_count = raw_ticks * 4
                exit_time = self._get_current_candle_time()
                exit_time_str = current_data['Date'].strftime("%H:%M")
                self.balance += profit - (self.position * self.commission)
                trade = {
                    'time': current_data['Date'],
                    'action': 'Close',
                    'exit_price': current_price,
                    'size': self.position,
                    'profit': profit,
                    'commission': total_comm,
                    'ticks': tick_count,
                    'time_in_trade': self.time_in_position,
                    'closing_method': "Normal",
                    'entry_time': getattr(self, 'entry_time', 0),
                    'exit_time': exit_time,
                    'entry_time_str': getattr(self, 'entry_time_str', "N/A"),
                    'exit_time_str': exit_time_str,
                    'trade_type': "Buy" if self.is_long else "Sell"
                }
                self.trade_history.append(trade)
                self.all_trade_log.append(trade)
                self.last_action_result = "Close All: Trade Closed"
                self.position = 0
                self.is_long = None
                self.entry_price = None
                self.time_in_position = 0
        if action == 2:
            self.last_action_result = "Hold: No Operation"
        
        # Reward computation
        if self.position != 0:
            self.time_in_position += 1
            open_pl = self._calculate_open_pnl()
            # Reward penalizes open losses; positive open P&L is not rewarded until trade closes.
            reward = (min(0, open_pl) / self.initial_balance) * 10
        else:
            open_pl = self._calculate_open_pnl()
            net_equity = self.initial_balance + self._calculate_realized_pnl() + open_pl
            reward = ((net_equity / self.initial_balance) - 1) * 10
            if net_equity >= self.target_balance:
                reward += 50  # bonus reward for reaching the profit target
        
        self.current_episode_reward += reward
        
        if net_equity <= self.min_balance:
            reward = -1.0
            done = True
            self.termination_counts["bankrupt"] += 1
            self.last_action_result += " | Account Bankrupted"
        elif net_equity >= self.target_balance:
            done = True
            self.termination_counts["profit_target"] += 1
            self.last_action_result += " | Profit Target Reached"
        elif self.current_step >= len(self.day_data) - 1:
            done = True
            self.termination_counts["end_of_day"] += 1
            self.last_action_result += " | End of Day Reached"
        
        if done:
            self.update_episode_counters(self.current_step, self.episode_trade_count)
            self.reset_episode()
        
        info = {
            'balance': net_equity,
            'position': self.position,
            'position_direction': "Long" if self.is_long else ("Short" if self.is_long is False else "Flat"),
            'trades': len(self.trade_history),
            'current_price': current_price,
            'step': self.current_step,
            'unrealized_pnl': self._calculate_open_pnl(),
            'last_action': self.last_action_result
        }
        
        if not done:
            self.current_step += 1
            obs = self._get_observation()
        else:
            obs = self._get_observation()
        return obs, reward, done, False, info
    
    # ------------------------ Rendering ------------------------
    def render(self):
        if self.render_mode != "human":
            return
        try:
            self.ax.cla()
            self.dashboard_ax.cla()
            
            day_data = self.day_data.copy()
            day_data["DateNum"] = day_data["Date"].apply(mdates.date2num)
            current_time = self.day_data.iloc[self.current_step]["Date"]
            half_window = timedelta(minutes=22.5)
            window_start = current_time - half_window
            window_end = current_time + half_window
            if window_start < self.day_data.iloc[0]["Date"]:
                window_start = self.day_data.iloc[0]["Date"]
                window_end = window_start + timedelta(minutes=45)
            if window_end > self.day_data.iloc[-1]["Date"]:
                window_end = self.day_data.iloc[-1]["Date"]
                window_start = window_end - timedelta(minutes=45)
            visible_data = day_data[(day_data["Date"] >= window_start) & (day_data["Date"] <= window_end)]
            
            candle_width = 0.0005
            for _, row in visible_data.iterrows():
                color = 'green' if row['Close'] >= row['Open'] else 'red'
                lower = min(row['Open'], row['Close'])
                height = abs(row['Close'] - row['Open'])
                rect = Rectangle((row["DateNum"] - candle_width/2, lower), candle_width, height, color=color)
                self.ax.add_patch(rect)
                self.ax.plot([row["DateNum"], row["DateNum"]], [row["Low"], row["High"]], color=color, linewidth=1)
            
            self.ax.set_xlim(visible_data["DateNum"].min()-candle_width, visible_data["DateNum"].max()+candle_width)
            self.ax.xaxis_date()
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            self.ax.set_xlabel("Time (HH:MM)")
            self.ax.set_ylabel("Price")
            self.ax.grid(True, alpha=0.3)
            
            current_price = float(self.day_data.iloc[self.current_step]['Close'])
            self.ax.set_ylim(current_price - 50, current_price + 50)
            
            self.ax.plot(day_data["DateNum"], day_data["EMA_10"], label="EMA", color="orange", linewidth=1)
            self.ax.plot(day_data["DateNum"], day_data["MA_20"], label="MA", color="blue", linewidth=1)
            self.ax.plot(day_data["DateNum"], day_data["BB_upper"], label="BB Upper", color="gray", linestyle="--", linewidth=1)
            self.ax.plot(day_data["DateNum"], day_data["BB_lower"], label="BB Lower", color="gray", linestyle="--", linewidth=1)
            self.ax.legend(loc="upper left", fontsize=8)
            
            span = candle_width * 3
            for trade in self.all_trade_log:
                trade_time = pd.to_datetime(trade['time'])
                date_num = mdates.date2num(trade_time)
                if trade['action'] in ['Buy', 'Sell']:
                    marker_color = 'green' if trade['action'] == 'Buy' else 'red'
                    self.ax.hlines(y=trade.get('entry_price', 0), xmin=date_num - span/2, xmax=date_num + span/2, colors=marker_color, linewidth=2)
                elif trade['action'] == 'Close':
                    marker_color = 'blue' if trade.get('closing_method', "Normal") == "Reversal" else 'black'
                    self.ax.hlines(y=trade.get('exit_price', 0), xmin=date_num - span/2, xmax=date_num + span/2, colors=marker_color, linewidth=2)
                    annotation = (f"PnL: ${trade.get('profit',0):.2f}\n"
                                  f"Ticks: {trade.get('ticks',0)}\n"
                                  f"Comm: ${trade.get('commission',0):.2f}")
                    self.ax.text(date_num, trade.get('exit_price', 0) + 0.01 * current_price, annotation,
                                 color='black', fontsize=8, ha='center', va='bottom')
            
            current_data = self.day_data.iloc[self.current_step]
            open_pl = self._calculate_open_pnl()
            net_equity = self.initial_balance + self._calculate_realized_pnl() + open_pl
            total_pl = net_equity - self.initial_balance
            
            dashboard_lines = []
            dashboard_lines.append("=== Automated Training Metrics ===")
            dashboard_lines.append(f"Days Mastered: {self.days_mastered}")
            dashboard_lines.append(f"Cumulative Episodes: {self.cumulative_episode_count}")
            dashboard_lines.append(f"Current Day Episodes: {self.day_total_episodes}")
            dashboard_lines.append(f"Bankrupt Episodes: {self.termination_counts['bankrupt']}")
            dashboard_lines.append(f"Profit Target Episodes: {self.termination_counts['profit_target']}")
            dashboard_lines.append(f"End-of-Day Episodes: {self.termination_counts['end_of_day']}")
            dashboard_lines.append(f"Current Step: {self.current_step}")
            dashboard_lines.append(f"Episode Trades: {self.episode_trade_count}")
            dashboard_lines.append(f"Last Action: {self.last_action_result}")
            dashboard_lines.append("=== Account Metrics ===")
            dashboard_lines.append(f"Net Equity: ${net_equity:.2f}")
            dashboard_lines.append(f"Total P&L: ${total_pl:.2f}")
            dashboard_lines.append(f"Realized P&L: ${self._calculate_realized_pnl():.2f}")
            dashboard_lines.append(f"Open P&L: ${open_pl:.2f}")
            dashboard_lines.append(f"Episode Commissions: ${self.episode_commissions:.2f}")
            total_contracts = sum(trade.get('size', 0) for trade in self.trade_history if trade.get('action') in ['Buy', 'Sell'])
            dashboard_lines.append(f"Contracts Traded: {total_contracts}")
            winning_contracts = sum(trade.get('size', 0) for trade in self.trade_history if trade.get('action')=='Close' and trade.get('profit',0)>0)
            losing_contracts = sum(trade.get('size', 0) for trade in self.trade_history if trade.get('action')=='Close' and trade.get('profit',0)<=0)
            dashboard_lines.append(f"Winning Trades: {winning_contracts}")
            dashboard_lines.append(f"Losing Trades: {losing_contracts}")
            dashboard_lines.append(f"Total $ Winnings: ${sum(trade.get('profit',0) for trade in self.trade_history if trade.get('action')=='Close' and trade.get('profit',0)>0):.2f}")
            dashboard_lines.append(f"Total $ Losses: ${sum(trade.get('profit',0) for trade in self.trade_history if trade.get('action')=='Close' and trade.get('profit',0)<=0):.2f}")
            dashboard_lines.append("=== Current Trade Metrics ===")
            current_trade_entry = self.entry_price if self.entry_price is not None else "N/A"
            current_trade_time = getattr(self, 'entry_time_str', "N/A")
            current_trade_ticks = int(abs(current_price - self.entry_price) / self.tick_size)*4 if self.entry_price is not None else 0
            dashboard_lines.append(f"Entry Price: {current_trade_entry}")
            dashboard_lines.append(f"Entry Time: {current_trade_time}")
            dashboard_lines.append(f"P&L Ticks: {current_trade_ticks}")
            dashboard_lines.append(f"Open P&L: ${open_pl:.2f}")
            dashboard_lines.append(f"Position Size: {self.position}")
            dashboard_lines.append(f"Position Value: ${self.position * self.tick_value:.2f}")
            dashboard_lines.append(f"Position Direction: {'Long' if self.is_long else ('Short' if self.is_long is False else 'Flat')}")
            dashboard_lines.append("=== Most Recent Trade Metrics ===")
            recent_trade = None
            for trade in reversed(self.all_trade_log):
                if trade.get('action') == 'Close':
                    recent_trade = trade
                    break
            if recent_trade:
                dashboard_lines.append(f"Entry Price: {recent_trade.get('entry_price','N/A')}")
                dashboard_lines.append(f"Exit Price: {recent_trade.get('exit_price','N/A')}")
                dashboard_lines.append(f"P&L Ticks: {recent_trade.get('ticks',0)}")
                dashboard_lines.append(f"Realized P&L: ${recent_trade.get('profit',0):.2f}")
                dashboard_lines.append(f"Position Size: {recent_trade.get('size',0)}")
                dashboard_lines.append(f"Position Value: ${recent_trade.get('size',0) * self.tick_value:.2f}")
                direction = "Long" if recent_trade.get('trade_type',"Buy") == "Buy" else "Short"
                dashboard_lines.append(f"Position Direction: {direction}")
                dashboard_lines.append(f"Commissions: ${recent_trade.get('commission',0):.2f}")
                dashboard_lines.append(f"Entry Time: {recent_trade.get('entry_time_str','N/A')}")
                dashboard_lines.append(f"Exit Time: {recent_trade.get('exit_time_str','N/A')}")
                dashboard_lines.append(f"Closing Method: {recent_trade.get('closing_method','Normal')}")
            else:
                dashboard_lines.append("No closed trades yet.")
            dashboard_lines.append("=== Market Data ===")
            dashboard_lines.append(f"Raw EMA (10): {current_data.get('EMA_10',0):.2f}")
            dashboard_lines.append(f"Raw MA (20): {current_data.get('MA_20',0):.2f}")
            dashboard_lines.append(f"Raw ATR (14): {current_data.get('ATR_14',0):.2f}")
            dashboard_lines.append(f"Trading Day: {self.current_day}")
            
            dashboard_text = "\n".join(dashboard_lines)
            self.dashboard_ax.clear()
            self.dashboard_ax.text(0.05, 0.95, dashboard_text, verticalalignment='top',
                                   horizontalalignment='left', fontsize=10, color='black',
                                   transform=self.dashboard_ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
            self.dashboard_ax.axis('off')
            
            self._figure.canvas.draw()
            plt.pause(0.001)
        except Exception as e:
            logger.error(f"Error during rendering: {e}")
    
    def _init_rendering(self):
        try:
            plt.close('all')
            self._figure, (self.ax, self.dashboard_ax) = plt.subplots(1, 2, figsize=(14, 6),
                                                                      gridspec_kw={'width_ratios': [3, 1]})
            self.ax.grid(True, alpha=0.3)
            self.ax.set_title("Live Trading Simulation")
            self.ax.set_xlabel("Time (HH:MM)")
            self.ax.set_ylabel("Price")
            self.dashboard_ax.axis('off')
            self._figure.autofmt_xdate()
            self._figure.show()
            plt.pause(0.001)
        except Exception as e:
            logger.error(f"Rendering initialization failed: {e}")
            self.render_mode = "fast"
            raise RuntimeError(f"Failed to initialize rendering: {e}")

# End of trading_env.py
