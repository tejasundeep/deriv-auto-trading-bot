import abc
from typing import List, Optional, Dict, Any, Tuple

from ml_engine import HybridMLStrategy
import pandas as pd
import ta

# ==========================================
# 1. Trading Strategies (ML Only)
# ==========================================

class BaseStrategy(abc.ABC):
    """Abstract Base Class for all trading strategies."""
    def __init__(self, name: str, description: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.config = config or {}

    @abc.abstractmethod
    def analyze(self, data: Any, timeframe: int = 1) -> Tuple[Optional[str], Optional[int]]:
        """
        Analyzes price series and returns a tuple of (signal, expiry_minutes).
        signal: "CALL", "PUT", or None.
        expiry_minutes: The recommended trade duration for this signal, or None.
        """
        pass

    @abc.abstractmethod
    def get_indicators(self, data: Any) -> Dict[str, Any]:
        """
        Returns strategy telemetry for the dashboard.
        """
        pass

class TechnicalIndicatorStrategy(BaseStrategy):
    """
    Technical Indicator Strategy based on explicit user rules.
    Dynamically adjusts parameters based on the timeframe.
    """
    # Timeframe lookup table (Timeframe -> Dict of parameters)
    TF_PARAMS = {
        15: {"ema_fast": 20, "ema_slow": 50,  "rsi_buy": 58, "rsi_sell": 42, "macd_f": 12, "macd_s": 26, "macd_sig": 9, "adx": 22, "stoch_k": 14, "stoch_d": 3, "expiry": 30},
        20: {"ema_fast": 20, "ema_slow": 50,  "rsi_buy": 58, "rsi_sell": 42, "macd_f": 12, "macd_s": 26, "macd_sig": 9, "adx": 22, "stoch_k": 14, "stoch_d": 3, "expiry": 40},
        25: {"ema_fast": 30, "ema_slow": 60,  "rsi_buy": 58, "rsi_sell": 42, "macd_f": 12, "macd_s": 26, "macd_sig": 9, "adx": 23, "stoch_k": 14, "stoch_d": 3, "expiry": 50},
        30: {"ema_fast": 30, "ema_slow": 70,  "rsi_buy": 58, "rsi_sell": 42, "macd_f": 12, "macd_s": 26, "macd_sig": 9, "adx": 23, "stoch_k": 14, "stoch_d": 3, "expiry": 60},
        40: {"ema_fast": 50, "ema_slow": 100, "rsi_buy": 58, "rsi_sell": 42, "macd_f": 12, "macd_s": 26, "macd_sig": 9, "adx": 24, "stoch_k": 14, "stoch_d": 3, "expiry": 80},
        45: {"ema_fast": 50, "ema_slow": 100, "rsi_buy": 58, "rsi_sell": 42, "macd_f": 12, "macd_s": 26, "macd_sig": 9, "adx": 24, "stoch_k": 14, "stoch_d": 3, "expiry": 90},
        50: {"ema_fast": 50, "ema_slow": 120, "rsi_buy": 58, "rsi_sell": 42, "macd_f": 12, "macd_s": 26, "macd_sig": 9, "adx": 25, "stoch_k": 14, "stoch_d": 3, "expiry": 100},
        55: {"ema_fast": 50, "ema_slow": 120, "rsi_buy": 58, "rsi_sell": 42, "macd_f": 12, "macd_s": 26, "macd_sig": 9, "adx": 25, "stoch_k": 14, "stoch_d": 3, "expiry": 110},
    }

    def __init__(self, config=None):
        super().__init__("Technical Indicators", "Dynamic Timeframe Multi-Indicator Rules", config)
        self.last_indicators = {}
        
    def analyze(self, data: List[Dict[str, float]], timeframe: int = 15) -> Tuple[Optional[str], Optional[int]]:
        # Fallback to 15m if timeframe is not in table (though it always should be)
        params = self.TF_PARAMS.get(timeframe, self.TF_PARAMS[15])
        
        if not data or len(data) < params["ema_slow"] + 20: 
            return None, None
            
        df = pd.DataFrame(data)
        
        # Calculate Indicators dynamically based on timeframe params
        df['EMA_Fast'] = ta.trend.ema_indicator(df['close'], window=params["ema_fast"])
        df['EMA_Slow'] = ta.trend.ema_indicator(df['close'], window=params["ema_slow"])
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        
        macd = ta.trend.MACD(df['close'], window_fast=params["macd_f"], window_slow=params["macd_s"], window_sign=params["macd_sig"])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['ADX'] = adx.adx()
        
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['ATR'] = atr.average_true_range()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close'], 
            window=params["stoch_k"], smooth_window=params["stoch_d"]
        )
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        latest = df.iloc[-1]
        
        self.last_indicators = {
            "Timeframe": timeframe,
            "EMA50": round(float(latest['EMA_Fast']), 5) if pd.notna(latest['EMA_Fast']) else 0,
            "EMA200": round(float(latest['EMA_Slow']), 5) if pd.notna(latest['EMA_Slow']) else 0,
            "RSI": round(float(latest['RSI']), 2) if pd.notna(latest['RSI']) else 0,
            "MACD": round(float(latest['MACD']), 5) if pd.notna(latest['MACD']) else 0,
            "MACD_Signal": round(float(latest['MACD_Signal']), 5) if pd.notna(latest['MACD_Signal']) else 0,
            "ADX": round(float(latest['ADX']), 2) if pd.notna(latest['ADX']) else 0,
            "ATR": round(float(latest['ATR']), 5) if pd.notna(latest['ATR']) else 0,
            "Stoch_K": round(float(latest['Stoch_K']), 2) if pd.notna(latest['Stoch_K']) else 0,
            "Stoch_D": round(float(latest['Stoch_D']), 2) if pd.notna(latest['Stoch_D']) else 0,
        }
        
        if self.last_indicators['ATR'] < 0.00005: 
            return None, None
            
        # BUY Condition
        if (latest['EMA_Fast'] > latest['EMA_Slow'] and
            latest['RSI'] > params["rsi_buy"] and
            latest['MACD'] > latest['MACD_Signal'] and
            latest['ADX'] > params["adx"] and
            latest['Stoch_K'] > latest['Stoch_D']):
            return "CALL", params["expiry"]
            
        # SELL Condition
        if (latest['EMA_Fast'] < latest['EMA_Slow'] and
            latest['RSI'] < params["rsi_sell"] and
            latest['MACD'] < latest['MACD_Signal'] and
            latest['ADX'] > params["adx"] and
            latest['Stoch_K'] < latest['Stoch_D']):
            return "PUT", params["expiry"]
            
        return None, None
        
    def get_indicators(self, data: Any = None) -> Dict[str, Any]:
        return self.last_indicators

# ==========================================
# 2. Money & Risk Management Modules
# ==========================================

class BaseRiskManager(abc.ABC):
    """
    Abstract Base Class for money management systems.
    """
    def __init__(self, base_stake: float, config: Optional[Dict[str, Any]] = None):
        self.base_stake = base_stake
        self.config = config or {}
        self.current_stake = base_stake

    @abc.abstractmethod
    def get_stake(self) -> float:
        """Returns the current calculated stake size, rounded to 2 decimals."""
        pass

    @abc.abstractmethod
    def on_win(self) -> float:
        """Called when a contract wins. Returns the next stake."""
        pass

    @abc.abstractmethod
    def on_loss(self) -> float:
        """Called when a contract loses. Returns the next stake."""
        pass

    @abc.abstractmethod
    def update_base_stake(self, new_base: float):
        """Safely updates base stakes on-the-fly."""
        pass

    @abc.abstractmethod
    def reset(self):
        """Resets the stake and steps to initial values."""
        pass


class FlatRiskManager(BaseRiskManager):
    """
    Flat Stake Risk Manager:
    Always trades the base stake size. Safe and highly recommended for steady profiles.
    """
    def get_stake(self) -> float:
        return round(self.current_stake, 2)

    def on_win(self) -> float:
        self.current_stake = self.base_stake
        return self.get_stake()

    def on_loss(self) -> float:
        self.current_stake = self.base_stake
        return self.get_stake()

    def update_base_stake(self, new_base: float):
        self.base_stake = new_base
        self.current_stake = new_base

    def reset(self):
        self.current_stake = self.base_stake


class MartingaleRiskManager(BaseRiskManager):
    """
    Classic Martingale Risk Manager:
    Multiplies stake size on losses, resets on wins.
    """
    def __init__(self, base_stake: float, config: Optional[Dict[str, Any]] = None):
        super().__init__(base_stake, config)
        self.multiplier = float(self.config.get("martingale_multiplier", 2.0))
        self.max_steps = int(self.config.get("martingale_max_steps", 5))
        
        self.current_step = 0
        self.losses_in_row = 0

    def get_stake(self) -> float:
        return round(self.current_stake, 2)

    def on_win(self) -> float:
        self.current_stake = self.base_stake
        self.current_step = 0
        self.losses_in_row = 0
        return self.get_stake()

    def on_loss(self) -> float:
        self.losses_in_row += 1
        self.current_step += 1
        
        if self.current_step > self.max_steps:
            # Safety cap reset to protect account wipeout
            self.current_stake = self.base_stake
            self.current_step = 0
        else:
            self.current_stake = self.current_stake * self.multiplier
            
        return self.get_stake()

    def update_base_stake(self, new_base: float):
        self.base_stake = new_base
        if self.current_step == 0:
            self.current_stake = new_base

    def reset(self):
        self.current_stake = self.base_stake
        self.current_step = 0
        self.losses_in_row = 0


class FibonacciRiskManager(BaseRiskManager):
    """
    Fibonacci Progression Risk Manager:
    On loss: Move forward 1 step in the Fibonacci sequence (multiplied by base stake).
    On win: Move backward 2 steps in the Fibonacci sequence (minimum step 0).
    Highly recommended recovery model that doesn't wipe accounts as fast as Martingale.
    """
    def __init__(self, base_stake: float, config: Optional[Dict[str, Any]] = None):
        super().__init__(base_stake, config)
        self.max_steps = int(self.config.get("fibonacci_max_steps", 8))
        
        # Fibonacci Sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89
        self.fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
        self.current_step = 0

    def get_stake(self) -> float:
        multiplier = self.fib[min(self.current_step, len(self.fib) - 1)]
        return round(self.base_stake * multiplier, 2)

    def on_win(self) -> float:
        # Move back 2 steps on win
        self.current_step = max(0, self.current_step - 2)
        return self.get_stake()

    def on_loss(self) -> float:
        # Move forward 1 step on loss
        self.current_step += 1
        if self.current_step >= self.max_steps:
            # Safety boundary reset
            self.current_step = 0
        return self.get_stake()

    def update_base_stake(self, new_base: float):
        self.base_stake = new_base

    def reset(self):
        self.current_step = 0


class DAlembertRiskManager(BaseRiskManager):
    """
    D'Alembert Risk Manager:
    On loss: Increase stake by a fixed increment unit.
    On win: Decrease stake by a fixed increment unit (minimum is the base stake).
    Steady linear recovery model.
    """
    def __init__(self, base_stake: float, config: Optional[Dict[str, Any]] = None):
        super().__init__(base_stake, config)
        self.increment = float(self.config.get("dalembert_increment", base_stake))

    def get_stake(self) -> float:
        return round(self.current_stake, 2)

    def on_win(self) -> float:
        # Decrease stake on win
        self.current_stake = max(self.base_stake, self.current_stake - self.increment)
        return self.get_stake()

    def on_loss(self) -> float:
        # Increase stake on loss
        self.current_stake = self.current_stake + self.increment
        return self.get_stake()

    def update_base_stake(self, new_base: float):
        # Dynamically scale current stake if base changes
        diff = new_base - self.base_stake
        self.base_stake = new_base
        self.current_stake = max(new_base, self.current_stake + diff)

    def reset(self):
        self.current_stake = self.base_stake


class OscarsGrindRiskManager(BaseRiskManager):
    """
    Oscar's Grind Risk Manager:
    Traded in cycles. The goal of each cycle is to make exactly +1 unit (base_stake) profit.
    Stakes start at 1 unit.
    On loss: Stake remains the same.
    On win: Increment stake size by 1 unit, unless doing so would make the cycle profit exceed +1 unit,
    in which case set the stake to exactly the size needed to reach exactly +1 unit net profit.
    Once cycle profit reaches +1 unit, the cycle resets.
    """
    def __init__(self, base_stake: float, config: Optional[Dict[str, Any]] = None):
        super().__init__(base_stake, config)
        self.cycle_profit = 0.0
        self.unit_size = base_stake
        self.current_stake = base_stake

    def get_stake(self) -> float:
        return round(self.current_stake, 2)

    def on_win(self) -> float:
        self.cycle_profit += self.current_stake
        
        if self.cycle_profit >= self.unit_size:
            # Cycle targets hit! Reset cycle
            self.reset()
        else:
            # We won but target is not yet hit. Increment stake size by 1 unit
            # Check safety: if current_stake + cycle_profit would exceed 1 unit target,
            # we scale down the next stake size to exactly what's needed.
            needed = self.unit_size - self.cycle_profit
            if self.current_stake + self.unit_size > needed:
                self.current_stake = max(self.unit_size, needed)
            else:
                self.current_stake += self.unit_size
                
        return self.get_stake()

    def on_loss(self) -> float:
        self.cycle_profit -= self.current_stake
        # Stake size remains the same on losses in Oscar's Grind
        return self.get_stake()

    def update_base_stake(self, new_base: float):
        self.base_stake = new_base
        self.unit_size = new_base

    def reset(self):
        self.cycle_profit = 0.0
        self.current_stake = self.base_stake


# ==========================================
# 3. Strategy & Risk Factories
# ==========================================

def get_strategy(strategy_name: str, config: Optional[Dict[str, Any]] = None) -> BaseStrategy:
    if strategy_name == "technical_indicators":
        return TechnicalIndicatorStrategy(config)
    return HybridMLStrategy(config)

def get_risk_manager(money_management: str, base_stake: float, config: Optional[Dict[str, Any]] = None) -> BaseRiskManager:
    config = config or {}
    if money_management == "martingale":
        return MartingaleRiskManager(base_stake, config)
    elif money_management == "fibonacci":
        return FibonacciRiskManager(base_stake, config)
    elif money_management == "dalembert":
        return DAlembertRiskManager(base_stake, config)
    elif money_management == "oscars_grind":
        return OscarsGrindRiskManager(base_stake, config)
    else:
        # Default flat stake manager
        return FlatRiskManager(base_stake, config)
