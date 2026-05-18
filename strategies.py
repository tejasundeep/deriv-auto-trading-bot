import abc
import math
from typing import List, Optional, Dict, Any

# ==========================================
# 1. Native High-Performance Indicators
# ==========================================

def calculate_sma(prices: List[float], period: int) -> Optional[float]:
    """Calculates Simple Moving Average."""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

def calculate_std_dev(prices: List[float], period: int, sma: float) -> Optional[float]:
    """Calculates Standard Deviation over a period, given its SMA."""
    if len(prices) < period:
        return None
    variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
    return math.sqrt(variance)

def calculate_ema(prices: List[float], period: int) -> List[float]:
    """
    Calculates Exponential Moving Average for a series.
    Returns a list of EMAs corresponding to elements from indices `period-1` onwards.
    """
    if len(prices) < period:
        return []
    
    ema_list = []
    # Base SMA
    sma = sum(prices[:period]) / period
    ema_list.append(sma)
    
    multiplier = 2.0 / (period + 1)
    for price in prices[period:]:
        next_ema = (price - ema_list[-1]) * multiplier + ema_list[-1]
        ema_list.append(next_ema)
        
    return ema_list

def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """
    Calculates RSI using Wilder's Smoothing Technique.
    """
    if len(prices) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)

    # First average using SMA
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder's Smoothing for subsequent intervals
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def calculate_bollinger_bands(prices: List[float], period: int = 20, num_std: float = 2.0) -> Optional[Dict[str, float]]:
    """
    Calculates Bollinger Bands.
    """
    if len(prices) < period:
        return None
    
    sma = calculate_sma(prices, period)
    if sma is None:
        return None
        
    std_dev = calculate_std_dev(prices, period, sma)
    if std_dev is None:
        return None

    return {
        "upper": sma + (num_std * std_dev),
        "middle": sma,
        "lower": sma - (num_std * std_dev),
        "std_dev": std_dev
    }

def calculate_macd(prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Optional[Dict[str, float]]:
    """
    Calculates MACD, Signal Line, and Histogram.
    """
    if len(prices) < slow_period + signal_period:
        return None

    fast_emas = calculate_ema(prices, fast_period)
    slow_emas = calculate_ema(prices, slow_period)
    
    if not fast_emas or not slow_emas:
        return None

    # Align Fast and Slow EMA lists.
    # Fast EMA starts after fast_period values. Slow EMA starts after slow_period values.
    # We slice fast_emas to match the length of slow_emas.
    offset = slow_period - fast_period
    macd_line = []
    for i in range(len(slow_emas)):
        macd_line.append(fast_emas[i + offset] - slow_emas[i])

    # Signal Line is the EMA of the MACD Line
    signal_emas = calculate_ema(macd_line, signal_period)
    if not signal_emas:
        return None

    return {
        "macd": macd_line[-1],
        "signal": signal_emas[-1],
        "histogram": macd_line[-1] - signal_emas[-1]
    }

# ==========================================
# 2. Trading Strategies (Base + Subclasses)
# ==========================================

class BaseStrategy(abc.ABC):
    """Abstract Base Class for all trading strategies."""
    def __init__(self, name: str, description: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.config = config or {}

    @abc.abstractmethod
    def analyze(self, ticks: List[float]) -> Optional[str]:
        """
        Analyzes price series and returns a signal: "CALL", "PUT", or None.
        """
        pass

    @abc.abstractmethod
    def get_indicators(self, ticks: List[float]) -> Dict[str, Any]:
        """
        Calculates and returns indicator state values for HUD telemetry.
        """
        pass


class CandleTrendStrategy(BaseStrategy):
    """
    Candle Trend Strategy:
    CALL when N consecutive 1-minute closed candles are rising.
    PUT when N consecutive 1-minute closed candles are falling.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="Candle Trend",
            description="Trades based on consecutive micro-trends of closed 1-minute candles.",
            config=config
        )
        self.consecutive_ticks = int(self.config.get("tick_trend_consecutive", 3))

    def analyze(self, ticks: List[float]) -> Optional[str]:
        if len(ticks) < self.consecutive_ticks + 1:
            return None

        recent_ticks = ticks[-(self.consecutive_ticks + 1):]
        rising = True
        falling = True

        for i in range(1, len(recent_ticks)):
            diff = recent_ticks[i] - recent_ticks[i - 1]
            if diff <= 0:
                rising = False
            if diff >= 0:
                falling = False

        if rising:
            return "CALL"
        elif falling:
            return "PUT"

        return None

    def get_indicators(self, ticks: List[float]) -> Dict[str, Any]:
        if len(ticks) < 2:
            return {"trend": "Neutral", "last_diff": 0.0}
        
        diff = ticks[-1] - ticks[-2]
        trend = "Rising" if diff > 0 else "Falling" if diff < 0 else "Flat"
        return {
            "trend": trend,
            "last_diff": round(diff, 4)
        }


class RSICrossoverStrategy(BaseStrategy):
    """
    RSI Crossover Strategy:
    CALL (Oversold) if RSI <= lower_bound.
    PUT (Overbought) if RSI >= upper_bound.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="RSI Crossover",
            description="Trades reversals when RSI enters extreme overbought/oversold levels.",
            config=config
        )
        self.period = int(self.config.get("rsi_period", 14))
        self.lower_bound = float(self.config.get("rsi_lower_bound", 30.0))
        self.upper_bound = float(self.config.get("rsi_upper_bound", 70.0))

    def analyze(self, ticks: List[float]) -> Optional[str]:
        if len(ticks) < self.period + 5:
            return None

        rsi = calculate_rsi(ticks, self.period)
        if rsi is None:
            return None

        if rsi <= self.lower_bound:
            return "CALL"
        elif rsi >= self.upper_bound:
            return "PUT"

        return None

    def get_indicators(self, ticks: List[float]) -> Dict[str, Any]:
        rsi = calculate_rsi(ticks, self.period)
        if rsi is None:
            return {"rsi": "N/A", "status": "Awaiting Ticks"}
        
        status = "Oversold" if rsi <= self.lower_bound else "Overbought" if rsi >= self.upper_bound else "Neutral"
        return {
            "rsi": round(rsi, 2),
            "status": status
        }


class BollingerBandsRSIStrategy(BaseStrategy):
    """
    Bollinger Bands + RSI Volatility Reversal Strategy:
    CALL: Tick crosses below Lower BB AND RSI <= oversold (30).
    PUT: Tick crosses above Upper BB AND RSI >= overbought (70).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="BB & RSI Reversal",
            description="Mean reversion strategy using Bollinger Bands and RSI confluence.",
            config=config
        )
        self.bb_period = int(self.config.get("bb_period", 20))
        self.bb_std_dev = float(self.config.get("bb_std_dev", 2.0))
        self.rsi_period = int(self.config.get("rsi_period", 14))
        self.rsi_lower_bound = float(self.config.get("rsi_lower_bound", 30.0))
        self.rsi_upper_bound = float(self.config.get("rsi_upper_bound", 70.0))

    def analyze(self, ticks: List[float]) -> Optional[str]:
        if len(ticks) < max(self.bb_period, self.rsi_period + 5):
            return None

        bb = calculate_bollinger_bands(ticks, self.bb_period, self.bb_std_dev)
        rsi = calculate_rsi(ticks, self.rsi_period)
        
        if not bb or rsi is None:
            return None

        price = ticks[-1]
        
        # We check crossover: price falls below lower band, and RSI is oversold
        if price <= bb["lower"] and rsi <= self.rsi_lower_bound:
            return "CALL"
        # Price climbs above upper band, and RSI is overbought
        elif price >= bb["upper"] and rsi >= self.rsi_upper_bound:
            return "PUT"

        return None

    def get_indicators(self, ticks: List[float]) -> Dict[str, Any]:
        bb = calculate_bollinger_bands(ticks, self.bb_period, self.bb_std_dev)
        rsi = calculate_rsi(ticks, self.rsi_period)
        
        if not bb or rsi is None:
            return {"rsi": "N/A", "bb_upper": "N/A", "bb_lower": "N/A", "status": "Awaiting Ticks"}
            
        price = ticks[-1]
        status = "Reversal BUY" if (price <= bb["lower"] and rsi <= self.rsi_lower_bound) else \
                 "Reversal SELL" if (price >= bb["upper"] and rsi >= self.rsi_upper_bound) else "Neutral"
                 
        return {
            "rsi": round(rsi, 2),
            "bb_upper": round(bb["upper"], 4),
            "bb_middle": round(bb["middle"], 4),
            "bb_lower": round(bb["lower"], 4),
            "status": status
        }


class EMAMACDConfluenceStrategy(BaseStrategy):
    """
    EMA Crossover with MACD Confluence:
    CALL: Fast EMA > Slow EMA AND MACD histogram > 0 AND rising.
    PUT: Fast EMA < Slow EMA AND MACD histogram < 0 AND falling.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="EMA & MACD Confluence",
            description="Follows trends combining double EMA crossovers and MACD momentum.",
            config=config
        )
        self.ema_fast_p = int(self.config.get("ema_fast", 9))
        self.ema_slow_p = int(self.config.get("ema_slow", 21))
        
        self.macd_fast = int(self.config.get("macd_fast", 12))
        self.macd_slow = int(self.config.get("macd_slow", 26))
        self.macd_signal = int(self.config.get("macd_signal", 9))

    def analyze(self, ticks: List[float]) -> Optional[str]:
        req_len = max(self.ema_slow_p, self.macd_slow + self.macd_signal) + 5
        if len(ticks) < req_len:
            return None

        fast_emas = calculate_ema(ticks, self.ema_fast_p)
        slow_emas = calculate_ema(ticks, self.ema_slow_p)
        macd = calculate_macd(ticks, self.macd_fast, self.macd_slow, self.macd_signal)
        
        if not fast_emas or not slow_emas or not macd:
            return None

        fast_ema = fast_emas[-1]
        slow_ema = slow_emas[-1]
        
        # Check crossover state
        if fast_ema > slow_ema and macd["histogram"] > 0:
            return "CALL"
        elif fast_ema < slow_ema and macd["histogram"] < 0:
            return "PUT"

        return None

    def get_indicators(self, ticks: List[float]) -> Dict[str, Any]:
        req_len = max(self.ema_slow_p, self.macd_slow + self.macd_signal)
        if len(ticks) < req_len:
            return {"macd": "N/A", "macd_signal": "N/A", "ema_fast": "N/A", "ema_slow": "N/A", "status": "Awaiting Ticks"}
            
        fast_emas = calculate_ema(ticks, self.ema_fast_p)
        slow_emas = calculate_ema(ticks, self.ema_slow_p)
        macd = calculate_macd(ticks, self.macd_fast, self.macd_slow, self.macd_signal)
        
        if not fast_emas or not slow_emas or not macd:
            return {"macd": "N/A", "macd_signal": "N/A", "ema_fast": "N/A", "ema_slow": "N/A", "status": "Awaiting Ticks"}
            
        status = "Bullish" if (fast_emas[-1] > slow_emas[-1] and macd["histogram"] > 0) else \
                 "Bearish" if (fast_emas[-1] < slow_emas[-1] and macd["histogram"] < 0) else "Neutral"
                 
        return {
            "ema_fast": round(fast_emas[-1], 4),
            "ema_slow": round(slow_emas[-1], 4),
            "macd": round(macd["macd"], 4),
            "macd_signal": round(macd["signal"], 4),
            "macd_hist": round(macd["histogram"], 4),
            "status": status
        }


class CandleVelocityStrategy(BaseStrategy):
    """
    Candle Velocity and Acceleration Strategy:
    Measures speed of 1-minute closed candle price changes (first derivative) and acceleration (second derivative).
    Signals CALL when velocity breaks above positive standard deviations.
    Signals PUT when velocity breaks below negative standard deviations.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="Candle Velocity Momentum",
            description="Trades dynamic breakout surges using candle velocity and acceleration standard deviations.",
            config=config
        )
        self.period = int(self.config.get("velocity_period", 10))
        self.threshold_sd = float(self.config.get("velocity_threshold_sd", 1.5))

    def analyze(self, ticks: List[float]) -> Optional[str]:
        # We need enough ticks to get a stable velocity average and standard deviation
        if len(ticks) < self.period + 5:
            return None

        # Calculate sliding differences (velocities)
        velocities = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
        
        # Calculate mean velocity and standard deviation of velocity over the window
        recent_vel = velocities[-self.period:]
        avg_vel = sum(recent_vel) / self.period
        
        variance = sum((v - avg_vel) ** 2 for v in recent_vel) / self.period
        std_vel = math.sqrt(variance)
        if std_vel == 0.0:
            std_vel = 0.0001
            
        current_vel = velocities[-1]
        # Acceleration: rate of change of speed
        acceleration = velocities[-1] - velocities[-2] if len(velocities) >= 2 else 0.0

        # Signals
        if current_vel > (avg_vel + self.threshold_sd * std_vel) and acceleration > 0:
            return "CALL"
        elif current_vel < (avg_vel - self.threshold_sd * std_vel) and acceleration < 0:
            return "PUT"

        return None

    def get_indicators(self, ticks: List[float]) -> Dict[str, Any]:
        if len(ticks) < self.period + 2:
            return {"velocity": "N/A", "acceleration": "N/A", "status": "Awaiting Ticks"}
            
        velocities = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
        recent_vel = velocities[-self.period:]
        avg_vel = sum(recent_vel) / self.period
        
        variance = sum((v - avg_vel) ** 2 for v in recent_vel) / self.period
        std_vel = math.sqrt(variance)
        
        current_vel = velocities[-1]
        acceleration = velocities[-1] - velocities[-2]
        
        upper_band = avg_vel + self.threshold_sd * std_vel
        lower_band = avg_vel - self.threshold_sd * std_vel
        
        status = "Upward Spike" if (current_vel > upper_band and acceleration > 0) else \
                 "Downward Spike" if (current_vel < lower_band and acceleration < 0) else "Stable"
                 
        return {
            "velocity": round(current_vel, 6),
            "acceleration": round(acceleration, 6),
            "vel_avg": round(avg_vel, 6),
            "vel_sd": round(std_vel, 6),
            "status": status
        }

# ==========================================
# 3. Money & Risk Management Modules
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
# 4. Strategy & Risk Factories
# ==========================================

def get_strategy(strategy_name: str, config: Optional[Dict[str, Any]] = None) -> BaseStrategy:
    config = config or {}
    if strategy_name == "rsi_crossover":
        return RSICrossoverStrategy(config)
    elif strategy_name == "bb_rsi_reversal":
        return BollingerBandsRSIStrategy(config)
    elif strategy_name == "ema_macd_confluence":
        return EMAMACDConfluenceStrategy(config)
    elif strategy_name == "tick_velocity" or strategy_name == "candle_velocity":
        return CandleVelocityStrategy(config)
    else:
        # Default strategy fallback
        return CandleTrendStrategy(config)

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
