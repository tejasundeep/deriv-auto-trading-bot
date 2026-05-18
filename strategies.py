import abc
from typing import List, Optional, Dict, Any

from ml_engine import HybridMLStrategy

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
    def analyze(self, ticks: List[float]) -> Optional[str]:
        """
        Analyzes price series and returns a signal: "CALL", "PUT", or None.
        """
        pass

    @abc.abstractmethod
    def get_indicators(self, ticks: List[float]) -> Dict[str, Any]:
        """
        Returns strategy telemetry for the ML dashboard.
        """
        pass
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
