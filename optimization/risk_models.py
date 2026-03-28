from abc import ABC, abstractmethod


class BaseRiskModel(ABC):
    @abstractmethod
    def position_size(self, capital: float, entry: float, sl: float) -> float:
        """Returns number of units/shares to trade."""


class FixedPctRisk(BaseRiskModel):
    def __init__(self, pct: float = 0.02):
        self.pct = pct

    def position_size(self, capital: float, entry: float, sl: float) -> float:
        price_risk = abs(entry - sl)
        if price_risk == 0:
            return 0.0
        risk_amount = capital * self.pct
        return risk_amount / price_risk


class FixedDollarRisk(BaseRiskModel):
    def __init__(self, amount: float = 20.0):
        self.amount = amount

    def position_size(self, capital: float, entry: float, sl: float) -> float:
        price_risk = abs(entry - sl)
        if price_risk == 0:
            return 0.0
        return self.amount / price_risk


class KellyRisk(BaseRiskModel):
    MAX_KELLY_FRACTION = 0.25  # cap at 25% of capital

    def __init__(self, win_rate: float = 0.5, avg_win: float = 2.0, avg_loss: float = 1.0):
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss

    def kelly_fraction(self) -> float:
        if self.avg_loss == 0:
            return 0.0
        b = self.avg_win / self.avg_loss
        f = (b * self.win_rate - (1 - self.win_rate)) / b
        return max(0.0, min(f, self.MAX_KELLY_FRACTION))

    def position_size(self, capital: float, entry: float, sl: float) -> float:
        price_risk = abs(entry - sl)
        if price_risk == 0:
            return 0.0
        risk_amount = capital * self.kelly_fraction()
        return risk_amount / price_risk
