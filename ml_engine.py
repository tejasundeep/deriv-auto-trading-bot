import asyncio
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import config as app_config


class SequenceFeatureBuilder:
    """Builds candle-structure sequence features from OHLC candles using fast NumPy vectorization."""

    def __init__(self, window_size: int = 30):
        self.window_size = int(window_size)

    def build_dataset(self, candles: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        if len(candles) < self.window_size + 2:
            return (
                np.empty((0, self.window_size, 15), dtype=float),
                np.empty((0,), dtype=float),
            )

        opens = np.array([c.get("open", c.get("close", 0.0)) for c in candles], dtype=float)
        closes = np.array([c.get("close", c.get("open", 0.0)) for c in candles], dtype=float)
        highs = np.array([c.get("high", o) for c, o in zip(candles, opens)], dtype=float)
        lows = np.array([c.get("low", o) for c, o in zip(candles, opens)], dtype=float)

        rng = np.maximum(highs - lows, 1e-8)
        body = closes - opens
        upper_wick = highs - np.maximum(opens, closes)
        lower_wick = np.minimum(opens, closes) - lows
        
        body_ratio = np.abs(body) / rng
        upper_wick_ratio = upper_wick / rng
        lower_wick_ratio = lower_wick / rng
        open_position = (opens - lows) / rng
        close_position = (closes - lows) / rng

        from numpy.lib.stride_tricks import sliding_window_view

        # Generate sliding windows of size self.window_size
        # Valid ends are self.window_size to len(candles) - 1 (exclusive of the very last element to allow +1 target)
        valid_ends = np.arange(self.window_size, len(candles) - 1)
        current_closes = closes[valid_ends]
        next_closes = closes[valid_ends + 1]
        labels = (next_closes > current_closes).astype(float)

        # Windows end precisely at valid_ends - meaning they contain indices from valid_ends - window_size to valid_ends
        # `sliding_window_view` returns all possible windows. 
        # The window ending at index `w` is at index `w - window_size + 1` in the view.
        # Since we want windows ending at `self.window_size` up to `len(candles) - 2`,
        # these correspond to indices 1 up to len(candles) - self.window_size - 1 in the sliding view.
        
        w_closes = sliding_window_view(closes, self.window_size)[1 : len(candles) - self.window_size]
        w_opens = sliding_window_view(opens, self.window_size)[1 : len(candles) - self.window_size]
        w_highs = sliding_window_view(highs, self.window_size)[1 : len(candles) - self.window_size]
        w_lows = sliding_window_view(lows, self.window_size)[1 : len(candles) - self.window_size]
        
        w_body_ratio = sliding_window_view(body_ratio, self.window_size)[1 : len(candles) - self.window_size]
        w_upper_ratio = sliding_window_view(upper_wick_ratio, self.window_size)[1 : len(candles) - self.window_size]
        w_lower_ratio = sliding_window_view(lower_wick_ratio, self.window_size)[1 : len(candles) - self.window_size]
        w_open_pos = sliding_window_view(open_position, self.window_size)[1 : len(candles) - self.window_size]
        w_close_pos = sliding_window_view(close_position, self.window_size)[1 : len(candles) - self.window_size]
        w_rng = sliding_window_view(rng, self.window_size)[1 : len(candles) - self.window_size]

        firsts = w_closes[:, 0].copy()
        firsts[firsts == 0.0] = 1.0
        
        means = w_closes.mean(axis=1, keepdims=True)
        stds = w_closes.std(axis=1, keepdims=True)
        stds[stds < 1e-8] = 1e-8
        
        diffs = np.diff(w_closes, axis=1)
        denom = w_closes[:, :-1].copy()
        denom[denom == 0.0] = 1.0
        ret_part = diffs / denom
        returns = np.zeros_like(w_closes)
        returns[:, 1:] = ret_part
        
        body_arr = np.zeros_like(w_closes)
        body_arr[:, 1:] = w_closes[:, 1:] - w_closes[:, :-1]
        
        range_span = w_closes.max(axis=1, keepdims=True) - w_closes.min(axis=1, keepdims=True)
        mean_denom = means.copy()
        mean_denom[mean_denom == 0.0] = 1.0
        
        abs_window = np.abs(w_closes)
        abs_window[abs_window < 1e-8] = 1.0
        
        firsts_expanded = firsts[:, None]
        
        seq_samples = np.stack([
            (w_opens / firsts_expanded) - 1.0,
            (w_highs / firsts_expanded) - 1.0,
            (w_lows / firsts_expanded) - 1.0,
            (w_closes / firsts_expanded) - 1.0,
            (w_closes / firsts_expanded) - 1.0,
            returns,
            (w_closes - means) / stds,
            body_arr / abs_window,
            np.broadcast_to(range_span / mean_denom, w_closes.shape),
            w_body_ratio,
            w_upper_ratio,
            w_lower_ratio,
            w_open_pos,
            w_close_pos,
            w_rng / mean_denom
        ], axis=2)
        
        return seq_samples, labels

    def extract_sequence(self, window: List[Dict[str, float]]) -> np.ndarray:
        # Fallback for live single-window extraction
        samples, _ = self.build_dataset(window + [{"close": window[-1].get("close", 0.0)}, {"close": 0.0}])
        if len(samples) > 0:
            return samples[0]
        return np.zeros((self.window_size, 15), dtype=float)



class SequenceHybridNet(nn.Module):
    """Fast, purely convolutional and GRU-based sequence model for rapid training."""
    def __init__(
        self,
        sequence_input_dim: int,
        hidden_units: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(sequence_input_dim, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_units),
            nn.GELU(),
            nn.Conv1d(hidden_units, hidden_units, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_units),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_units,
            hidden_size=hidden_units // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.combined = nn.Sequential(
            nn.Linear(hidden_units, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq shape: (batch_size, seq_len, sequence_input_dim)
        x = seq.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        
        # GRU returns output and hidden state
        x, _ = self.gru(x)
        
        # Global Average Pooling
        seq_repr = x.mean(dim=1)
        
        return self.combined(seq_repr)


class SimpleBinarySequenceModel:
    """PyTorch hybrid sequence model for next-candle direction classification."""

    def __init__(
        self,
        hidden_units: int = 48,
        learning_rate: float = 0.001,
        epochs: int = 40,
        l2_penalty: float = 1e-4,
        seed: int = 42,
        batch_size: int = 64,
    ):
        self.hidden_units = int(hidden_units)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.l2_penalty = float(l2_penalty)
        self.seed = int(seed)
        self.batch_size = int(batch_size)

        self.sequence_input_dim: Optional[int] = None
        self.seq_mean_: Optional[np.ndarray] = None
        self.seq_std_: Optional[np.ndarray] = None
        self.model: Optional[SequenceHybridNet] = None
        self.fitted: bool = False
        self.train_samples: int = 0

    def _init_model(self, sequence_input_dim: int) -> None:
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.sequence_input_dim = int(sequence_input_dim)
        self.model = SequenceHybridNet(sequence_input_dim=self.sequence_input_dim, hidden_units=self.hidden_units)

    def _normalize(self, seq: np.ndarray) -> np.ndarray:
        seq_n = seq
        if self.seq_mean_ is not None and self.seq_std_ is not None:
            seq_n = (seq - self.seq_mean_) / self.seq_std_
        return seq_n

    def fit(self, X_seq: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        X_seq = np.asarray(X_seq, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if X_seq.size == 0 or y.size == 0:
            raise ValueError("No training data available.")

        if self.model is None or self.sequence_input_dim != X_seq.shape[2]:
            self._init_model(X_seq.shape[2])

        self.seq_mean_ = X_seq.mean(axis=(0, 1), keepdims=True)
        self.seq_std_ = X_seq.std(axis=(0, 1), keepdims=True)
        self.seq_std_[self.seq_std_ < 1e-8] = 1.0

        X_seq = self._normalize(X_seq)
        dataset = TensorDataset(torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

        val_size = max(1, int(len(dataset) * 0.2))
        train_size = len(dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )
        train_loader = DataLoader(train_ds, batch_size=min(self.batch_size, max(1, train_size)), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=min(self.batch_size, max(1, val_size)), shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_penalty)
        criterion = nn.BCEWithLogitsLoss()
        best_state = None
        best_val_loss = float("inf")

        self.model.train()
        for _ in range(self.epochs):
            for seq_batch, y_batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(seq_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            val_loss = self._evaluate_loss(val_loader, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }

        if best_state:
            self.model.load_state_dict(best_state["model"])

        self.fitted = True
        self.train_samples = int(X_seq.shape[0])
        return self.metrics(X_seq, y.ravel())

    def _evaluate_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        if self.model is None:
            return float("inf")
        self.model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for seq_batch, y_batch in loader:
                logits = self.model(seq_batch)
                loss = criterion(logits, y_batch)
                losses.append(float(loss.item()))
        self.model.train()
        return float(sum(losses) / max(1, len(losses)))

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        if not self.fitted or self.model is None:
            raise ValueError("Model is not trained yet.")

        X_seq = np.asarray(X_seq, dtype=float)
        if X_seq.ndim == 2:
            X_seq = X_seq.reshape(1, *X_seq.shape)
        X_seq = self._normalize(X_seq)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_seq, dtype=torch.float32))
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
        return probs

    def metrics(self, X_seq: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        probs = self.predict_proba(X_seq)
        preds = (probs >= 0.5).astype(int)
        accuracy = float((preds == y.astype(int)).mean())
        loss = float(
            -np.mean(
                y * np.log(np.clip(probs, 1e-8, 1.0 - 1e-8))
                + (1.0 - y) * np.log(np.clip(1.0 - probs, 1e-8, 1.0 - 1e-8))
            )
        )
        return {"accuracy": round(accuracy, 4), "loss": round(loss, 6)}

    def save(self, path: str) -> None:

        if not self.fitted or self.model is None:
            raise ValueError("Model is not trained yet.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "hidden_units": self.hidden_units,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "sequence_input_dim": self.sequence_input_dim,
                "state_dict": self.model.state_dict(),
                "seq_mean": self.seq_mean_,
                "seq_std": self.seq_std_,
                "train_samples": self.train_samples,
            },
            path,
        )

    def reset(self) -> None:
        self.sequence_input_dim = None
        self.seq_mean_ = None
        self.seq_std_ = None
        self.model = None
        self.fitted = False
        self.train_samples = 0

    @classmethod
    def load(cls, path: str) -> Optional["SimpleBinarySequenceModel"]:
        if not os.path.exists(path):
            return None

        try:
            if path.endswith(".pkl"):
                import pickle
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, cls):
                    return obj
                elif isinstance(obj, dict):
                    data = obj
                else:
                    raise ValueError(f"Unsupported pickle object type: {type(obj)}")
            else:
                data = torch.load(path, map_location="cpu", weights_only=False)
            model = cls(
                hidden_units=int(data.get("hidden_units", 48)),
                learning_rate=float(data.get("learning_rate", 0.001)),
                epochs=int(data.get("epochs", 40)),
                batch_size=int(data.get("batch_size", 64)),
            )
            model.sequence_input_dim = int(data.get("sequence_input_dim", 3))
            model.seq_mean_ = data.get("seq_mean")
            model.seq_std_ = data.get("seq_std")
            model.train_samples = int(data.get("train_samples", 0))
            model._init_model(model.sequence_input_dim)

            state_dict = data.get("state_dict", {})
            current_state = model.model.state_dict()
            compatible_state = {}
            for key, tensor in state_dict.items():
                if key in current_state and current_state[key].shape == tensor.shape:
                    compatible_state[key] = tensor

            if not compatible_state:
                return None

            current_state.update(compatible_state)
            model.model.load_state_dict(current_state)
            model.fitted = len(compatible_state) == len(current_state)
            return model
        except Exception:
            return None


@dataclass
class PatternProfile:
    pattern_id: int
    name: str
    count: int
    hit_rate: float
    avg_return: float
    direction: str
    confidence: float
    body_ratio: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    open_position: float


@dataclass
class MLDecision:
    signal: Optional[str]
    probability_up: Optional[float]
    pattern_id: Optional[int]
    pattern_name: str
    status: str


class HybridMLStrategy:
    """Pure ML strategy: candle pattern discovery plus sequence classifier."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = "ML Pattern Engine"
        self.description = "Trains on Deriv candle history, discovers repeating patterns, and trades only when ML confidence is strong."

        self.window_size = int(self.config.get("ml_window_size", 30))
        self.min_samples = int(self.config.get("ml_min_samples", 250))
        self.history_window = int(self.config.get("ml_history_candles", 2000))
        self.buy_threshold = float(self.config.get("ml_buy_threshold", 0.58))
        self.sell_threshold = float(self.config.get("ml_sell_threshold", 0.42))
        self.retrain_every = int(self.config.get("ml_retrain_every", 60))
        self.hidden_units = int(self.config.get("ml_hidden_units", 64))
        self.learning_rate = float(self.config.get("ml_learning_rate", 0.01))
        self.epochs = int(self.config.get("ml_epochs", 120))
        self.batch_size = int(self.config.get("ml_batch_size", 64))
        self.l2_penalty = float(self.config.get("ml_l2_penalty", 1e-4))
        self.model_path = self.config.get("ml_model_path", app_config.DEFAULT_ML_MODEL_PATH)
        self.max_patterns = int(self.config.get("ml_max_patterns", 12))
        self.min_pattern_samples = int(self.config.get("ml_min_pattern_samples", 50))
        self.pattern_min_hit_rate = float(self.config.get("ml_pattern_min_hit_rate", 0.62))
        self.pattern_top_k = int(self.config.get("ml_pattern_top_k", 3))
        self.pattern_confidence_threshold = float(self.config.get("ml_pattern_confidence_threshold", 0.55))
        self.regime_slices = int(self.config.get("ml_regime_slices", 4))

        self.feature_builder = SequenceFeatureBuilder(window_size=self.window_size)
        self.model: SimpleBinarySequenceModel = SimpleBinarySequenceModel.load(self.model_path) or SimpleBinarySequenceModel(
            hidden_units=self.hidden_units,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
            l2_penalty=self.l2_penalty,
        )

        self.training_task: Optional[asyncio.Task] = None
        self.last_train_size: int = self.model.train_samples if self.model.fitted else 0
        self.last_metrics: Dict[str, Any] = {}
        self.last_probability: Optional[float] = None
        self.last_pattern_id: Optional[int] = None
        self.last_pattern_name: str = "N/A"
        self.patterns: List[PatternProfile] = []
        self.last_status: str = "Awaiting training data"
        self.last_retrain_count: int = 0
        self.last_live_candle_metrics: Dict[str, Any] = {}
        self.last_pattern_debug: Dict[str, Any] = {}

    def _candle_metrics(self, candles: List[Dict[str, Any]]) -> Dict[str, float]:
        if not candles:
            return {
                "body_ratio": 0.0,
                "upper_wick_ratio": 0.0,
                "lower_wick_ratio": 0.0,
                "open_position": 0.5,
                "close_position": 0.5,
                "bias_heat": 0.0,
                "wick_imbalance": 0.0,
                "confidence_meter": 0.0,
                "regime": "neutral",
                "color": "neutral",
            }

        metrics = {"body": [], "upper": [], "lower": [], "position": [], "close_position": [], "bias": [], "color": []}
        for candle in candles[-self.window_size :]:
            open_p = float(candle.get("open", 0.0))
            high_p = float(candle.get("high", open_p))
            low_p = float(candle.get("low", open_p))
            close_p = float(candle.get("close", open_p))
            rng = max(high_p - low_p, 1e-8)
            body = abs(close_p - open_p) / rng
            upper = (high_p - max(open_p, close_p)) / rng
            lower = (min(open_p, close_p) - low_p) / rng
            position = (close_p - low_p) / rng
            if close_p > open_p:
                color = "bull"
            elif close_p < open_p:
                color = "bear"
            else:
                color = "doji"
            bias = (close_p - open_p) / rng
            metrics["body"].append(body)
            metrics["upper"].append(upper)
            metrics["lower"].append(lower)
            metrics["position"].append(position)
            metrics["close_position"].append((close_p - low_p) / rng)
            metrics["bias"].append(bias)
            metrics["color"].append(color)

        body_ratio = round(float(np.mean(metrics["body"])), 4)
        upper_ratio = round(float(np.mean(metrics["upper"])), 4)
        lower_ratio = round(float(np.mean(metrics["lower"])), 4)
        open_position = round(float(np.mean(metrics["position"])), 4)
        close_position = round(float(np.mean(metrics["close_position"])), 4)
        bias_heat = round(float(np.mean(metrics["bias"])), 4)
        wick_imbalance = round(float(np.mean(metrics["upper"]) - np.mean(metrics["lower"])), 4)
        regime = "bullish" if bias_heat > 0.08 else "bearish" if bias_heat < -0.08 else "balanced"

        return {
            "body_ratio": body_ratio,
            "upper_wick_ratio": upper_ratio,
            "lower_wick_ratio": lower_ratio,
            "open_position": open_position,
            "close_position": close_position,
            "bias_heat": bias_heat,
            "wick_imbalance": wick_imbalance,
            "regime": regime,
            "color": max(set(metrics["color"]), key=metrics["color"].count),
        }

    def get_pattern_summary(self, candles: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        ranked = sorted(self.patterns, key=lambda p: (p.confidence, p.hit_rate, p.count), reverse=True)
        candle_metrics = self._candle_metrics(candles or [])
        self.last_live_candle_metrics = candle_metrics
        top_patterns = []
        for rank, p in enumerate(ranked[: self.pattern_top_k], start=1):
            top_patterns.append(
                {
                    "rank": rank,
                    "name": p.name,
                    "direction": p.direction,
                    "hit_rate": p.hit_rate,
                    "confidence": p.confidence,
                    "samples": p.count,
                    "avg_return": p.avg_return,
                    "body_ratio": p.body_ratio,
                    "upper_wick_ratio": p.upper_wick_ratio,
                    "lower_wick_ratio": p.lower_wick_ratio,
                    "open_position": p.open_position,
                    "close_position": p.open_position,
                    "bias_heat": round(p.open_position - 0.5, 4),
                    "wick_imbalance": round(p.upper_wick_ratio - p.lower_wick_ratio, 4),
                    "strength": "elite" if p.confidence >= 0.85 else "strong" if p.confidence >= 0.75 else "watch",
                }
            )
        return {
            "status": self.last_status,
            "top_patterns": top_patterns,
            "pattern_count": len(self.patterns),
            "last_pattern": self.last_pattern_name,
            "last_probability_up": self.last_probability,
            "candle_metrics": candle_metrics,
            "confidence_meter": round(
                min(
                    1.0,
                    max(
                        0.0,
                        max(abs(candle_metrics["bias_heat"]) * 1.5, candle_metrics["body_ratio"], abs(candle_metrics["wick_imbalance"]))
                    ),
                ),
                4,
            ),
            "abstain_mode": bool(
                not self.patterns
                or self.last_probability is None
                or (self.sell_threshold < self.last_probability < self.buy_threshold)
            ),
            "confidence_threshold": self.pattern_confidence_threshold,
            "pattern_debug": self.last_pattern_debug,
        }

    def reset_model(self, remove_checkpoint: bool = False) -> Dict[str, Any]:
        self.model.reset()
        self.patterns = []
        self.last_status = "Model reset"
        self.last_metrics = {}
        self.last_probability = None
        self.last_pattern_id = None
        self.last_pattern_name = "N/A"
        if remove_checkpoint and os.path.exists(self.model_path):
            try:
                os.remove(self.model_path)
            except OSError:
                pass
        return {"status": self.last_status, "checkpoint_removed": bool(remove_checkpoint)}

    def _pattern_feature_matrix(self, X_seq: np.ndarray) -> np.ndarray:
        return X_seq.reshape(X_seq.shape[0], -1)

    def _cluster_patterns(self, X_seq: np.ndarray, y: np.ndarray, regime: str = "all") -> List[PatternProfile]:
        if X_seq.size == 0:
            return []
            
        max_samples = 15000
        if len(X_seq) > max_samples:
            rng_sub = np.random.default_rng(42)
            indices = rng_sub.choice(len(X_seq), size=max_samples, replace=False)
            X_seq_sub = X_seq[indices]
            y_sub = y[indices]
        else:
            X_seq_sub = X_seq
            y_sub = y
            
        flat = self._pattern_feature_matrix(X_seq_sub)
        k = min(self.max_patterns, max(2, int(math.sqrt(len(flat)))))
        k = min(k, len(flat))
        if k < 2:
            return []

        rng = np.random.default_rng(42)
        centers = flat[rng.choice(len(flat), size=k, replace=False)]
        labels = np.zeros(len(flat), dtype=int)
        
        for _ in range(25):
            distances = np.zeros((len(flat), k), dtype=float)
            for idx in range(k):
                distances[:, idx] = np.sum((flat - centers[idx])**2, axis=1)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            new_centers = []
            for idx in range(k):
                members = flat[labels == idx]
                if len(members) == 0:
                    new_centers.append(centers[idx])
                else:
                    new_centers.append(members.mean(axis=0))
            centers = np.asarray(new_centers, dtype=float)

        patterns: List[PatternProfile] = []
        for idx in range(k):
            members = np.where(labels == idx)[0]
            if len(members) < self.min_pattern_samples:
                continue
            member_labels = y_sub[members]
            hit_rate = float(member_labels.mean())
            support = len(members) / max(1, len(flat))
            confidence = min(1.0, (hit_rate * 0.7) + (support * 0.3))
            direction = "CALL" if hit_rate >= 0.5 else "PUT"
            avg_return = float((member_labels.mean() * 2.0) - 1.0)
            patterns.append(
                PatternProfile(
                    pattern_id=idx + 1,
                    name=f"{regime} Pattern {idx + 1}",
                    count=int(len(members)),
                    hit_rate=round(hit_rate, 4),
                    avg_return=round(avg_return, 4),
                    direction=direction,
                    confidence=round(confidence, 4),
                    body_ratio=round(float(np.mean(np.abs(flat[members, -1]))), 4),
                    upper_wick_ratio=round(float(np.std(flat[members])), 4),
                    lower_wick_ratio=round(float(np.mean(np.abs(flat[members, 0]))), 4),
                    open_position=round(float(np.median(flat[members, -1])), 4),
                )
            )
        patterns.sort(key=lambda p: (p.confidence, p.hit_rate, p.count), reverse=True)
        return patterns[: self.max_patterns]

    def _discover_patterns_across_regimes(self, X_seq: np.ndarray, y: np.ndarray) -> List[PatternProfile]:
        if len(X_seq) == 0:
            self.last_pattern_debug = {"raw": 0, "kept": 0, "reason": "no_samples"}
            return []
        regime_size = max(1, len(X_seq) // max(1, self.regime_slices))
        discovered: List[PatternProfile] = []
        for slice_idx in range(self.regime_slices):
            start = slice_idx * regime_size
            end = len(X_seq) if slice_idx == self.regime_slices - 1 else min(len(X_seq), start + regime_size)
            if end - start < self.min_pattern_samples:
                continue
            regime_patterns = self._cluster_patterns(X_seq[start:end], y[start:end], regime=f"Regime {slice_idx + 1}")
            discovered.extend(regime_patterns)

        if not discovered:
            discovered = self._cluster_patterns(X_seq, y, regime="Global")

        discovered.sort(key=lambda p: (p.confidence, p.count, p.hit_rate), reverse=True)
        chosen = discovered[: max(self.pattern_top_k, min(12, len(discovered)))]
        self.last_pattern_debug = {
            "raw": len(discovered),
            "strict": len(discovered),
            "relaxed": len(discovered),
            "fallback": len(discovered),
            "kept": len(chosen),
        }
        return chosen

    def _match_pattern(
        self, seq_feat: np.ndarray, patterns: Optional[List[PatternProfile]] = None
    ) -> Tuple[Optional[int], str]:
        patterns = patterns if patterns is not None else self.patterns
        if not patterns:
            return None, "N/A"

        ranked = sorted(patterns, key=lambda p: (p.confidence, p.count, p.hit_rate), reverse=True)
        best = ranked[0]
        return best.pattern_id, best.name

    def _decide_signal(
        self,
        probability_up: float,
        pattern_id: Optional[int],
        patterns: Optional[List[PatternProfile]] = None,
        candle_metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], str]:
        metrics = candle_metrics or {}
        regime = str(metrics.get("regime", "balanced"))
        patterns = patterns if patterns is not None else self.patterns
        if pattern_id is None:
            if probability_up >= self.buy_threshold:
                return "CALL", f"ML bullish {probability_up:.3f}"
            if probability_up <= self.sell_threshold:
                return "PUT", f"ML bearish {probability_up:.3f}"
            return None, f"Neutral: prob={probability_up:.3f}"

        profile = next((p for p in patterns if p.pattern_id == pattern_id), None)
        if profile is None:
            if probability_up >= self.buy_threshold:
                return "CALL", f"ML bullish {probability_up:.3f}"
            if probability_up <= self.sell_threshold:
                return "PUT", f"ML bearish {probability_up:.3f}"
            return None, f"Neutral: prob={probability_up:.3f}"

        model_call = probability_up >= self.buy_threshold
        model_put = probability_up <= self.sell_threshold
        pattern_call = profile.direction == "CALL"
        pattern_put = profile.direction == "PUT"

        if regime == "bearish" and pattern_call:
            return None, f"{profile.name} skipped: bearish regime"
        if regime == "bullish" and pattern_put:
            return None, f"{profile.name} skipped: bullish regime"
        if regime == "balanced":
            return None, f"{profile.name} skipped: balanced regime"

        if pattern_call and model_call:
            return "CALL", f"{profile.name} bullish {probability_up:.3f} agree"
        if pattern_put and model_put:
            return "PUT", f"{profile.name} bearish {probability_up:.3f} agree"

        if model_call and not pattern_put:
            return "CALL", f"ML bullish {probability_up:.3f} via {profile.name}"
        if model_put and not pattern_call:
            return "PUT", f"ML bearish {probability_up:.3f} via {profile.name}"

        return None, f"{profile.name} neutral: prob={probability_up:.3f} regime={regime}"

    def _latest_features(self, closes: List[float]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        values = np.asarray(closes, dtype=float)
        if values.size < self.window_size:
            return None
        candles = self._approx_candles_from_closes(values.tolist())
        if len(candles) < self.window_size:
            return None
        window = candles[-self.window_size :]
        return self.feature_builder.extract_sequence(window), np.asarray([0.0], dtype=float)

    def _approx_candles_from_closes(self, closes: List[float]) -> List[Dict[str, Any]]:
        candles: List[Dict[str, Any]] = []
        if len(closes) < 2:
            return candles
        start_idx = max(1, len(closes) - self.window_size)
        for idx in range(start_idx, len(closes)):
            open_p = float(closes[idx - 1])
            close_p = float(closes[idx])
            high_p = max(open_p, close_p)
            low_p = min(open_p, close_p)
            candles.append({"open": open_p, "high": high_p, "low": low_p, "close": close_p})
        return candles

    def should_retrain(self, closes: List[float]) -> bool:
        if len(closes) < self.min_samples:
            return False
        if not self.model.fitted:
            return True
        return (len(closes) - self.last_train_size) >= self.retrain_every

    async def train_async(self, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.training_task and not self.training_task.done():
            return {"status": "training_in_progress"}

        async def _runner() -> Dict[str, Any]:
            return await asyncio.to_thread(self.train, candles)

        self.training_task = asyncio.create_task(_runner())
        return await self.training_task

    def train(self, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        values = list(candles[-self.history_window :])
        X_seq, y = self.feature_builder.build_dataset(values)
        if X_seq.size == 0 or y.size == 0:
            self.last_status = "Waiting for more history"
            return {"status": self.last_status, "samples": 0}

        metrics = self.model.fit(X_seq, y)
        self.patterns = self._discover_patterns_across_regimes(X_seq, y)
        self.model.save(self.model_path)
        self.last_metrics = metrics
        self.last_train_size = len(values)
        self.last_retrain_count += 1
        self.last_status = f"Trained on {len(X_seq)} samples | patterns={len(self.patterns)}"
        return {
            "status": self.last_status,
            "samples": int(X_seq.shape[0]),
            "metrics": metrics,
            "patterns": [p.__dict__ for p in self.patterns],
        }

    def analyze(self, data: List[Any], timeframe: int = 1) -> Tuple[Optional[str], Optional[int]]:
        decision = self.evaluate(data)
        return decision.signal, None

    def evaluate(self, data: List[Any]) -> MLDecision:
        ticks = [float(d["close"]) if isinstance(d, dict) else float(d) for d in data]
        if len(ticks) < self.window_size + 2:
            self.last_status = "Awaiting history"
            return MLDecision(None, None, None, "N/A", self.last_status)

        if not self.model.fitted:
            self.last_status = "Model not trained yet"
            return MLDecision(None, None, None, "N/A", self.last_status)

        features = self._latest_features(ticks)
        if features is None:
            self.last_status = "Insufficient data"
            return MLDecision(None, None, None, "N/A", self.last_status)

        seq_feat, _ = features
        probability_up = float(self.model.predict_proba(seq_feat)[0])
        candle_metrics = self._candle_metrics(self._approx_candles_from_closes(ticks))
        pattern_id, pattern_name = self._match_pattern(seq_feat)
        signal, status = self._decide_signal(probability_up, pattern_id, candle_metrics=candle_metrics)

        self.last_probability = probability_up
        self.last_pattern_id = pattern_id
        self.last_pattern_name = pattern_name
        self.last_status = status
        return MLDecision(signal, probability_up, pattern_id, pattern_name, status)

    def get_indicators(self, ticks: List[float]) -> Dict[str, Any]:
        decision = self.evaluate(ticks)
        payload: Dict[str, Any] = {
            "ml_status": decision.status,
            "ml_probability_up": round(decision.probability_up, 4) if decision.probability_up is not None else "N/A",
            "ml_signal": decision.signal or "WAIT",
            "ml_trained": bool(self.model.fitted),
            "ml_samples": int(self.model.train_samples),
            "ml_pattern": decision.pattern_name,
        }
        return payload
