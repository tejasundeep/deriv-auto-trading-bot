import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


class BotStorage:
    def __init__(self, db_path: Optional[str] = None):
        base_dir = os.path.dirname(__file__)
        default_path = os.path.join(base_dir, "data", "deriv_bot.sqlite3")
        self.db_path = db_path or default_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS candle_history (
                    symbol TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, epoch)
                );

                CREATE TABLE IF NOT EXISTS trade_history (
                    contract_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    stake REAL NOT NULL,
                    payout REAL NOT NULL,
                    profit REAL NOT NULL,
                    status TEXT NOT NULL,
                    entry_spot REAL,
                    exit_spot REAL,
                    traded_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    samples INTEGER NOT NULL,
                    accuracy REAL,
                    loss REAL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    total_samples INTEGER NOT NULL,
                    traded_signals INTEGER NOT NULL,
                    wins INTEGER NOT NULL,
                    losses INTEGER NOT NULL,
                    neutral INTEGER NOT NULL,
                    accuracy REAL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def save_candles(self, symbol: str, candles: Iterable[Dict[str, Any]]) -> int:
        rows = []
        for candle in candles:
            if not candle:
                continue
            if "epoch" not in candle:
                continue
            rows.append(
                (
                    symbol,
                    int(candle["epoch"]),
                    float(candle["open"]),
                    float(candle["high"]),
                    float(candle["low"]),
                    float(candle["close"]),
                )
            )

        if not rows:
            return 0

        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO candle_history(symbol, epoch, open, high, low, close)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    def fetch_candles(self, symbol: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        sql = """
            SELECT symbol, epoch, open, high, low, close
            FROM candle_history
            WHERE symbol = ?
        """
        params: List[Any] = [symbol]
        if limit is not None:
            sql += " ORDER BY epoch DESC LIMIT ?"
            params.append(int(limit))
        else:
            sql += " ORDER BY epoch ASC"

        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        if limit is not None:
            rows = list(reversed(rows))
        return [
            {
                "symbol": row["symbol"],
                "epoch": int(row["epoch"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
            for row in rows
        ]

    def fetch_closes(self, symbol: str, limit: Optional[int] = None) -> List[float]:
        candles = self.fetch_candles(symbol, limit=limit)
        return [float(c["close"]) for c in candles]

    def import_candles(self, symbol: str, candles: Iterable[Dict[str, Any]]) -> int:
        return self.save_candles(symbol, candles)

    def fetch_trades(self, symbol: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        sql = """
            SELECT contract_id, symbol, direction, stake, payout, profit, status,
                   entry_spot, exit_spot, traded_at, payload_json
            FROM trade_history
        """
        params: List[Any] = []
        clauses: List[str] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY traded_at DESC, contract_id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        items: List[Dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"]) if row["payload_json"] else {}
            items.append(
                {
                    "contract_id": row["contract_id"],
                    "symbol": row["symbol"],
                    "direction": row["direction"],
                    "stake": float(row["stake"]),
                    "payout": float(row["payout"]),
                    "profit": float(row["profit"]),
                    "status": row["status"],
                    "entry_spot": row["entry_spot"],
                    "exit_spot": row["exit_spot"],
                    "traded_at": row["traded_at"],
                    "payload": payload,
                }
            )
        return items

    def import_trades(self, trades: Iterable[Dict[str, Any]]) -> int:
        count = 0
        with self._lock, self._connect() as conn:
            for trade in trades:
                contract_id = str(trade.get("contract_id") or "")
                if not contract_id:
                    contract_id = f"imported-{uuid.uuid4().hex}"
                conn.execute(
                    """
                    INSERT OR REPLACE INTO trade_history (
                        contract_id, symbol, direction, stake, payout, profit, status,
                        entry_spot, exit_spot, traded_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        contract_id,
                        str(trade.get("symbol", "")),
                        str(trade.get("direction", "")),
                        float(trade.get("stake", 0.0)),
                        float(trade.get("payout", 0.0)),
                        float(trade.get("profit", 0.0)),
                        str(trade.get("status", "")),
                        trade.get("entry_spot"),
                        trade.get("exit_spot"),
                        str(trade.get("traded_at") or trade.get("time") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        json.dumps(trade, default=str),
                    ),
                )
                count += 1
            conn.commit()
        return count

    def save_trade(self, trade: Dict[str, Any]) -> None:
        payload = dict(trade)
        contract_id = str(payload.get("contract_id", ""))
        if not contract_id:
            return

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trade_history (
                    contract_id, symbol, direction, stake, payout, profit, status,
                    entry_spot, exit_spot, traded_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    contract_id,
                    str(payload.get("symbol", "")),
                    str(payload.get("direction", "")),
                    float(payload.get("stake", 0.0)),
                    float(payload.get("payout", 0.0)),
                    float(payload.get("profit", 0.0)),
                    str(payload.get("status", "")),
                    payload.get("entry_spot"),
                    payload.get("exit_spot"),
                    str(payload.get("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
                    json.dumps(payload, default=str),
                ),
            )
            conn.commit()

    def save_model_run(self, strategy_name: str, samples: int, accuracy: Optional[float], loss: Optional[float], status: str, payload: Dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_runs(strategy_name, samples, accuracy, loss, status, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_name,
                    int(samples),
                    accuracy,
                    loss,
                    status,
                    json.dumps(payload, default=str),
                ),
            )
            conn.commit()

    def save_backtest_run(self, strategy_name: str, symbol: str, summary: Dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO backtest_runs(
                    strategy_name, symbol, total_samples, traded_signals, wins, losses, neutral, accuracy, status, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_name,
                    symbol,
                    int(summary.get("total_samples", 0)),
                    int(summary.get("traded_signals", 0)),
                    int(summary.get("wins", 0)),
                    int(summary.get("losses", 0)),
                    int(summary.get("neutral", 0)),
                    summary.get("accuracy"),
                    str(summary.get("status", "ok")),
                    json.dumps(summary, default=str),
                ),
            )
            conn.commit()

    def get_latest_backtest(self, strategy_name: str, symbol: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM backtest_runs
                WHERE strategy_name = ? AND symbol = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (strategy_name, symbol),
            ).fetchone()
        if not row:
            return None
        payload = json.loads(row["payload_json"])
        payload["id"] = int(row["id"])
        payload["created_at"] = row["created_at"]
        return payload

    def fetch_backtest_runs(self, strategy_name: Optional[str] = None, symbol: Optional[str] = None, limit: Optional[int] = 20) -> List[Dict[str, Any]]:
        sql = """
            SELECT id, strategy_name, symbol, total_samples, traded_signals, wins, losses, neutral,
                   accuracy, status, created_at, payload_json
            FROM backtest_runs
        """
        params: List[Any] = []
        clauses: List[str] = []
        if strategy_name:
            clauses.append("strategy_name = ?")
            params.append(strategy_name)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        items: List[Dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"]) if row["payload_json"] else {}
            payload.update(
                {
                    "id": int(row["id"]),
                    "strategy_name": row["strategy_name"],
                    "symbol": row["symbol"],
                    "total_samples": int(row["total_samples"]),
                    "traded_signals": int(row["traded_signals"]),
                    "wins": int(row["wins"]),
                    "losses": int(row["losses"]),
                    "neutral": int(row["neutral"]),
                    "accuracy": row["accuracy"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                }
            )
            items.append(payload)
        return items

    def clear_ml_history(self) -> Dict[str, int]:
        with self._lock, self._connect() as conn:
            backtest_deleted = conn.execute("DELETE FROM backtest_runs").rowcount or 0
            model_deleted = conn.execute("DELETE FROM model_runs").rowcount or 0
            conn.commit()
        return {
            "backtest_runs_deleted": int(backtest_deleted),
            "model_runs_deleted": int(model_deleted),
        }
