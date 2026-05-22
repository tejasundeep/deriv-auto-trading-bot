import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm

from ml_engine import HybridMLStrategy


DEFAULT_CSV = Path(r"C:\Users\Teja\Desktop\po_bot\EURUSD_M1_ALL.csv")
DEFAULT_MODEL = Path("models") / "eurusd_sequence_model.pt"
DEFAULT_SUMMARY = Path("reports") / "eurusd_summary.json"
DEFAULT_CHECKPOINT = Path("checkpoints") / "eurusd_resume.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline EUR/USD trainer for the ML Pattern Engine."
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Path to the EUR/USD CSV file (expects date,time,open,high,low,close[,volume]).",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL),
        help="Where to save the trained model checkpoint.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Sequence window size used by the model.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=500,
        help="Minimum candles required before training.",
    )
    parser.add_argument(
        "--history-candles",
        type=int,
        default=500000,
        help="Maximum candles to keep in memory for training/backtest.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth candle. Use >1 to downsample very large files.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Backtest train/test split ratio.",
    )
    parser.add_argument(
        "--summary-json",
        default=str(DEFAULT_SUMMARY),
        help="Optional path to write a JSON summary report.",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=str(DEFAULT_CHECKPOINT),
        help="Where to persist resume state between interrupted runs.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from the latest checkpoint if available.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200000,
        help="Number of raw CSV rows to read per chunk when streaming.",
    )
    parser.add_argument(
        "--fit-mode",
        choices=["single", "chunked"],
        default="chunked",
        help="single loads up to --history-candles at once; chunked streams and trains batch by batch.",
    )
    parser.add_argument("--hidden-units", type=int, default=64, help="Model width for the hybrid sequence net.")
    parser.add_argument("--learning-rate", type=float, default=0.0005, help="Optimizer learning rate.")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs per fit call.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--l2-penalty", type=float, default=1e-4, help="Weight decay / L2 penalty.")
    parser.add_argument("--walk-train-window", type=int, default=100000, help="Training window size for walk-forward evaluation.")
    parser.add_argument("--walk-test-window", type=int, default=20000, help="Test window size for walk-forward evaluation.")
    parser.add_argument("--walk-step", type=int, default=20000, help="Step between walk-forward folds.")
    return parser.parse_args()


def _parse_dt(date_str: str, time_str: str) -> int:
    dt = datetime.strptime(f"{date_str} {time_str}", "%Y.%m.%d %H:%M")
    return int(dt.timestamp())


def _parse_timestamp(value: str) -> int:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Empty timestamp")
    try:
        dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError:
        dt = datetime.strptime(cleaned, "%Y.%m.%d %H:%M")
    return int(dt.timestamp())


def iter_candles(csv_path: Path, stride: int = 1) -> Iterable[Dict[str, Any]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for idx, row in enumerate(reader):
            if not row or len(row) < 6:
                continue
            if stride > 1 and (idx % stride) != 0:
                continue
            try:
                if idx == 0 and row[0].strip().lower() in {"timestamp", "date", "time"}:
                    continue
                if len(row) >= 6 and "-" in row[0] and ":" in row[0] and row[0].count("-") >= 2:
                    candle = {
                        "epoch": _parse_timestamp(row[0]),
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                    }
                else:
                    date_str, time_str = row[0].strip(), row[1].strip()
                    candle = {
                        "epoch": _parse_dt(date_str, time_str),
                        "open": float(row[2]),
                        "high": float(row[3]),
                        "low": float(row[4]),
                        "close": float(row[5]),
                    }
                yield candle
            except Exception:
                continue


def load_candles(csv_path: Path, max_candles: int, stride: int) -> List[Dict[str, Any]]:
    candles: List[Dict[str, Any]] = []
    for candle in iter_candles(csv_path, stride=stride):
        candles.append(candle)
        if max_candles > 0 and len(candles) >= max_candles:
            break
    return candles


def estimate_row_count(csv_path: Path, stride: int) -> int:
    total = 0
    with csv_path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for idx, row in enumerate(reader):
            if not row or len(row) < 6:
                continue
            if stride > 1 and (idx % stride) != 0:
                continue
            total += 1
    return total


def chunked_candles(csv_path: Path, chunk_size: int, stride: int) -> Iterable[List[Dict[str, Any]]]:
    buffer: List[Dict[str, Any]] = []
    for candle in iter_candles(csv_path, stride=stride):
        buffer.append(candle)
        if len(buffer) >= chunk_size:
            yield buffer
            buffer = buffer[-31:]
    if buffer:
        yield buffer


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        return {}
    try:
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_checkpoint(checkpoint_path: Path, payload: Dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def train_chunked(
    strategy: HybridMLStrategy,
    csv_path: Path,
    chunk_size: int,
    stride: int,
    checkpoint_path: Path,
    resume: bool,
    total_rows_hint: Optional[int] = None,
) -> Dict[str, Any]:
    last_result: Dict[str, Any] = {"status": "no_data", "samples": 0}
    total_samples = 0
    chunk_count = 0
    started_at = time.time()
    resume_state = load_checkpoint(checkpoint_path) if resume else {}
    resume_from_chunk = int(resume_state.get("chunk_count", 0)) if resume_state else 0
    if resume and resume_state:
        print(f"[resume] Found checkpoint: chunk={resume_from_chunk} | samples={resume_state.get('loaded_candles', 0)}")
        if Path(str(resume_state.get("model_path", strategy.model_path))).exists():
            restored = strategy.model.load(str(resume_state.get("model_path", strategy.model_path)))
            if restored:
                strategy.model = restored
                strategy.patterns = []
                print("[resume] Restored model weights from checkpoint")

    if not resume_state:
        strategy.model.reset()
        strategy.patterns = []

    print("[phase 1/2] Streaming training across the full CSV in chunks...")
    
    total_chunks = (total_rows_hint // chunk_size) + 1 if total_rows_hint else None
    pbar = tqdm(total=total_chunks, desc="Training", initial=resume_from_chunk)

    for chunk in chunked_candles(csv_path, chunk_size=chunk_size, stride=stride):
        chunk_count += 1
        if chunk_count <= resume_from_chunk:
            continue
        if len(chunk) < strategy.min_samples:
            pbar.update(1)
            continue

        X_seq, y = strategy.feature_builder.build_dataset(chunk)
        if X_seq.size == 0 or y.size == 0:
            pbar.update(1)
            continue

        if not strategy.model.fitted:
            last_result = strategy.train(chunk)
        else:
            last_result = strategy.model.fit(X_seq, y)
            strategy.patterns = strategy._discover_patterns_across_regimes(X_seq, y)
            strategy.last_metrics = last_result.get("metrics", {})
            strategy.last_status = f"Chunk-trained on {len(X_seq)} samples | patterns={len(strategy.patterns)}"
            last_result = {
                "status": strategy.last_status,
                "samples": int(X_seq.shape[0]),
                "metrics": last_result.get("metrics", {}),
                "patterns": [p.__dict__ for p in strategy.patterns],
            }

        total_samples += int(last_result.get("samples", 0))
        pbar.set_postfix({"samples": total_samples, "acc": last_result.get("metrics", {}).get("accuracy", 0)})
        pbar.update(1)

        save_checkpoint(
            checkpoint_path,
            {
                "csv": str(csv_path),
                "model_path": strategy.model_path,
                "chunk_count": chunk_count,
                "loaded_candles": total_samples,
                "stride": stride,
                "chunk_size": chunk_size,
                "updated_at": datetime.now().isoformat(),
            },
        )
        
    pbar.close()

    if chunk_count == 0:
        return {"status": "no_chunks", "samples": 0}

    return {
        "status": last_result.get("status", "chunked_complete"),
        "samples": total_samples,
        "metrics": last_result.get("metrics", {}),
        "patterns": last_result.get("patterns", []),
        "chunk_count": chunk_count,
        "elapsed_seconds": round(time.time() - started_at, 2),
    }


def walk_forward_backtest(
    candles: List[Dict[str, Any]],
    base_config: Dict[str, Any],
    train_window: int,
    test_window: int,
    step: int,
) -> Dict[str, Any]:
    if len(candles) < train_window + test_window:
        return {
            "status": "insufficient_data_for_walk_forward",
            "folds": 0,
            "total_samples": len(candles),
            "traded_signals": 0,
            "wins": 0,
            "losses": 0,
            "neutral": 0,
            "accuracy": None,
            "trade_accuracy": None,
            "coverage": 0.0,
        }

    fold_results: List[Dict[str, Any]] = []
    wins = losses = neutral = traded = 0
    model_hits = 0
    model_total = 0
    fold_idx = 0
    started_at = time.time()

    start = 0
    while start + train_window + test_window <= len(candles):
        fold_idx += 1
        train_slice = candles[start : start + train_window]
        test_slice = candles[start + train_window : start + train_window + test_window]

        fold_strategy = HybridMLStrategy(base_config)
        fold_strategy.train(train_slice)
        fold_backtest = fold_strategy.backtest(train_slice + test_slice, train_ratio=train_window / (train_window + test_window))

        wins += int(fold_backtest.get("wins", 0))
        losses += int(fold_backtest.get("losses", 0))
        neutral += int(fold_backtest.get("neutral", 0))
        traded += int(fold_backtest.get("traded_signals", 0))
        if fold_backtest.get("accuracy") is not None:
            model_total += int(fold_backtest.get("total_samples", 0))
            model_hits += int(round(float(fold_backtest["accuracy"]) * int(fold_backtest.get("total_samples", 0))))

        fold_results.append(
            {
                "fold": fold_idx,
                "train_samples": len(train_slice),
                "test_samples": len(test_slice),
                "backtest": fold_backtest,
            }
        )
        print(
            f"[walk-forward fold {fold_idx}] signals={fold_backtest.get('traded_signals', 0)} "
            f"| trade_acc={fold_backtest.get('trade_accuracy')} | coverage={fold_backtest.get('coverage')}"
        )
        elapsed = max(1e-6, time.time() - started_at)
        folds_left = max(0, (len(candles) - (fold_idx * (train_window + test_window))) // max(1, step))
        eta_seconds = int((elapsed / fold_idx) * folds_left) if fold_idx else 0
        print(
            f"[walk-forward] elapsed={int(elapsed)}s | eta={eta_seconds // 3600:02d}:{(eta_seconds % 3600) // 60:02d}:{eta_seconds % 60:02d}"
        )

        start += max(step, test_window)

    accuracy = round(model_hits / model_total, 4) if model_total else None
    trade_accuracy = round(wins / traded, 4) if traded else None
    coverage = round(traded / model_total, 4) if model_total else 0.0
    return {
        "status": "walk_forward_complete",
        "folds": len(fold_results),
        "total_samples": len(candles),
        "traded_signals": traded,
        "wins": wins,
        "losses": losses,
        "neutral": neutral,
        "accuracy": accuracy,
        "trade_accuracy": trade_accuracy,
        "coverage": coverage,
        "fold_results": fold_results,
        "elapsed_seconds": round(time.time() - started_at, 2),
    }


def walk_forward_backtest_stream(
    csv_path: Path,
    base_config: Dict[str, Any],
    train_window: int,
    test_window: int,
    step: int,
    stride: int,
) -> Dict[str, Any]:
    window_size = train_window + test_window
    buffer: List[Dict[str, Any]] = []
    fold_results: List[Dict[str, Any]] = []
    wins = losses = neutral = traded = 0
    model_hits = 0
    model_total = 0
    fold_idx = 0
    consumed = 0
    carry = 0

    print("[phase 2/2] Streaming walk-forward backtest across the full CSV...")

    for candle in iter_candles(csv_path, stride=stride):
        buffer.append(candle)
        consumed += 1
        if len(buffer) < window_size:
            continue

        carry += 1
        if carry < max(1, step):
            continue

        fold_idx += 1
        fold_slice = buffer[-window_size:]
        train_slice = fold_slice[:train_window]
        test_slice = fold_slice[train_window:]

        fold_strategy = HybridMLStrategy(base_config)
        fold_strategy.train(train_slice)
        fold_backtest = fold_strategy.backtest(train_slice + test_slice, train_ratio=train_window / window_size)

        wins += int(fold_backtest.get("wins", 0))
        losses += int(fold_backtest.get("losses", 0))
        neutral += int(fold_backtest.get("neutral", 0))
        traded += int(fold_backtest.get("traded_signals", 0))
        if fold_backtest.get("accuracy") is not None:
            model_total += int(fold_backtest.get("total_samples", 0))
            model_hits += int(round(float(fold_backtest["accuracy"]) * int(fold_backtest.get("total_samples", 0))))

        fold_results.append(
            {
                "fold": fold_idx,
                "train_samples": len(train_slice),
                "test_samples": len(test_slice),
                "backtest": fold_backtest,
            }
        )
        print(
            f"[walk-forward fold {fold_idx}] signals={fold_backtest.get('traded_signals', 0)} "
            f"| trade_acc={fold_backtest.get('trade_accuracy')} | coverage={fold_backtest.get('coverage')}"
        )

        buffer = buffer[max(0, len(buffer) - test_window):]
        carry = 0

    accuracy = round(model_hits / model_total, 4) if model_total else None
    trade_accuracy = round(wins / traded, 4) if traded else None
    coverage = round(traded / model_total, 4) if model_total else 0.0
    return {
        "status": "walk_forward_complete",
        "folds": len(fold_results),
        "total_samples": consumed,
        "traded_signals": traded,
        "wins": wins,
        "losses": losses,
        "neutral": neutral,
        "accuracy": accuracy,
        "trade_accuracy": trade_accuracy,
        "coverage": coverage,
        "fold_results": fold_results,
    }


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    config = {
        "ml_window_size": args.window_size,
        "ml_min_samples": args.min_samples,
        "ml_history_candles": args.history_candles,
        "ml_model_path": args.model_path,
        "ml_buy_threshold": 0.58,
        "ml_sell_threshold": 0.42,
        "ml_retrain_every": 60,
        "ml_hidden_units": args.hidden_units,
        "ml_learning_rate": args.learning_rate,
        "ml_epochs": args.epochs,
        "ml_l2_penalty": args.l2_penalty,
        "ml_batch_size": args.batch_size,
        "ml_max_patterns": 12,
        "ml_min_pattern_samples": 50,
        "ml_pattern_min_hit_rate": 0.62,
        "ml_pattern_top_k": 3,
        "ml_pattern_confidence_threshold": 0.55,
        "ml_regime_slices": 4,
    }

    strategy = HybridMLStrategy(config)

    if args.fit_mode == "single":
        print(f"[load] Reading candles from: {csv_path}")
        candles = load_candles(csv_path, max_candles=args.history_candles, stride=args.stride)
        print(f"[load] Loaded {len(candles):,} candles")
        if len(candles) < args.min_samples:
            raise RuntimeError(
                f"Not enough candles for training: {len(candles)} < {args.min_samples}"
            )
        print("[train] Training model...")
        train_result = strategy.train(candles)
        loaded_count = len(candles)
        backtest_source = candles
        checkpoint_path = Path(args.checkpoint_file).expanduser().resolve()
    else:
        print(f"[load] Streaming candles from: {csv_path}")
        print(f"[train] Chunked training enabled | chunk_size={args.chunk_size:,} | stride={args.stride}")
        estimated_rows = estimate_row_count(csv_path, stride=args.stride)
        print(f"[progress] Estimated usable rows: {estimated_rows:,}")
        checkpoint_path = Path(args.checkpoint_file).expanduser().resolve()
        train_result = train_chunked(
            strategy,
            csv_path,
            chunk_size=args.chunk_size,
            stride=args.stride,
            checkpoint_path=checkpoint_path,
            resume=args.resume,
            total_rows_hint=estimated_rows,
        )
        loaded_count = int(train_result.get("samples", 0))
        backtest_source = None

    print(json.dumps(train_result, indent=2, default=str))

    print("[backtest] Running walk-forward backtest...")
    if backtest_source is not None:
        wf_train_window = min(args.walk_train_window, max(1, len(backtest_source) - args.walk_test_window))
        wf_test_window = min(args.walk_test_window, max(1, len(backtest_source) // 10))
        backtest_result = walk_forward_backtest(
            candles=backtest_source,
            base_config=config,
            train_window=wf_train_window,
            test_window=wf_test_window,
            step=args.walk_step,
        )
    else:
        wf_train_window = max(args.walk_train_window, args.window_size + 1)
        wf_test_window = max(args.walk_test_window, args.window_size + 1)
        backtest_result = walk_forward_backtest_stream(
            csv_path=csv_path,
            base_config=config,
            train_window=wf_train_window,
            test_window=wf_test_window,
            step=max(1, args.walk_step),
            stride=args.stride,
        )
    print(json.dumps(backtest_result, indent=2, default=str))

    summary = {
        "csv": str(csv_path),
        "loaded_candles": loaded_count,
        "train_result": train_result,
        "backtest_result": backtest_result,
        "model_path": args.model_path,
    }

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"[save] Summary written to: {summary_path}")

    if checkpoint_path.exists():
        print(f"[resume] Checkpoint saved at: {checkpoint_path}")
    print(f"[save] Model checkpoint saved to: {args.model_path}")
    print(f"[done] Total elapsed: {backtest_result.get('elapsed_seconds', 0)} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
