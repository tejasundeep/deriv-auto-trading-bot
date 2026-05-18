import asyncio
import csv
import io
import os
import json
import logging
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import uvicorn

from config import PORT, HOST
from deriv_bot import DerivBot

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DerivBotServer")

# Initialize FastAPI App
app = FastAPI(
    title="Antigravity Deriv Bot",
    description="Asynchronous Auto-Trading Binary Bot with Glassmorphism Dashboard."
)

# Core trading bot instance
bot = DerivBot()

# Keep track of active dashboard connections
active_connections: List[WebSocket] = []

async def broadcast_telemetry(telemetry: dict):
    """
    Broadcasts the latest bot telemetry to all connected dashboard WebSockets.
    """
    if not active_connections:
        return

    # Create a copy of connections to safely remove disconnected clients during iteration
    for connection in list(active_connections):
        try:
            await connection.send_json(telemetry)
        except Exception:
            try:
                active_connections.remove(connection)
            except ValueError:
                pass

# Link the bot telemetry callback to our broadcast function
bot.on_state_change = broadcast_telemetry


@app.on_event("startup")
async def startup_event():
    """
    Triggers when the FastAPI server starts.
    Spawns the DerivBot lifecycle loop inside the shared async event loop.
    """
    logger.info("Initializing Core Trading Engine task...")
    asyncio.create_task(bot.run())


@app.get("/")
async def get_dashboard():
    """
    Serves the premium dashboard index.html file.
    """
    index_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(index_path):
        return {"error": "Dashboard template not found. Please verify templates/index.html exists."}
    return FileResponse(index_path)


def _csv_response(filename: str, rows: list[dict], fieldnames: list[str]):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/export/csv/candles")
async def export_candles_csv(symbol: str | None = None):
    target_symbol = symbol or bot.settings["symbol"]
    rows = bot.storage.fetch_candles(target_symbol)
    return _csv_response(
        f"{target_symbol}_candles.csv",
        rows,
        ["symbol", "epoch", "open", "high", "low", "close"],
    )


@app.get("/api/export/csv/trades")
async def export_trades_csv(symbol: str | None = None):
    rows = bot.storage.fetch_trades(symbol=symbol)
    return _csv_response(
        f"{(symbol or 'all')}_trades.csv",
        rows,
        ["contract_id", "symbol", "direction", "stake", "payout", "profit", "status", "entry_spot", "exit_spot", "traded_at"],
    )


@app.post("/api/import/csv/candles")
async def import_candles_csv(file: UploadFile = File(...), symbol: str | None = Form(None)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    raw = await file.read()
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        if not row.get("epoch"):
            continue
        rows.append(
            {
                "epoch": int(float(row["epoch"])),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        )

    target_symbol = symbol or (rows[0].get("symbol") if rows and rows[0].get("symbol") else bot.settings["symbol"])
    imported = bot.storage.import_candles(target_symbol, rows)
    if target_symbol == bot.settings["symbol"] and imported > 0:
        bot._load_cached_symbol_history(target_symbol)
        bot.trigger_ui_update()
    return JSONResponse({"status": "ok", "imported": imported, "symbol": target_symbol})


@app.post("/api/import/csv/trades")
async def import_trades_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    raw = await file.read()
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        if not row.get("contract_id"):
            row["contract_id"] = f"imported-{len(rows)}"
        rows.append(row)

    imported = bot.storage.import_trades(rows)
    bot.trigger_ui_update()
    return JSONResponse({"status": "ok", "imported": imported})


@app.websocket("/ws/client")
async def websocket_client_endpoint(websocket: WebSocket):
    """
    Handles local WebSocket connections from dashboard web clients.
    Exchanges telemetry snapshots and accepts control commands.
    """
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"Dashboard client connected. Active connections: {len(active_connections)}")

    try:
        # Immediately synchronize new client with current state
        await websocket.send_json(bot.get_telemetry())
        
        # Listen for client command actions
        while True:
            raw_message = await websocket.receive_text()
            try:
                message = json.loads(raw_message)
                action = message.get("action")
                payload = message.get("data", {})

                logger.info(f"Received action command: {action}")

                if action == "toggle_bot":
                    start_signal = payload.get("start", False)
                    bot.toggle_bot(start_signal)
                    
                elif action == "update_settings":
                    await bot.update_settings(payload)

                elif action == "toggle_account_mode":
                    mode = payload.get("account_mode", "Demo")
                    await bot.toggle_account_mode(mode)

                elif action == "train_ml":
                    await bot.train_ml_now()

                elif action == "run_backtest":
                    await bot.run_backtest_now()

                elif action == "reset_ml_model":
                    remove_checkpoint = bool(payload.get("remove_checkpoint", True))
                    await bot.reset_ml_model(remove_checkpoint)

                elif action == "reset_ml_registry":
                    remove_checkpoint = bool(payload.get("remove_checkpoint", True))
                    await bot.reset_ml_registry(remove_checkpoint)

            except json.JSONDecodeError:
                logger.warning("Received invalid non-JSON packet from dashboard client.")
            except Exception as cmd_err:
                logger.error(f"Error handling dashboard command: {str(cmd_err)}")

    except WebSocketDisconnect:
        logger.info("Dashboard client disconnected.")
    finally:
        try:
            active_connections.remove(websocket)
        except ValueError:
            pass
        logger.info(f"Active connections remaining: {len(active_connections)}")


if __name__ == "__main__":
    import sys
    
    PID_FILE = os.path.join(os.path.dirname(__file__), "server.pid")
    
    # Check for CLI shutdown flags
    if "--stop" in sys.argv or "stop" in sys.argv:
        if not os.path.exists(PID_FILE):
            print(f"[-] No running server found (missing {PID_FILE}).")
            sys.exit(0)
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
            
            print(f"[*] Stopping bot server with PID {pid}...")
            if sys.platform == "win32":
                # Silently and forcibly terminate process on Windows
                os.system(f"taskkill /F /PID {pid} >nul 2>&1")
            else:
                import signal
                os.kill(pid, signal.SIGTERM)
            
            print(f"[+] Bot server successfully stopped.")
        except Exception as e:
            print(f"[-] Failed to stop server process: {e}")
        finally:
            if os.path.exists(PID_FILE):
                try:
                    os.remove(PID_FILE)
                except Exception:
                    pass
        sys.exit(0)
        
    # Standard server launch: save current process ID to PID file
    try:
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))
    except Exception as e:
        logger.warning(f"Could not write PID file: {e}")

    logger.info(f"Starting server on http://{HOST}:{PORT}")
    try:
        uvicorn.run(app, host=HOST, port=PORT)
    finally:
        # Clean up PID file on clean/graceful shutdown
        if os.path.exists(PID_FILE):
            try:
                os.remove(PID_FILE)
            except Exception:
                pass
