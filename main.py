import asyncio
import os
import json
import logging
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
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
