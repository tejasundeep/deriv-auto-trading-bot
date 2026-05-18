import asyncio
import json
import logging
import time
import traceback
import uuid
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Coroutine
import websockets
from websockets.exceptions import ConnectionClosed

import config
from strategies import get_strategy, get_risk_manager, BaseStrategy, BaseRiskManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DerivBotCore")

class DerivBot:
    def __init__(self):
        # Bot configurations & parameters (loaded from config/defaults)
        self.api_token: str = config.DERIV_API_TOKEN
        self.app_id: str = str(config.DERIV_APP_ID)
        
        self.settings: Dict[str, Any] = {
            "symbol": config.DEFAULT_SYMBOL,
            "stake": config.DEFAULT_AMOUNT,
            "duration": 1,
            "duration_unit": "m",
            "strategy": config.DEFAULT_STRATEGY,
            
            # Money Management Configurations
            "money_management": "martingale",
            "martingale_multiplier": 2.0,
            "martingale_max_steps": 5,
            "fibonacci_max_steps": 8,
            "dalembert_increment": 1.0,
            "oscars_grind_target": 1.0,
            
            # Strategy / Indicator Parameters
            "rsi_period": 14,
            "rsi_lower_bound": 30.0,
            "rsi_upper_bound": 70.0,
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "ema_fast": 9,
            "ema_slow": 21,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "velocity_period": 10,
            "velocity_threshold_sd": 1.5,
            "tick_trend_consecutive": 3,

            "target_profit": config.DEFAULT_TARGET_PROFIT,
            "stop_loss": config.DEFAULT_STOP_LOSS,
            "currency": config.DEFAULT_CURRENCY
        }

        # Bot Runtime States
        self.is_running: bool = False             # Is the bot currently running strategy?
        self.is_connected: bool = False
        self.authorized: bool = False
        
        self.balance: float = 0.0
        self.currency: str = "USD"
        self.email: str = ""
        self.account_type: str = "Demo"
        self.account_mode: str = "Demo"
        
        self.candles: List[Dict[str, Any]] = []    # Rolling 1-minute candles array
        self.ticks: List[float] = []               # Closed candle closing prices
        self.tick_epochs: List[int] = []           # Closed candle epochs
        
        self.active_trade: Optional[Dict[str, Any]] = None  # Info about the current open contract
        self.placing_trade: bool = False                    # Safety flag to prevent overlapping orders
        self.trade_history: List[Dict[str, Any]] = []       # List of completed contracts
        
        # Statistics
        self.total_profit_loss: float = 0.0
        self.wins: int = 0
        self.losses: int = 0
        
        self.logs: List[str] = []                  # Real-time console logs

        # Asynchronous helper structures
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.pending_requests: Dict[Any, asyncio.Future] = {}
        self.req_id_counter: int = 1000
        
        # Managers
        # Drawdown and Performance Tracking
        self.peak_balance: float = 0.0
        self.max_drawdown: float = 0.0

        # Strategy and Risk Managers
        self.strategy: BaseStrategy = get_strategy(self.settings["strategy"], self.settings)
        self.risk_manager: BaseRiskManager = get_risk_manager(
            money_management=self.settings.get("money_management", "martingale"),
            base_stake=self.settings["stake"],
            config=self.settings
        )

        # Dashboard callback for pushing state changes
        self.on_state_change: Optional[Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]] = None

    def log(self, message: str):
        """Adds a log entry with a timestamp and triggers UI update."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_log = f"[{timestamp}] {message}"
        logger.info(formatted_log)
        
        # Save log (limit to 100 entries to save memory)
        self.logs.append(formatted_log)
        if len(self.logs) > 100:
            self.logs.pop(0)
            
        self.trigger_ui_update()

    def trigger_ui_update(self):
        """Asynchronously schedule a state broadcast to dashboard client."""
        if self.on_state_change:
            asyncio.create_task(self.on_state_change(self.get_telemetry()))

    def get_telemetry(self) -> Dict[str, Any]:
        """Prepares a thread-safe snapshot of the system state for telemetry."""
        win_rate = 0.0
        total_trades = self.wins + self.losses
        if total_trades > 0:
            win_rate = round((self.wins / total_trades) * 100, 2)

        indicators_hud = {}
        if self.ticks:
            try:
                indicators_hud = self.strategy.get_indicators(self.ticks)
            except Exception as ind_err:
                indicators_hud = {"status": f"Error: {str(ind_err)}"}

        drawdown_pct = 0.0
        if self.peak_balance > 0.0:
            drawdown_pct = round((self.max_drawdown / self.peak_balance) * 100, 2)

        return {
            "is_running": self.is_running,
            "is_connected": self.is_connected,
            "authorized": self.authorized,
            "balance": round(self.balance, 2),
            "currency": self.currency,
            "email": self.email,
            "account_type": self.account_type,
            "account_mode": self.account_mode,
            "ticks": self.ticks[-120:],  # Limit to 120 latest ticks to populate candlesticks
            "tick_epochs": self.tick_epochs[-120:],
            "candles": self.candles[-60:],  # Send the last 60 actual OHLC candles for precise financial charts
            "active_trade": self.active_trade,
            "trade_history": self.trade_history[-20:], # Send 20 latest history items
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": win_rate,
            "total_profit_loss": round(self.total_profit_loss, 2),
            "settings": self.settings,
            "logs": self.logs,
            "current_stake": self.risk_manager.get_stake(),
            "peak_balance": round(self.peak_balance, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": drawdown_pct,
            "indicators_hud": indicators_hud
        }

    async def update_settings(self, new_settings: Dict[str, Any]):
        """Safely updates trading settings on-the-fly."""
        old_symbol = self.settings["symbol"]
        old_strategy = self.settings["strategy"]
        
        # Update settings dictionary
        for k, v in new_settings.items():
            if k in self.settings:
                # Typecast appropriately
                if k in ["stake", "target_profit", "stop_loss", "martingale_multiplier", 
                         "dalembert_increment", "oscars_grind_target", 
                         "rsi_lower_bound", "rsi_upper_bound", "bb_std_dev", "velocity_threshold_sd"]:
                    self.settings[k] = float(v)
                elif k in ["duration", "martingale_max_steps", "fibonacci_max_steps", 
                           "rsi_period", "bb_period", "ema_fast", "ema_slow", 
                           "macd_fast", "macd_slow", "macd_signal", "velocity_period", 
                           "tick_trend_consecutive"]:
                    self.settings[k] = int(v)
                else:
                    self.settings[k] = v
            else:
                self.settings[k] = v

        # Enforce locked pure 1-minute trading
        self.settings["duration"] = 1
        self.settings["duration_unit"] = "m"

        # Dynamic instantiation of the strategy and risk manager with the updated settings dictionary
        self.risk_manager = get_risk_manager(
            money_management=self.settings.get("money_management", "martingale"),
            base_stake=self.settings["stake"],
            config=self.settings
        )

        self.strategy = get_strategy(self.settings["strategy"], self.settings)
        self.log(f"Settings Refreshed: Strategy={self.strategy.name} | MoneyManagement={self.settings.get('money_management')}")

        # Check if symbol subscription needs to be updated
        if self.settings["symbol"] != old_symbol and self.is_connected:
            self.candles = []
            self.ticks = []
            self.tick_epochs = []
            await self._subscribe_candles(self.settings["symbol"])
            await self._unsubscribe_candles(old_symbol)
            self.log(f"Switched candle subscription to: {self.settings['symbol']}")

        self.log("Settings updated successfully.")
        self.trigger_ui_update()

    async def toggle_account_mode(self, mode: str):
        """Switches the trading account mode between Demo and Real, then restarts the socket session."""
        if mode not in ["Demo", "Real"]:
            return
            
        if self.account_mode == mode:
            return
            
        if self.is_running:
            self.log("[WARNING] Cannot switch account mode while AutoTrade is active. Please stop the bot first.")
            return
            
        self.log(f"Switching account mode: [{self.account_mode}] -> [{mode}]")
        self.account_mode = mode
        
        # Trigger reconnection by disconnecting current WebSocket if active
        if self.ws:
            self.log("Closing active WebSocket stream to initiate reconnection in new mode...")
            await self.ws.close()

    async def run(self):
        """Main lifecycle of the bot. Runs connection and logic loops."""
        self.log("Starting trading engine lifecycle...")
        
        while True:
            try:
                # 1. Determine WebSocket URL and authentication status
                ws_url = "wss://api.derivws.com/trading/v1/options/ws/public"
                is_demo_or_real = False
                
                if self.api_token:
                    self.log("Exchanging API token for WebSocket OTP...")
                    headers = {
                        "Deriv-App-ID": str(self.app_id),
                        "Authorization": f"Bearer {self.api_token}"
                    }
                    
                    # Run REST requests in a threadpool to remain fully asynchronous and non-blocking
                    def fetch_otp(mode):
                        accounts_url = "https://api.derivws.com/trading/v1/options/accounts"
                        resp = requests.get(accounts_url, headers=headers, timeout=10.0)
                        if resp.status_code != 200:
                            raise ConnectionError(f"REST Account Fetch Failed (Status {resp.status_code}): {resp.text}")
                        
                        accounts_data = resp.json()
                        accounts = accounts_data.get("data", [])
                        if not accounts:
                            raise ValueError("No active trading accounts found on this token.")
                        
                        # Filter accounts to find the one matching user requested mode (Real vs Demo)
                        target_type = "real" if mode == "Real" else "demo"
                        active_account = None
                        warning_msg = None
                        
                        for acc in accounts:
                            if acc.get("account_type") == target_type:
                                active_account = acc
                                break
                                
                        if not active_account:
                            warning_msg = f"[WARNING] Requested account mode [{mode}] was not found on this token! Defaulting to first available account."
                            active_account = accounts[0]
                        
                        account_id = active_account.get("account_id")
                        balance = float(active_account.get("balance", 0.0))
                        currency = active_account.get("currency", "USD")
                        email = active_account.get("email", "Deriv Client")
                        account_type = "Demo" if active_account.get("account_type") == "demo" else "Real"
                        
                        otp_endpoint = f"https://api.derivws.com/trading/v1/options/accounts/{account_id}/otp"
                        otp_resp = requests.post(otp_endpoint, headers=headers, timeout=10.0)
                        if otp_resp.status_code != 200:
                            raise ConnectionError(f"REST OTP Request Failed (Status {otp_resp.status_code}): {otp_resp.text}")
                        
                        otp_data = otp_resp.json()
                        url = otp_data["data"]["url"]
                        return url, balance, currency, email, account_type, warning_msg
                    
                    # Execute fetch_otp asynchronously via to_thread, passing self.account_mode
                    url, balance, currency, email, account_type, warning_msg = await asyncio.to_thread(fetch_otp, self.account_mode)
                    
                    self.balance = balance
                    self.currency = currency
                    self.email = email
                    self.account_type = account_type
                    self.authorized = True
                    ws_url = url
                    is_demo_or_real = True
                    
                    if warning_msg:
                        self.log(warning_msg)
                    self.log(f"Authenticated successfully via OTP! Account: {self.email} ({self.account_type}) | Balance: {self.currency} {self.balance}")
                else:
                    self.log("[WARNING] No API token supplied. Dashboard will run in view-only mode.")
                    self.authorized = False
                
                self.log(f"Connecting to Deriv WebSocket: {ws_url}...")
                
                async with websockets.connect(ws_url) as ws:
                    self.ws = ws
                    self.is_connected = True
                    self.log("WebSocket connection established successfully.")
                    
                    # Start message listener and keep-alive heartbeat ping tasks
                    listener_task = asyncio.create_task(self._message_listener())
                    ping_task = asyncio.create_task(self._heartbeat())
                    
                    if is_demo_or_real:
                        self.log("Activating balance and contract subscription feeds...")
                        # Subscribe to balance updates
                        await self.ws.send(json.dumps({"balance": 1, "subscribe": 1}))
                        # Subscribe to open contract updates for real-time tracking
                        await self.ws.send(json.dumps({"proposal_open_contract": 1, "subscribe": 1}))
                        self.log("Feeds active. Ready to trade.")
                    
                    # Initial symbol subscription to 1-minute candles
                    await self._subscribe_candles(self.settings["symbol"])

                    # Wait for listener to complete or raise exception
                    await listener_task
                    
            except (ConnectionClosed, Exception) as e:
                self.is_connected = False
                self.authorized = False
                self.ws = None
                self.log(f"[ERROR] Connection lost or exception occurred: {str(e)}")
                
                # Check for rate limiting errors (HTTP 429 / Cloudflare 1015) to apply a longer cooling backoff
                err_str = str(e)
                delay = 5
                if "429" in err_str or "1015" in err_str or "Rate Limit" in err_str:
                    delay = 30
                    self.log("[RATE LIMIT] Temporary API rate limit or Cloudflare block detected. Cooling down connection for 30 seconds...")
                else:
                    self.log("Attempting reconnection in 5 seconds...")
                
                # Cleanup pending futures
                for fut in self.pending_requests.values():
                    if not fut.done():
                        fut.set_exception(e)
                self.pending_requests.clear()
                
                await asyncio.sleep(delay)

    async def _heartbeat(self):
        """Sends keep-alive pings to Deriv every 30 seconds."""
        while self.is_connected and self.ws:
            try:
                await asyncio.sleep(30)
                await self.ws.send(json.dumps({"ping": 1}))
            except Exception:
                break

    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to send a request over WebSocket and await the specific response via req_id."""
        if not self.ws or not self.is_connected:
            raise ConnectionError("WebSocket is not connected.")

        # Ensure unique integer req_id is attached (Deriv requires positive integers for req_id)
        req_id = request.get("req_id")
        if req_id is None:
            self.req_id_counter += 1
            req_id = self.req_id_counter
        request["req_id"] = req_id

        # Create a Future to await the response
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.pending_requests[req_id] = future

        # Send request
        await self.ws.send(json.dumps(request))

        # Await future with timeout
        try:
            response = await asyncio.wait_for(future, timeout=10.0)
            return response
        except asyncio.TimeoutError:
            self.pending_requests.pop(req_id, None)
            raise TimeoutError(f"Request {request.get('msg_type')} timed out.")


    async def _subscribe_candles(self, symbol: str):
        """Subscribes to live 1-minute candle history and updates for the specified asset."""
        if self.ws:
            self.log(f"Subscribing to 1-minute candle history updates for {symbol}...")
            await self.ws.send(json.dumps({
                "ticks_history": symbol,
                "style": "candles",
                "granularity": 60,
                "subscribe": 1,
                "end": "latest",
                "count": 100
            }))

    async def _unsubscribe_candles(self, symbol: str):
        """Unsubscribes from live 1-minute candle updates."""
        if self.ws:
            self.log(f"Unsubscribing from 1-minute candle updates for {symbol}...")
            await self.ws.send(json.dumps({"forget_all": "candles"}))

    async def _message_listener(self):
        """Asynchronously listens and processes all incoming messages from Deriv."""
        async for message in self.ws:
            try:
                data = json.loads(message)
                msg_type = data.get("msg_type")

                # Check if this corresponds to a pending req_id future
                req_id = data.get("req_id")
                if req_id is not None:
                    future = None
                    if req_id in self.pending_requests:
                        future = self.pending_requests.pop(req_id)
                    elif str(req_id) in self.pending_requests:
                        future = self.pending_requests.pop(str(req_id))
                    elif int(req_id) in self.pending_requests:
                        try:
                            future = self.pending_requests.pop(int(req_id))
                        except ValueError:
                            pass
                    
                    if future and not future.done():
                        future.set_result(data)

                # Process specific API message types
                if "error" in data and msg_type not in ["proposal", "buy"]:
                    # Global errors
                    self.log(f"[SERVER ERROR] {data['error']['message']}")
                    continue

                if msg_type == "candles":
                    candles_list = data.get("candles", [])
                    self.candles = [
                        {
                            "open": float(c["open"]),
                            "high": float(c["high"]),
                            "low": float(c["low"]),
                            "close": float(c["close"]),
                            "epoch": int(c["epoch"])
                        }
                        for c in candles_list
                    ]
                    # Update ticks arrays to align with closed candle closing prices (self.candles[:-1] represents fully closed candles!)
                    if len(self.candles) > 1:
                        self.ticks = [c["close"] for c in self.candles[:-1]]
                        self.tick_epochs = [c["epoch"] for c in self.candles[:-1]]
                    else:
                        self.ticks = [c["close"] for c in self.candles]
                        self.tick_epochs = [c["epoch"] for c in self.candles]
                        
                    self.log(f"Initialized rolling candlestick history with {len(self.candles)} closed candles.")
                    self.trigger_ui_update()

                elif msg_type == "ohlc":
                    ohlc = data.get("ohlc")
                    if ohlc and ohlc["symbol"] == self.settings["symbol"] and int(ohlc["granularity"]) == 60:
                        current_epoch = int(ohlc["open_time"])
                        candle_data = {
                            "open": float(ohlc["open"]),
                            "high": float(ohlc["high"]),
                            "low": float(ohlc["low"]),
                            "close": float(ohlc["close"]),
                            "epoch": current_epoch
                        }
                        
                        if not self.candles:
                            self.candles.append(candle_data)
                            self.ticks = [c["close"] for c in self.candles]
                            self.tick_epochs = [c["epoch"] for c in self.candles]
                        else:
                            # Check if the epoch matches the latest one in our history (meaning it's an active/open update)
                            if self.candles[-1]["epoch"] == current_epoch:
                                self.candles[-1] = candle_data
                            elif self.candles[-1]["epoch"] < current_epoch:
                                # A new minute epoch has started! This means the previous candle has just CLOSED!
                                # 1. Finalize the closed candle closes inside self.ticks/self.tick_epochs
                                self.ticks = [c["close"] for c in self.candles]
                                self.tick_epochs = [c["epoch"] for c in self.candles]
                                
                                # 2. Append the new active/open candle
                                self.candles.append(candle_data)
                                if len(self.candles) > 150:
                                    self.candles.pop(0)
                                    
                                # 3. Trigger 1-minute strategy check precisely on this candle-close transition!
                                await self.process_candle_close()
                                
                        self.trigger_ui_update()

                elif msg_type == "balance":
                    bal = data.get("balance")
                    if bal:
                        self.balance = float(bal["balance"])
                        self.currency = bal["currency"]
                        
                        # Update drawdown analytics
                        if self.peak_balance == 0.0:
                            self.peak_balance = self.balance
                        else:
                            self.peak_balance = max(self.peak_balance, self.balance)
                            
                        if self.peak_balance > 0.0:
                            drawdown = self.peak_balance - self.balance
                            if drawdown > 0:
                                self.max_drawdown = max(self.max_drawdown, drawdown)
                                
                        self.trigger_ui_update()

                elif msg_type == "proposal_open_contract":
                    poc = data.get("proposal_open_contract")
                    if poc:
                        await self.process_contract_update(poc)

            except Exception as e:
                logger.error(f"Error processing message: {str(e)}\n{traceback.format_exc()}")

    async def process_candle_close(self):
        """Decision loop triggered strictly on every closed 1-minute candle boundary transition."""
        self.trigger_ui_update()

        if not self.is_running or not self.authorized:
            return

        # Check safety profit/loss limits
        if self.total_profit_loss >= self.settings["target_profit"]:
            self.log(f"[STOP] TARGET PROFIT REACHED! Total: +${self.total_profit_loss:.2f}. Disabling bot.")
            self.is_running = False
            self.trigger_ui_update()
            return
        
        if self.total_profit_loss <= -self.settings["stop_loss"]:
            self.log(f"[STOP] STOP LOSS HIT! Total: -${abs(self.total_profit_loss):.2f}. Disabling bot.")
            self.is_running = False
            self.trigger_ui_update()
            return

        # If a trade is active or currently being placed, do not analyze for new signals
        if self.active_trade or self.placing_trade:
            return

        # Analyze candle closes (ticks lists represents closed candles)
        signal = self.strategy.analyze(self.ticks)
        if signal in ["CALL", "PUT"]:
            self.log(f"Strategy Close Signal Detected: [{signal}]. Executing 1-Minute contract buy...")
            asyncio.create_task(self.execute_trade(signal))

    async def execute_trade(self, direction: str):
        """Executes a dual-step binary contract purchase (Proposal -> Buy)."""
        if self.placing_trade:
            return
            
        self.placing_trade = True
        stake = self.risk_manager.get_stake()
        
        self.log(f"Initiating order. Direction: {direction} | Stake: {self.currency} {stake}")
        
        try:
            # 1. Request Price Proposal
            proposal_req = {
                "proposal": 1,
                "amount": stake,
                "basis": "stake",
                "contract_type": direction,
                "currency": self.currency,
                "duration": self.settings["duration"],
                "duration_unit": self.settings["duration_unit"],
                "underlying_symbol": self.settings["symbol"]
            }
            
            res_proposal = await self._send_request(proposal_req)
            
            if "error" in res_proposal:
                self.log(f"[PROPOSAL REJECTED] {res_proposal['error']['message']}")
                self.placing_trade = False
                return

            prop = res_proposal["proposal"]
            proposal_id = prop["id"]
            ask_price = float(prop["ask_price"])
            
            # 2. Purchase Contract
            self.log(f"Proposal acquired. ID: {proposal_id[:10]}... | Ask Price: {ask_price}. Submitting buy order...")
            
            buy_req = {
                "buy": proposal_id,
                "price": ask_price
            }
            
            res_buy = await self._send_request(buy_req)
            
            if "error" in res_buy:
                self.log(f"[BUY ORDER REJECTED] {res_buy['error']['message']}")
                self.placing_trade = False
                return
            
            # Store temporary active trade information
            buy_info = res_buy["buy"]
            self.active_trade = {
                "contract_id": buy_info["contract_id"],
                "transaction_id": buy_info["transaction_id"],
                "direction": direction,
                "stake": ask_price,
                "symbol": self.settings["symbol"],
                "status": "open",
                "progress": 0,
                "profit": 0.0,
                "payout": 0.0,
                "start_time": datetime.now().strftime("%H:%M:%S")
            }
            
            self.log(f"Contract Purchased successfully! ID: {buy_info['contract_id']} | Stake: ${ask_price}")
            self.trigger_ui_update()

        except Exception as e:
            self.log(f"[TRADE FAILURE] Execution error: {str(e)}")
            self.placing_trade = False

    async def process_contract_update(self, poc: Dict[str, Any]):
        """Processes updates on active open contracts in real-time."""
        if not self.active_trade or poc["contract_id"] != self.active_trade["contract_id"]:
            return

        status = poc["status"]
        self.active_trade["profit"] = float(poc["profit"])
        self.active_trade["payout"] = float(poc.get("payout", 0.0))
        
        # Calculate progress in seconds (for 1-minute options)
        if "date_start" in poc and "date_expiry" in poc:
            total_dur = int(poc["date_expiry"]) - int(poc["date_start"])
            if total_dur > 0:
                elapsed = int(time.time()) - int(poc["date_start"])
                self.active_trade["progress"] = min(elapsed, total_dur)
            
        if status in ["won", "lost", "sold"]:
            # Contract is finalized
            profit_loss = float(poc["profit"])
            payout = float(poc.get("sell_price", 0.0))
            
            contract_history = {
                "contract_id": poc["contract_id"],
                "symbol": self.active_trade["symbol"],
                "direction": self.active_trade["direction"],
                "stake": self.active_trade["stake"],
                "payout": payout,
                "profit": profit_loss,
                "status": status,
                "entry_spot": poc.get("entry_tick"),
                "exit_spot": poc.get("exit_tick"),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            self.trade_history.append(contract_history)
            self.total_profit_loss += profit_loss
            
            if status == "won":
                self.wins += 1
                self.log(f"[TRADE WIN] Contract {poc['contract_id']} won! Profit: +${profit_loss:.2f}")
                self.risk_manager.on_win()
            else:
                self.losses += 1
                self.log(f"[TRADE LOSS] Contract {poc['contract_id']} lost! Loss: -${abs(profit_loss):.2f}")
                self.risk_manager.on_loss()

            # Reset flags
            self.active_trade = None
            self.placing_trade = False
            self.trigger_ui_update()
        else:
            # Still open, update UI
            self.trigger_ui_update()
            
    def toggle_bot(self, start: bool):
        """Starts or stops the strategy analyzer execution."""
        if start == self.is_running:
            return

        if start:
            if not self.authorized:
                self.log("[ERROR] Cannot start bot. Sessions is not authorized. Check API token.")
                return
            
            # Reset session statistics for a fresh run
            self.total_profit_loss = 0.0
            self.wins = 0
            self.losses = 0
            self.peak_balance = self.balance
            self.max_drawdown = 0.0
            self.risk_manager.reset()
            
            self.is_running = True
            self.log(f"Bot STARTED. Monitoring {self.settings['symbol']} using strategy: {self.strategy.name}...")
        else:
            self.is_running = False
            self.log("Bot STOPPED. Monitoring suspended.")
            self.risk_manager.reset()  # Reset Risk multipliers on stop
            self.placing_trade = False
            
        self.trigger_ui_update()
