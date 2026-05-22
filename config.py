import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Deriv Platform Configurations
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "")

# Load App ID safely as a pure string (no integer casting needed for modern Options API)
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089").strip()
if not DERIV_APP_ID:
    DERIV_APP_ID = "1089"

# Dashboard Server Configurations
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "127.0.0.1")

# Default Bot Parameters
DEFAULT_SYMBOL = "frxEURUSD"      # EUR/USD Forex Pair
DEFAULT_AMOUNT = 1.0              # Default $1.00 USD stake
DEFAULT_DURATION = "auto"         # Default auto-matched contract duration
DEFAULT_DURATION_UNIT = "m"       # "m" for minutes duration unit
DEFAULT_CURRENCY = "USD"          # Account currency (default USD)
DEFAULT_TARGET_PROFIT = 10.0      # Stop trading when profit reaches +$10.00
DEFAULT_STOP_LOSS = 10.0          # Stop trading when loss reaches -$10.00
DEFAULT_STRATEGY = "technical_indicators"    # Default strategy

# Default ML Model Configuration
DEFAULT_ML_MODEL_PATH = "models/eurusd_sequence_model.pkl" if os.path.exists("models/eurusd_sequence_model.pkl") else "models/eurusd_sequence_model.pt"

