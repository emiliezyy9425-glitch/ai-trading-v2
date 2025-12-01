import argparse
import asyncio
import os
import importlib
import math
from pathlib import Path
import time
import multiprocessing as mp
from datetime import date, datetime, time as dtime, timedelta, timezone
from expiry_utils import get_nearest_friday, get_nearest_friday_expiry
import pandas as pd
import logging
import functools
import sys
import pytz
import psutil
import csv
import schedule
import smtplib
from ib_insync import (
    IB,
    Stock,
    util,
    Option,
    MarketOrder,
    LimitOrder,
    RequestError,
    Contract,
)
import json
import numpy as np
import subprocess
import random  # Assuming used in placeholders like check_max_drawdown
from threading import Lock
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from collections.abc import Mapping
from joblib import load as joblib_load
import torch
import torch.nn as nn
from collections import defaultdict, deque
import math
import traceback  # Added for full stack traces
import re
from training_lookback import get_training_lookback_duration_string
from option_chain_skip import should_skip_option_chain

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger().handlers[0].setFormatter(
    logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
)


def retry(
    exceptions: Exception | tuple[type[BaseException], ...] = Exception,
    tries: int = 3,
    delay: float = 1,
    backoff: float = 2,
    max_delay: float | None = None,
    jitter: float = 0,
    logger: logging.Logger | None = None,
):
    """Lightweight retry decorator to avoid external dependency failures.

    This mirrors the common semantics of the ``retry`` package used in
    production: retry on ``exceptions`` for ``tries`` attempts with exponential
    ``backoff`` and optional ``jitter``/``max_delay`` constraints. The final
    attempt will re-raise the caught exception.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            remaining = max(1, int(tries))
            wait = max(0.0, float(delay))

            for attempt in range(remaining):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # type: ignore[misc]
                    if attempt >= remaining - 1:
                        raise

                    jitter_delta = random.uniform(0, float(jitter)) if jitter else 0.0
                    sleep_for = wait + jitter_delta

                    if logger:
                        logger.warning(
                            "Retrying %s in %.1fs (%d/%d) after %s",
                            func.__name__,
                            sleep_for,
                            attempt + 1,
                            remaining,
                            exc,
                        )

                    time.sleep(sleep_for)
                    wait *= float(backoff) if backoff else 1.0
                    if max_delay is not None:
                        wait = min(wait, float(max_delay))

        return wrapper

    return decorator

env_path = Path(__file__).resolve().with_name(".env")
dotenv_spec = importlib.util.find_spec("dotenv")

if env_path.exists():
    if dotenv_spec is not None:
        dotenv = importlib.import_module("dotenv")
        if hasattr(dotenv, "load_dotenv"):
            dotenv.load_dotenv(env_path)
            print("Loaded .env from source directory (local/dev mode)")
    else:
        print("python-dotenv not available – continuing with system env vars")
else:
    print(".env file not found in image – using container environment variables (production mode)")

# Suppress TensorFlow warnings if needed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
MAX_WORKERS = 36
REQUEST_LOCK = Lock()


def ensure_event_loop() -> asyncio.AbstractEventLoop:
    """Return a thread-local event loop, creating one when absent.

    ``ib_insync``'s synchronous helpers (for example ``IB.qualifyContracts``)
    internally rely on ``asyncio`` event loops. Threads spawned by
    ``ThreadPoolExecutor`` do not create a loop automatically, which leads to
    ``RuntimeError: There is no current event loop`` when IBKR calls occur in
    worker threads. Initializing a loop on demand keeps those calls working
    without changing the existing synchronous code paths.
    """

    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
# Default set of U.S. market holidays. Extend this with concrete ``date``
# objects (Eastern calendar) when holiday-aware scheduling is required.
US_MARKET_HOLIDAYS: set[date] = set()
US_EASTERN = pytz.timezone("America/New_York")
# Assuming these are defined in separate files; placeholders for imports
import ml_predictor
from ml_predictor import predict_with_all_models

# These are required by the main script — ml_predictor now defines them again
MODEL_NAMES = ("LSTM", "Transformer")
MODEL_DECISION_COLUMNS = tuple(f"{name}_decision" for name in MODEL_NAMES)
from indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_td_sequential,
    calculate_tds_trend,
    detect_pivots,
    calculate_fib_levels_from_pivots,
    calculate_fib_time_zones,
    calculate_pivot_points,
    calculate_zig_zag,
    calculate_high_volatility,
    calculate_volume_spike,
    calculate_volume_weighted_category,
    calculate_atr,
    calculate_adx,
    calculate_obv,
    calculate_stochastic_oscillator,
    calculate_level_weight,
    fast_candlestick_patterns,
    get_candlestick_patterns,
    summarize_td_sequential,
    DEFAULT_CANDLESTICK_PATTERN_NAMES,
    DEFAULT_CANDLESTICK_PATTERN_FEATURES,
    DEFAULT_CANDLESTICK_PATTERN_COLUMNS,
    DEFAULT_CANDLESTICK_PATTERN_CODES,
    add_legacy_candlestick_columns,
)
from self_learn import (
    FEATURE_NAMES as TRAINING_FEATURE_NAMES,
    CANDLESTICK_FEATURES,
    _BASE_FEATURES,
)
from feature_engineering import (
    count_fib_timezones,
    default_feature_values,
    derive_fibonacci_features,
    encode_td9,
    encode_vol_cat,
    encode_zig,
    add_golden_price_features,
)
from sp500_breadth import calculate_s5tw_ibkr
import atexit
from scripts.close_options import close_all_option_positions
from scripts.download_historical_prices import fetch_iv_delta_spx
from scripts.generate_historical_data import (
    append_historical_data,
    generate_historical_data,
    _augment_timeframe_features as _historical_augment_timeframe_features,
    _finalise_feature_frame as _historical_finalise_feature_frame,
)
from tickers_cache import TICKERS_FILE_PATH, get_cached_tickers, load_tickers
# ``get_round_lot_size`` was added recently to ``ibkr_utils``. Some deployed
# environments (particularly Docker images built before the function landed)
# still ship an older version of the helper module which lacks the symbol. To
# keep this script backward compatible we attempt to import the helper and
# provide an inline fallback that mirrors the IBKR 2025 round-lot table when
# it is unavailable.
try:
    from ibkr_utils import format_duration, round_to_min_tick, get_round_lot_size
except ImportError:  # pragma: no cover - exercised only in legacy runtimes
    from ibkr_utils import format_duration, round_to_min_tick

    def get_round_lot_size(price: float) -> int:
        """Return the IBKR round-lot size based on the 2025 MDI rules."""

        if price <= 0:
            return 1
        if price <= 250:
            return 100
        if price <= 1000:
            return 40
        if price <= 10000:
            return 10
        return 1
# IB connection handle used across the module. Initialising it to ``None``
# prevents ``NameError`` during interpreter shutdown when the ``cleanup``
# routine attempts to reference it.
ib: Optional[IB] = None
def cleanup(): # pragma: no cover - invoked at process exit
    if ib and ib.isConnected():
        ib.disconnect()
    if os.path.exists(pid_file):
        os.remove(pid_file)
    # ``pytest`` and other runners may shut down logging before this ``atexit``
    # hook fires, leaving handlers with closed streams. Attempting to log in
    # that state raises ``ValueError: I/O operation on closed file`` during test
    # teardown. Remove any closed handlers and only log if a handler remains.
    try:
        for h in list(logger.handlers):
            stream = getattr(h, "stream", None)
            if stream is not None and getattr(stream, "closed", False):
                logger.removeHandler(h)
        if logger.handlers:
            logger.info("✅ Cleaned up resources on exit.")
        else: # Fall back to stdout to avoid raising during interpreter shutdown
            print("✅ Cleaned up resources on exit.")
    except Exception:
        # As a last resort, ensure the message is emitted without propagating
        # unexpected shutdown-time errors.
        try:
            print("✅ Cleaned up resources on exit.")
        except Exception:
            pass
# Add outside the loop
reconnect_attempts = 0
max_backoff = 600 # 10 minutes max
atexit.register(cleanup)
# Determine project root dynamically so tests and runtime environments that do
# not mount the code at ``/app`` still function correctly.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
pid_file = os.path.join(PROJECT_ROOT, "ai_agent.pid")


def _last_completed_bar_timestamp(
    timeframe: str = "1 hour", reference: Optional[datetime] = None
) -> datetime:
    ref = (reference or datetime.now(timezone.utc)).astimezone(timezone.utc)
    ref = ref.replace(second=0, microsecond=0)
    tf = timeframe.strip().lower()
    if tf in {"1 day", "1d", "daily"}:
        return (ref - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    if tf in {"4 hours", "4h"}:
        current_block = ref.replace(
            hour=(ref.hour // 4) * 4, minute=0, second=0, microsecond=0
        )
        return current_block - timedelta(hours=4)
    # Default to hourly-style alignment (1h, 60m, etc.).
    current_hour = ref.replace(minute=0, second=0, microsecond=0)
    return current_hour - timedelta(hours=1)
def _resolve_contract_details(ticker: str) -> tuple[str, str, str, str]:
    """Return the stock routing tuple (symbol, exchange, currency, primary)."""

    ticker = ticker.upper().strip()

    # Known NASDAQ names that require ``primaryExchange`` for SMART routing.
    nasdaq_stocks = {
        "TSLA",
        "TSLL",
        "NVDA",
        "AMD",
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "META",
        "NFLX",
        "COIN",
        "MSTR",
        "HOOD",
        "SOFI",
        "RIVN",
        "LCID",
    }

    symbol = ticker
    exchange = "SMART"
    currency = "USD"
    primary = "NASDAQ" if ticker in nasdaq_stocks else ""

    return symbol, exchange, currency, primary


def qualify_stock(ib: IB, symbol: str) -> Stock:
    """Create and qualify a SMART-routed stock contract for ``symbol``."""

    sym, exchange, currency, primary = _resolve_contract_details(symbol)
    contract = Stock(sym, exchange, currency)
    if primary:
        contract.primaryExchange = primary

    qualified = ib.qualifyContracts(contract)
    if not qualified:
        raise ValueError(f"Could not qualify contract for {symbol}")
    return qualified[0]
def get_option_multiplier(ticker: str, exchange: str) -> str:
    """Return the IBKR contract multiplier for an option.
    The multiplier varies by exchange. U.S. options generally use ``100`` while
    Hong Kong contracts commonly settle for ``500`` shares – though some, like
    BABA (9988), use ``500``. For unknown combinations a safe default of
    ``100`` is used and a warning is logged so the mapping can be extended.
    """
    if exchange in {"SMART", "CBOE", "ISE"}: # Most U.S. exchanges
        return "100"
    if exchange == "SEHK": # Hong Kong Exchange
        hk_multipliers = {
            "3690": "500", # Meituan
            "700": "100", # Tencent
            "9988": "500", # Alibaba
            "388": "100", # HKEX
            "2097": "500", # MIXUE GROUP (assumed; confirm if options exist)
            "168": "2000", # assumed default
            "1398": "1000", # assumed default
            "6682": "100", # assumed default
            "941": "500", # assumed default
            "762": "2000", # assumed default
            "728": "200", # assumed default
            "688351": "500", # assumed default
            "1918": "2000", # assumed default
            "2202": "2000", # assumed default
            "119": "3000", # assumed default
            "300": "100", # assumed default
            "9698": "200", # assumed default
        }
        multiplier = hk_multipliers.get(ticker, "500")
        if ticker not in hk_multipliers:
            logger.warning(
                f"⚠️ Unknown multiplier for SEHK/{ticker}; defaulting to '500'"
            )
        return multiplier
    logger.warning(
        f"⚠️ Unknown multiplier for {exchange}/{ticker}; defaulting to '100'"
    )
    return "100"
def _stock_contract(ib: IB, ticker: str) -> Stock:
    """Return a fully qualified stock contract for ``ticker``."""

    return qualify_stock(ib, ticker)
def _option_contract(
    ticker: str, expiry: str, strike: float, right: str
) -> Option:
    symbol, exchange, currency, _ = _resolve_contract_details(ticker)
    multiplier = get_option_multiplier(ticker, exchange)
    return Option(
        symbol,
        expiry,
        strike,
        right,
        exchange,
        multiplier=multiplier,
        currency=currency,
    )
# ===========================
# Data Lake: Persistence Utils
# ===========================
try:
    import pyarrow # noqa: F401
    _HAS_ARROW = True
except Exception:
    _HAS_ARROW = False
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)
def _to_utc_index(
    df: pd.DataFrame, date_col_hint: Optional[str] = None
) -> pd.DataFrame:
    d = df.copy()
    idx = None
    if date_col_hint and date_col_hint in d.columns:
        idx = pd.to_datetime(d[date_col_hint], errors="coerce", utc=True)
    else:
        for cand in ("timestamp", "date", "datetime", "time"):
            if cand in d.columns:
                idx = pd.to_datetime(d[cand], errors="coerce", utc=True)
                break
    if idx is None:
        idx = pd.to_datetime(d.index, errors="coerce", utc=True)
    mask = idx.notna()
    d = d.loc[mask].copy()
    d.index = idx[mask]
    d.index.name = "timestamp"
    ordered = [c for c in ["open", "high", "low", "close", "volume"] if c in d.columns]
    others = [c for c in d.columns if c not in ordered]
    d = d[ordered + others]
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d
def _merge_frames(existing: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new
    all_cols = sorted(set(existing.columns) | set(new.columns))
    existing = existing.reindex(columns=all_cols)
    new = new.reindex(columns=all_cols)
    out = pd.concat([existing, new], axis=0)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out
def _write_parquet_or_csv(df: pd.DataFrame, path_no_ext: str) -> str:
    try:
        if _HAS_ARROW:
            out_path = path_no_ext + ".parquet"
            df.reset_index().to_parquet(out_path, index=False, engine="pyarrow")
        else:
            out_path = path_no_ext + ".csv"
            df.to_csv(
                out_path,
                index=True,
                index_label="timestamp",
                date_format="%Y-%m-%dT%H:%M:%SZ",
            )
        return out_path
    except Exception as e:
        logger.error(f"❌ Failed to write data to {path_no_ext}: {e}")
        raise
def _read_parquet_or_csv(path_no_ext: str) -> Optional[pd.DataFrame]:
    pqt = path_no_ext + ".parquet"
    csv_path = path_no_ext + ".csv"
    if os.path.exists(pqt) and _HAS_ARROW:
        df = pd.read_parquet(pqt)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.set_index("timestamp")
        return df
    elif os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    return None
def save_raw_bars(
    df: pd.DataFrame,
    ticker: str,
    timeframe: str,
    source: str,
    date_col_hint: Optional[str] = None,
) -> Optional[str]:
    root = os.path.join(
        PROJECT_ROOT, "data", "lake", "raw", ticker, timeframe.replace(" ", "_")
    )
    _ensure_dir(root)
    d = _to_utc_index(df, date_col_hint=date_col_hint)
    if d.empty:
        return None
    start = d.index.min().strftime("%Y%m%dT%H%M%SZ")
    end = d.index.max().strftime("%Y%m%dT%H%M%SZ")
    base = os.path.join(root, f"{source}_{start}_{end}")
    path = _write_parquet_or_csv(d, base)
    logger.info(f"📦 Saved raw bars → {path}")
    return path
def save_curated_bars(
    df: pd.DataFrame,
    ticker: str,
    timeframe: str,
    date_col_hint: Optional[str] = None,
) -> str:
    root = os.path.join(PROJECT_ROOT, "data", "lake", "curated")
    _ensure_dir(root)
    base = os.path.join(root, f"{ticker}_{timeframe.replace(' ', '_')}")
    existing = _read_parquet_or_csv(base)
    d = _to_utc_index(df, date_col_hint=date_col_hint)
    merged = _merge_frames(existing, d)
    out_path = _write_parquet_or_csv(merged, base)
    logger.info(f"🧱 Updated curated store → {out_path} (rows={len(merged)})")
    return out_path
def load_curated_bars(
    ticker: str,
    timeframe: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    base = os.path.join(
        PROJECT_ROOT,
        "data",
        "lake",
        "curated",
        f"{ticker}_{timeframe.replace(' ', '_')}",
    )
    df = _read_parquet_or_csv(base)
    if df is None:
        return pd.DataFrame()
    if start:
        df = df[df.index >= pd.to_datetime(start, utc=True)]
    if end:
        df = df[df.index <= pd.to_datetime(end, utc=True)]
    return df


def get_optimal_duration(
    ticker: str,
    timeframe: str,
    existing: Optional[pd.DataFrame] = None,
) -> str:
    """Return an IBKR duration string based on gaps in the curated store."""

    fallback = {
        "1 hour": format_duration(2, "M"),
        "4 hours": format_duration(6, "M"),
        "1 day": format_duration(1, "Y"),
    }.get(timeframe, HISTORICAL_DATA_DURATION)

    from_now = datetime.now(timezone.utc)
    frame = existing if existing is not None else load_curated_bars(ticker, timeframe)

    if frame is None or frame.empty:
        return fallback

    last_ts = frame.index.max()
    if pd.isna(last_ts):
        return fallback

    last_ts = pd.Timestamp(last_ts)
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize(timezone.utc)
    else:
        last_ts = last_ts.tz_convert(timezone.utc)

    days_missing = max((from_now - last_ts.to_pydatetime()).days + 30, 0)

    if days_missing < 2:
        return format_duration(2, "D")
    if days_missing <= 30:
        return format_duration(1, "M")
    if days_missing <= 180:
        return format_duration(6, "M")
    return fallback
# Setup logging
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
logger = logging.getLogger("ai_agent")
logger.setLevel(logging.INFO)
logger.propagate = False
trade_logger = logging.getLogger("ai_agent.trade_log")
trade_logger.setLevel(logging.INFO)
trade_logger.propagate = False
class _SafeStreamHandler(logging.StreamHandler):
    """Stream handler that gracefully handles unsupported Unicode characters.
    Windows terminals commonly default to legacy code pages (for example CP1252)
    that cannot represent emoji characters used throughout the log messages. In
    those environments the default :class:`logging.StreamHandler` raises a
    ``UnicodeEncodeError`` which interrupts the program. By attempting the
    write and, on failure, re-encoding with ``errors='replace'`` we preserve the
    log output (emoji are replaced with ``?``) without crashing.
    """
    def emit(self, record: logging.LogRecord) -> None: # pragma: no cover - IO
        try:
            msg = self.format(record)
            stream = self.stream
            if stream is None:
                return
            terminator = self.terminator
            encoding = getattr(stream, "encoding", None)
            if encoding:
                try:
                    stream.write(msg + terminator)
                except UnicodeEncodeError:
                    safe_msg = msg.encode(encoding, errors="replace").decode(encoding)
                    stream.write(safe_msg + terminator)
            else:
                stream.write(msg + terminator)
            self.flush()
        except Exception:
            self.handleError(record)
if not logger.handlers:
    file_handler = logging.FileHandler(
        os.path.join(PROJECT_ROOT, "logs", "ai_agent.log"), encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    trade_file_handler = logging.FileHandler(
        os.path.join(PROJECT_ROOT, "logs", "ai_agent_trade.log"),
        encoding="utf-8",
    )
    trade_file_handler.setLevel(logging.INFO)
    trade_file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(message)s")
    )
    stream_handler = _SafeStreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)
    trade_logger.addHandler(trade_file_handler)
    logger.addHandler(stream_handler)


def log_model_decision(
    *,
    ticker: str,
    decision: str,
    detail: Mapping[str, object] | None = None,
) -> None:
    """Record the model's decision and confidences to the dedicated trade log."""

    detail_map = detail if isinstance(detail, Mapping) else {}
    detail_confidences = detail_map.get("confidences", {}) if isinstance(
        detail_map, Mapping
    ) else {}
    trade_logger.info(
        "decision|ticker=%s|decision=%s|confidence=%.6f|trigger=%s|confidences=LSTM=%.6f,Transformer=%.6f",
        ticker,
        decision,
        float(detail_map.get("confidence", 0.0)) if isinstance(detail_map, Mapping) else 0.0,
        detail_map.get("trigger", "unknown") if isinstance(detail_map, Mapping) else "unknown",
        detail_confidences.get("LSTM", 0.0)
        if isinstance(detail_confidences, Mapping)
        else 0.0,
        detail_confidences.get("Transformer", 0.0)
        if isinstance(detail_confidences, Mapping)
        else 0.0,
    )


def log_trade_execution(
    *,
    ticker: str,
    action: str,
    quantity: int,
    price: float,
    source: str,
) -> None:
    """Record concise trade execution details to the dedicated trade log."""

    trade_logger.info(
        (
            "execution|ticker=%s|action=%s|quantity=%s|price=%.3f|source=%s"
        ),
        ticker,
        action,
        quantity,
        price,
        source,
    )

_FEATURE_ORDER_ARTIFACT = os.path.join(PROJECT_ROOT, "models", "feature_order.joblib")
_SCALER_ARTIFACT = os.path.join(PROJECT_ROOT, "models", "scaler.joblib")

_S5TW_CACHE_SECONDS = 3600
_s5tw_cache_value: Optional[float] = None
_s5tw_cache_time: Optional[datetime] = None
last_processed_bar: dict[str, datetime] = {}


def get_current_s5tw() -> float:
    """Fetch (and cache) the latest S5TW breadth reading."""

    global _s5tw_cache_time, _s5tw_cache_value

    now = datetime.now()
    if _s5tw_cache_time and (now - _s5tw_cache_time).total_seconds() < _S5TW_CACHE_SECONDS:
        return _s5tw_cache_value if _s5tw_cache_value is not None else 50.0

    if ib is None or not ib.isConnected():
        logger.warning(
            "S5TW breadth unavailable because IBKR is disconnected; using neutral 50%%"
        )
        value = 50.0
    else:
        try:
            value = float(ib.run(calculate_s5tw_ibkr(ib)))
        except Exception as exc:
            logger.error("S5TW failed, using neutral 50%%: %s", exc)
            value = 50.0

    _s5tw_cache_value = value
    _s5tw_cache_time = now
    logger.info("Using live S5TW breadth: %.2f%%", value)
    return value


def _load_training_feature_artifacts() -> tuple[list[str], Optional[object]]:
    """Load the persisted feature order and scaler produced during training.

    The helper degrades gracefully when the artifacts are missing by falling
    back to the canonical schema defined in ``self_learn.FEATURE_NAMES``.
    """

    feature_order: list[str] = list(TRAINING_FEATURE_NAMES)
    scaler: Optional[object] = None

    if os.path.exists(_FEATURE_ORDER_ARTIFACT):
        try:
            loaded = joblib_load(_FEATURE_ORDER_ARTIFACT)
            if isinstance(loaded, (list, tuple, pd.Index, np.ndarray)):
                feature_order = [str(col) for col in loaded]
            else:
                logger.warning(
                    "⚠️ feature_order.joblib did not contain a sequence. Using built-in order."
                )
            logger.info("📐 Loaded training feature order (%d columns).", len(feature_order))
        except Exception as exc:
            logger.warning(
                "⚠️ Failed to load %s: %s. Falling back to default feature order.",
                _FEATURE_ORDER_ARTIFACT,
                exc,
            )
    else:
        logger.info(
            "ℹ️ No training feature-order artifact found at %s; using default schema.",
            _FEATURE_ORDER_ARTIFACT,
        )

    if os.path.exists(_SCALER_ARTIFACT):
        try:
            scaler = joblib_load(_SCALER_ARTIFACT)
            if not hasattr(scaler, "transform"):
                logger.warning(
                    "⚠️ Loaded scaler artifact from %s lacks a transform() method; ignoring.",
                    _SCALER_ARTIFACT,
                )
                scaler = None
            else:
                logger.info("📏 Loaded training scaler from %s.", _SCALER_ARTIFACT)
        except Exception as exc:
            logger.warning(
                "⚠️ Failed to load training scaler %s: %s. Proceeding without scaling.",
                _SCALER_ARTIFACT,
                exc,
            )
            scaler = None
    else:
        logger.info(
            "ℹ️ No scaler artifact found at %s; predictions will use raw features.",
            _SCALER_ARTIFACT,
        )

    return feature_order, scaler


FEATURE_ORDER, TRAINING_SCALER = _load_training_feature_artifacts()

# Check for existing instances
# if os.path.exists(pid_file):
# with open(pid_file, 'r') as f:
# pid = int(f.read().strip())
# if psutil.pid_exists(pid):
# logger.error(f"❌ Another instance of AI Agent is running with PID {pid}. Exiting.")
# sys.exit(1)
# with open(pid_file, 'w') as f:
# f.write(str(os.getpid()))
# ========== CONFIG ==========
TRADE_LOG_PATH = os.path.join(PROJECT_ROOT, "data", "trade_log.csv")
LAST_TRADE_RESULTS_PATH = os.path.join(PROJECT_ROOT, "data", "last_trade_results.json")
MODEL_DECISION_LOG_PATH = os.path.join(PROJECT_ROOT, "data", "model_decision_log.csv")
# Columns stored in the trade log CSV. Extended to cover all fields used in
# tests and reporting utilities.
CSV_COLUMNS = [
    "timestamp",
    "ticker",
    "price",
    "decision",
    "fib",
    "tds",
    "td9",
    "RSI",
    "MACD",
    "Signal",
    "Volume",
    "IV",
    "Delta",
    "FibTimeZones",
    "ZigZagTrend",
    "HighVol",
    "VolSpike",
    "VolCategory",
    "Source",
    "ACCOUNT",
    "ai_decision",
    "ml_decision",
    *MODEL_DECISION_COLUMNS,
]
MODEL_DECISION_LOG_COLUMNS = [
    "timestamp",
    "ticker",
    "price",
    "decision_price",
    "final_decision",
    "reason",
    "trigger_model",
    "trigger_confidence",
]
for name in MODEL_NAMES:
    MODEL_DECISION_LOG_COLUMNS.extend(
        [f"{name}_vote", f"{name}_confidence"]
    )
LIVE_TRADE_SETUPS_PATH = os.path.join(PROJECT_ROOT, "data", "live_trade_setups.csv")
LIVE_TRADE_COLUMNS = [
    "timestamp",
    "ticker",
    "direction",
    "entry_price",
    "position_size",
    "position_value",
    "account_equity",
    "ml_decision",
    "td9_1h",
    "reason",
]
TD9_LONG_VALUES = {9, 10, 11, 12, 13}
TD9_SHORT_VALUES = {-9, -10, -11, -12, -13}

TSLQ_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.992,   # very high — only fires on nuclear confidence
    # Bottom 4 disabled (max conf too low)
}


LABU_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.995,   # very high — only fires on extreme confidence
    # Bottom 4 disabled (max conf too low)
}


AAPU_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.988,
    # Bottom 4 disabled (max conf too low)
}


APPX_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.988,
    # Bottom 4 disabled (max conf too low)
}


AVGX_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.988,
    # Bottom 4 disabled (max conf too low)
}


BLSX_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.988,
    # Bottom 4 disabled (max conf too low)
}


ORCX_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.988,
    # Bottom 4 disabled (max conf too low)
}


QSU_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.988,
    # Bottom 4 disabled (max conf too low)
}


AMDL_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.988,
    # Bottom 4 disabled (max conf too low)
}


AMZU_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.988,
    # Bottom 4 disabled (max conf too low)
}


CRWL_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.988,
    # Bottom 4 disabled (max conf too low)
}


CRCG_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.992,   # very high — only fires on extreme confidence
    # Bottom 4 disabled (max conf too low)
}


ARCX_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.990,   # very high — only fires on extreme confidence
    # Bottom 4 disabled (max conf too low)
}


SPXL_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.988,
    # Bottom 4 disabled (max conf too low)
}


SOXL_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.988,
    # Bottom 4 disabled (max conf too low)
}


ASMG_THRESHOLDS = {
    "LSTM": 0.98,
    "Transformer": 0.988,
    # Bottom 4 disabled (max conf too low)
}


METU_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.988,
    # Bottom 4 disabled (max conf too low)
}

NVDU_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.988,
    # Bottom 4 disabled (max conf too low)
}

NVTX_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.988,
    # Bottom 4 disabled (max conf too low)
}

PLTU_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.988,
    # Bottom 4 disabled (max conf too low)
}





XPP_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.988,
    # Bottom 4 disabled (max conf too low)
}


YINN_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.994,   # extremely high — only fires on nuclear confidence
    # Bottom 4 disabled (max conf too low)
}


YANG_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.994,   # extremely high — only fires on nuclear confidence
    # Bottom 4 disabled (max conf too low)
}


TNA_THRESHOLDS = {
    "LSTM":        0.98,
    "Transformer": 0.994,   # extremely high — only fires on nuclear confidence
    # Bottom 4 disabled (max conf too low)
}


# Ticker-specific confidence thresholds
TICKER_SETTINGS = {
    "TSLA": {
        "thresholds": {
            "LSTM": 0.985,
            "Transformer": 0.982,
        },
    },
    "CONL": {
        "thresholds": {
            "LSTM": 0.980,
            "Transformer": 0.984,
        },
    },
    "TSLL": {
        "thresholds": {
            "LSTM": 0.985,
            "Transformer": 0.982,
        },
    },
    "TSLQ": {
        "thresholds": TSLQ_THRESHOLDS,
    },
    "MSTX": {
        "thresholds": {
            "LSTM": 0.98,
            "Transformer": 0.97,
        },
    },
    "TQQQ": {
        "thresholds": {
            "LSTM": 0.984,
            "Transformer": 0.981,
        },
    },
    "UVIX": {
        "thresholds": {
            "LSTM": 0.980,
            "Transformer": 0.977,
        },
    },
    "SMCL": {  # high-vol profile mirrors UVIX
        "thresholds": {
            "LSTM": 0.98,
            "Transformer": 0.97,
        },
    },
    "SVXY": {
        "thresholds": {
            "LSTM": 0.98,
            "Transformer": 0.988,
            # Bottom 4 disabled (max conf too low)
        },
    },
    "AAPU": {
        "thresholds": AAPU_THRESHOLDS,
    },
    "APPX": {
        "thresholds": APPX_THRESHOLDS,
    },
    "AVGX": {
        "thresholds": AVGX_THRESHOLDS,
    },
    "BLSX": {
        "thresholds": BLSX_THRESHOLDS,
    },
    "QSU": {
        "thresholds": QSU_THRESHOLDS,
    },
    "AMDL": {
        "thresholds": AMDL_THRESHOLDS,
    },
    "AMZU": {
        "thresholds": AMZU_THRESHOLDS,
    },
    "ASMG": {
        "thresholds": ASMG_THRESHOLDS,
    },
    "ORCX": {
        "thresholds": ORCX_THRESHOLDS,
    },
    "CRWL": {
        "thresholds": CRWL_THRESHOLDS,
    },
    "CRCG": {
        "thresholds": CRCG_THRESHOLDS,
    },
    "ARCX": {
        "thresholds": ARCX_THRESHOLDS,
    },
    "METU": {
        "thresholds": METU_THRESHOLDS,
    },
    "NVDU": {
        "thresholds": NVDU_THRESHOLDS,
    },
    "NVTX": {
        "thresholds": NVTX_THRESHOLDS,
    },
    "MSOX": {
        "thresholds": {
            "LSTM": 0.98,
            "Transformer": 0.97,
        },
    },
    "PLTU": {
        "thresholds": PLTU_THRESHOLDS,
    },
    "SOXL": {
        "thresholds": {
            "LSTM": SOXL_THRESHOLDS["LSTM"],
            "Transformer": SOXL_THRESHOLDS["Transformer"],
        },
    },
    "RIOX": {
        "thresholds": {
            "LSTM": 0.98,
            "Transformer": 0.987,
        },
    },
    "FNGU": {
        "thresholds": {
            "LSTM": 0.98,
            "Transformer": 0.988,
            # Bottom 4 disabled (max conf too low)
        },
    },
    "LABU": {
        "thresholds": LABU_THRESHOLDS,
    },
    "SPXL": {
        "thresholds": {
            "LSTM": SPXL_THRESHOLDS["LSTM"],
            "Transformer": SPXL_THRESHOLDS["Transformer"],
        },
    },
    "XPP": {
        "thresholds": XPP_THRESHOLDS,
    },
    "TNA": {
        "thresholds": TNA_THRESHOLDS,
    },
    "YANG": {
        "thresholds": YANG_THRESHOLDS,
    },
    "YINN": {
        "thresholds": YINN_THRESHOLDS,
    },
}
def _normalize_model_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return "".join(str(name).split()).upper()


def _get_ticker_settings(ticker: Optional[str]) -> dict:
    """Return per-ticker model thresholds, defaulting to TSLA settings."""

    fallback = TICKER_SETTINGS.get("TSLA", {})
    if not ticker:
        return fallback
    return TICKER_SETTINGS.get(str(ticker).upper(), fallback)


def _ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
def write_trade_csv(row: dict):
    """row must contain all CSV_COLUMNS keys; missing keys will be filled with ''. Appends a row and writes header if file is new/empty."""
    _ensure_parent_dir(TRADE_LOG_PATH)
    safe_row = {k: row.get(k, "") for k in CSV_COLUMNS}
    write_header = (
        not os.path.exists(TRADE_LOG_PATH) or os.path.getsize(TRADE_LOG_PATH) == 0
    )
    with open(TRADE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(safe_row)


def write_model_decision_log(row: dict) -> None:
    """Append a normalized row to the live model decision log."""

    _ensure_parent_dir(MODEL_DECISION_LOG_PATH)
    safe_row = {k: row.get(k, "") for k in MODEL_DECISION_LOG_COLUMNS}
    write_header = (
        not os.path.exists(MODEL_DECISION_LOG_PATH)
        or os.path.getsize(MODEL_DECISION_LOG_PATH) == 0
    )
    with open(MODEL_DECISION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MODEL_DECISION_LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(safe_row)


def write_live_trade_setup(row: dict) -> None:
    """Append live trade setups used for discretionary execution."""

    _ensure_parent_dir(LIVE_TRADE_SETUPS_PATH)
    safe_row = {col: row.get(col, "") for col in LIVE_TRADE_COLUMNS}
    write_header = (
        not os.path.exists(LIVE_TRADE_SETUPS_PATH)
        or os.path.getsize(LIVE_TRADE_SETUPS_PATH) == 0
    )
    with open(LIVE_TRADE_SETUPS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LIVE_TRADE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(safe_row)


def _format_model_decisions(detail: Mapping[str, object] | None) -> Dict[str, str]:
    """Normalize individual model votes into trade-log columns."""

    votes: Mapping[str, str] = {}
    if isinstance(detail, Mapping):
        raw_votes = detail.get("votes", {})
        if isinstance(raw_votes, Mapping):
            votes = raw_votes

    formatted: Dict[str, str] = {}
    for name in MODEL_NAMES:
        column = f"{name}_decision"
        vote = votes.get(name)
        formatted[column] = str(vote).upper() if vote else "MISSING"
    return formatted
SELF_REVIEW_OUTPUT = os.path.join(PROJECT_ROOT, "logs", "self_review_summary.txt")
TRADE_REPORT_PATH = os.path.join(PROJECT_ROOT, "logs", "trade_report.txt")
REPORT_TRIGGER_PATH = os.path.join(PROJECT_ROOT, "data", "generate_report.txt")
HISTORICAL_DATA_DURATION = get_training_lookback_duration_string()
HISTORICAL_DATA_UPDATE_INTERVAL = 7 * 24 * 3600
HISTORICAL_DATA_FILE = os.path.join(PROJECT_ROOT, "data", "historical_data.csv")
IBKR_FEATURE_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "ibkr_feature_cache")
DATA_COLLECTION_INTERVAL = 3600
REQUEST_TIMESTAMPS: deque[float] = deque()
REQUEST_LIMIT = 60
REQUEST_WINDOW = 600
# Enforce a gentle drip-rate to avoid long sleeps when the window fills.
REQUEST_MIN_SPACING = REQUEST_WINDOW / REQUEST_LIMIT


def _throttle_request(reason: str | None = None) -> None:
    """Gate outbound IBKR calls to stay within ``REQUEST_LIMIT`` over ``REQUEST_WINDOW`` seconds."""

    with REQUEST_LOCK:
        now = time.monotonic()
        while REQUEST_TIMESTAMPS and now - REQUEST_TIMESTAMPS[0] >= REQUEST_WINDOW:
            REQUEST_TIMESTAMPS.popleft()

        # If we're issuing requests faster than the steady-state allowance, pause briefly
        # to smooth bursts before the limit is reached. This avoids hitting a long sleep
        # later in the window while still honoring the hard cap below.
        if REQUEST_TIMESTAMPS:
            next_allowed = REQUEST_TIMESTAMPS[-1] + REQUEST_MIN_SPACING
            if now < next_allowed:
                time.sleep(next_allowed - now)
                now = time.monotonic()
                while REQUEST_TIMESTAMPS and now - REQUEST_TIMESTAMPS[0] >= REQUEST_WINDOW:
                    REQUEST_TIMESTAMPS.popleft()

        if len(REQUEST_TIMESTAMPS) >= REQUEST_LIMIT:
            sleep_for = REQUEST_WINDOW - (now - REQUEST_TIMESTAMPS[0])
            context = f" ({reason})" if reason else ""
            logger.warning(
                f"⚠️ Request window saturated ({len(REQUEST_TIMESTAMPS)}/{REQUEST_LIMIT}). "
                f"Sleeping {sleep_for:.1f}s to manage rate limits{context}."
            )
            time.sleep(max(sleep_for, 0))
            now = time.monotonic()
            while REQUEST_TIMESTAMPS and now - REQUEST_TIMESTAMPS[0] >= REQUEST_WINDOW:
                REQUEST_TIMESTAMPS.popleft()

        REQUEST_TIMESTAMPS.append(time.monotonic())
FEATURE_COUNT = 75 # Updated to include S&P 500 breadth feature
IBKR_ACCOUNT = os.getenv("IBKR_ACCOUNT", "DU123456")
DRY_RUN = os.getenv("DRY_RUN", "False").lower() == "true"
# Load configurations from .env
ORDER_SIZE = int(os.getenv("ORDER_SIZE", "1"))
MAX_DAILY_PUT_TRADES = int(os.getenv("MAX_DAILY_PUT_TRADES", "15"))
MAX_DAILY_CALL_TRADES = int(os.getenv("MAX_DAILY_CALL_TRADES", "15"))
DEFAULT_EQUITY_FRACTION = 0.01  # Target 1% of account equity per trade
TD9_RULE_BUY_VALUES = {9, 10, 11, 12, 13}
TD9_RULE_SELL_VALUES = {-9, -10, -11, -12, -13}
TD9_RULE_EQUITY_FRACTION = 0.01  # Allocate 1% of equity to TD9 rule trades
TD9_RULE_MAX_EQUITY_FRACTION = 0.10  # Cap TD9 positions at 10% of equity
TD9_RULE_SOURCE = "TD9_RULE"
# Validate .env configurations
required_env_vars = ["ORDER_SIZE", "MAX_DAILY_PUT_TRADES", "MAX_DAILY_CALL_TRADES"]
for var in required_env_vars:
    if not os.getenv(var):
        logger.warning(f"⚠️ Environment variable {var} not set. Using default value.")

# Track open trades and the most recent realized outcome per ticker/strategy so
# that the sizing logic can apply TD9-specific pyramiding rules while leaving
# ML-driven trades unchanged.
_OPEN_TRADES: Dict[str, Dict[str, Any]] = {}
_LAST_TRADE_RESULTS: Dict[str, Dict[str, Dict[str, float]]] = {}
_GLOBAL_TRADE_RESULT_KEY = "__GLOBAL__"


def _load_last_trade_results(path: str = LAST_TRADE_RESULTS_PATH) -> None:
    """Hydrate in-memory last-trade results from a JSON cache if present."""

    global _LAST_TRADE_RESULTS
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("⚠️ Failed to load last trade cache from %s: %s", path, exc)
        return

    if not isinstance(cached, Mapping):
        logger.warning("⚠️ Ignoring malformed last trade cache at %s", path)
        return

    hydrated: Dict[str, Dict[str, Dict[str, float]]] = {}
    for ticker_key, strat_map in cached.items():
        if not isinstance(strat_map, Mapping):
            continue
        strat_results: Dict[str, Dict[str, float]] = {}
        for strategy, entry in strat_map.items():
            if not isinstance(entry, Mapping):
                continue
            try:
                strat_results[str(strategy)] = {
                    "pnl": float(entry.get("pnl", 0.0)),
                    "size": float(entry.get("size", 0.0)),
                }
            except Exception:
                continue
        if strat_results:
            hydrated[str(ticker_key)] = strat_results
    _LAST_TRADE_RESULTS.update(hydrated)


def _persist_last_trade_results(path: str = LAST_TRADE_RESULTS_PATH) -> None:
    """Persist the last realised PnL for each ticker/strategy to disk."""

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_LAST_TRADE_RESULTS, f, indent=2)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("⚠️ Failed to persist last trade cache to %s: %s", path, exc)


_load_last_trade_results()


def _ticker_key(ticker: str) -> str:
    return ticker.upper()


def evaluate_td9_rule(td9_value: int) -> Optional[str]:
    """Return the stock decision implied by the TD9 1H rule."""

    if td9_value in TD9_RULE_BUY_VALUES:
        return "BUY"
    if td9_value in TD9_RULE_SELL_VALUES:
        return "SELL"
    return None


def _record_open_trade(
    ticker: str, side: str, quantity: int, price: float, strategy: str
) -> None:
    """Persist the aggregate open trade state for ``ticker`` and strategy."""

    if quantity <= 0:
        return
    key = _ticker_key(ticker)
    entry = _OPEN_TRADES.get(key)
    if entry and entry.get("side") == side:
        prev_qty = float(entry.get("size", 0))
        prev_price = float(entry.get("price", 0))
        new_qty = prev_qty + quantity
        if new_qty <= 0:
            return
        avg_price = ((prev_price * prev_qty) + (price * quantity)) / new_qty
        entry.update({"price": avg_price, "size": new_qty, "strategy": strategy})
        _OPEN_TRADES[key] = entry
        return
    _OPEN_TRADES[key] = {
        "side": side,
        "price": float(price),
        "size": float(quantity),
        "strategy": strategy,
    }


def _record_completed_trade(
    ticker: str, closing_side: str, exit_price: float, quantity: int
) -> None:
    """Store the realised PnL for the most recently closed trade."""

    if quantity <= 0:
        return
    key = _ticker_key(ticker)
    entry = _OPEN_TRADES.get(key)
    if not entry or entry.get("side") != closing_side:
        return
    entry_qty = float(entry.get("size", 0))
    if entry_qty <= 0:
        return
    qty_to_close = min(entry_qty, float(quantity))
    entry_price = float(entry.get("price", 0))
    if closing_side == "LONG":
        pnl = (exit_price - entry_price) * qty_to_close
    else:
        pnl = (entry_price - exit_price) * qty_to_close
    remaining = entry_qty - qty_to_close
    if remaining > 0:
        entry["size"] = remaining
        _OPEN_TRADES[key] = entry
    else:
        _OPEN_TRADES.pop(key, None)
    strategy_label = entry.get("strategy") if entry else None
    if not strategy_label:
        strategy_label = _GLOBAL_TRADE_RESULT_KEY
    result_entry = {"pnl": pnl, "size": qty_to_close}
    strategy_bucket = _LAST_TRADE_RESULTS.setdefault(key, {})
    strategy_bucket[strategy_label] = result_entry
    strategy_bucket[_GLOBAL_TRADE_RESULT_KEY] = {
        "pnl": pnl,
        "size": qty_to_close,
        "strategy": strategy_label,
    }
    _persist_last_trade_results()


def get_last_trade_result(
    ticker: str, strategy: Optional[str] = None
) -> Tuple[Optional[float], Optional[int]]:
    """Return the most recent realised PnL and size for ``ticker``/``strategy``."""

    entry = _LAST_TRADE_RESULTS.get(_ticker_key(ticker))
    if not entry:
        return None, None
    key = strategy or _GLOBAL_TRADE_RESULT_KEY
    strat_entry = entry.get(key)
    if strat_entry is None and strategy:
        strat_entry = entry.get(_GLOBAL_TRADE_RESULT_KEY)
    if not strat_entry:
        return None, None
    return float(strat_entry.get("pnl", 0.0)), int(strat_entry.get("size", 0))


def _determine_position_size(
    ticker: str,
    price: float,
    net_liq: float,
    equity_fraction: Optional[float],
    last_pnl: Optional[float],
    last_size: Optional[int],
    *,
    allow_loss_doubling: bool = False,
    max_equity_fraction: Optional[float] = None,
) -> Tuple[int, bool]:
    """Return the share quantity for the next trade and whether it doubled."""

    if price <= 0 or net_liq <= 0:
        return 0, False
    lot_size = get_round_lot_size(price)
    base_fraction = equity_fraction if equity_fraction is not None else DEFAULT_EQUITY_FRACTION
    base_qty = int((net_liq * base_fraction) // price)
    if lot_size > 1:
        base_qty = (base_qty // lot_size) * lot_size
    used_loss_sizing = False
    if allow_loss_doubling:
        if (
            last_pnl is not None
            and last_pnl < 0
            and last_size
            and last_size > 0
        ):
            qty_equity = max(int(last_size) * 2, lot_size)
            used_loss_sizing = True
        else:
            qty_equity = base_qty
    else:
        qty_equity = base_qty
    max_alloc_qty: Optional[int] = None
    if max_equity_fraction is not None and max_equity_fraction > 0:
        max_alloc_qty = int((net_liq * max_equity_fraction) // price)
        if lot_size > 1 and max_alloc_qty > 0:
            max_alloc_qty = (max_alloc_qty // lot_size) * lot_size
        if max_alloc_qty is not None and max_alloc_qty <= 0:
            max_alloc_qty = None
    if max_alloc_qty:
        qty_equity = min(qty_equity, max_alloc_qty)
    affordable = int(net_liq // price)
    if lot_size > 1:
        affordable = (affordable // lot_size) * lot_size
    if affordable <= 0:
        return 0, False
    qty_equity = min(qty_equity, affordable)
    if qty_equity <= 0 and used_loss_sizing:
        used_loss_sizing = False
    return qty_equity, used_loss_sizing


def _calculate_pyramiding_quantity(
    current_size: int, lot_size: int, price: float, net_liq: float
) -> int:
    """Return the incremental shares needed to double ``current_size``.

    The helper respects lot size constraints and available equity so that
    pyramiding never requests more shares than can be afforded or routed.
    """

    if current_size <= 0 or price <= 0 or net_liq <= 0:
        return 0

    target_size = current_size * 2
    if lot_size > 1:
        target_size = max(lot_size, math.ceil(target_size / lot_size) * lot_size)

    qty_equity = target_size - current_size
    if lot_size > 1 and qty_equity > 0:
        qty_equity = max(lot_size, (qty_equity // lot_size) * lot_size)

    affordable = int(net_liq // price)
    if lot_size > 1:
        affordable = (affordable // lot_size) * lot_size

    return max(0, min(qty_equity, affordable))
# Initialize OpenAI and xAI clients
#
# These clients were previously instantiated unconditionally using API keys
# from the environment. In environments where those variables are not set the
# ``OpenAI`` constructor raises an exception which prevents this module from
# being imported. The unit tests patch these client instances, so here we
# construct a lightweight fallback client that mimics the attribute structure
logger.info("Script initialization completed, proceeding to main loop...")
# ========== IBKR Connection ==========
def connect_ibkr(
    max_retries: int = 3,
    initial_client_id: Optional[int] = None,
    delay: int = 5,
) -> Optional[IB]:
    ibkr_port = int(os.getenv("IBKR_PORT", "7496"))
    host_env = os.getenv("IBKR_HOST", "")
    hosts: list[str] = []
    if host_env:
        hosts.extend(
            host.strip()
            for host in host_env.replace(";", ",").split(",")
            if host.strip()
        )
    default_hosts = ["host.docker.internal", "127.0.0.1", "localhost"]
    for default_host in default_hosts:
        if default_host not in hosts:
            hosts.append(default_host)
    base_client_id = int(
        os.getenv(
            "IBKR_CLIENT_ID",
            str(initial_client_id if initial_client_id is not None else 100),
        )
    )
    attempt_counter = {"count": 0}
    def _connect_single(target_host: str) -> IB:
        attempt_counter["count"] += 1
        client_id = base_client_id + attempt_counter["count"] - 1
        ib = IB()
        logger.info(
            f"Attempting connection with clientId={client_id} to {target_host}:{ibkr_port}..."
        )
        try:
            ib.connect(target_host, ibkr_port, clientId=client_id, timeout=30)  # Fixed: use clientId kwarg
        except Exception as e:
            logger.error(f"Connection failed: {e}\n{traceback.format_exc()}")  # Added full stack
            raise
        ib.sleep(2)
        if not ib.isConnected():
            logger.warning(
                f"⚠️ Connection attempt {attempt_counter['count']} failed with clientId={client_id} on {target_host}."
            )
            raise RuntimeError("IBKR connection failed")
        ib.reqCurrentTime()
        logger.info("✅ Connection validated with current time request.")
        logger.info(
            f"✅ Connected to IB Gateway successfully with clientId={client_id} (Account: {IBKR_ACCOUNT})."
        )
        return ib
    connect_with_retry = retry(tries=max_retries, delay=delay)(_connect_single)
    for host in hosts:
        try:
            return connect_with_retry(host)
        except Exception as e:
            logger.error(
                f"❌ Failed to connect to IBKR at {host}:{ibkr_port} after {max_retries} attempts: {e}\n{traceback.format_exc()}"
            )
    logger.error(
        "❌ All IBKR connection attempts failed. Verify IB Gateway/TWS is running and reachable."
    )
    logger.error(
        "Set the IBKR_HOST environment variable to the hostname or IP of your TWS/IB Gateway machine if it is not running on this host."
    )
    return None
# ========== Account Balance Check ==========
@retry(tries=3, delay=5, backoff=2)
def get_cash_balance(ib: IB) -> Optional[float]:
    try:
        account_values = ib.accountValues(account=IBKR_ACCOUNT)
        for item in account_values:
            if item.tag == "CashBalance" and item.currency == "USD":
                cash_balance = float(item.value)
                logger.info(
                    f"✅ Account cash balance: ${cash_balance:.2f} (Account: {IBKR_ACCOUNT})"
                )
                return cash_balance
        logger.warning("⚠️ CashBalance not found in account values.")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to fetch cash balance: {str(e)}\n{traceback.format_exc()}")
        return None
def get_position_size(ib: IB, ticker: str) -> int:
    """Return the share position for ``ticker`` in the connected account.
    If the position cannot be determined, ``0`` is returned so that the
    caller can fall back to a safe default.
    """
    try:
        positions = ib.positions()
        for p in positions:
            contract = getattr(p, "contract", None)
            if (
                contract
                and getattr(contract, "symbol", "") == ticker
                and getattr(contract, "secType", "") == "STK"
            ):
                return int(p.position)
        return 0
    except Exception as e:
        logger.error(f"❌ Failed to fetch position for {ticker}: {e}\n{traceback.format_exc()}")
        return 0
def get_short_call_positions(ib: IB, ticker: str) -> int:
    """Return the number of short CALL contracts for ``ticker``."""
    total = 0
    try:
        for p in ib.positions():
            contract = getattr(p, "contract", None)
            if (
                contract
                and getattr(contract, "symbol", "") == ticker
                and getattr(contract, "secType", "") == "OPT"
                and getattr(contract, "right", "").upper() == "C"
                and getattr(p, "position", 0) < 0
            ):
                total += abs(int(p.position))
    except Exception as e:
        logger.error(f"❌ Failed to fetch CALL option positions for {ticker}: {e}\n{traceback.format_exc()}")
    return total
def get_open_short_call_orders(ib: IB, ticker: str) -> int:
    """Return number of SELL CALL contracts in open orders for ``ticker``."""
    total = 0
    try:
        for trade in getattr(ib, "openTrades", lambda: [])():
            order = getattr(trade, "order", None)
            contract = getattr(trade, "contract", None)
            if (
                getattr(contract, "symbol", "") == ticker
                and getattr(contract, "secType", "") == "OPT"
                and getattr(contract, "right", "").upper() == "C"
                and getattr(order, "action", "").upper() == "SELL"
            ):
                total += int(getattr(order, "totalQuantity", 0))
    except Exception as e:
        logger.error(f"❌ Failed to fetch open CALL orders for {ticker}: {e}\n{traceback.format_exc()}")
    return total
def get_net_liquidity(ib: IB) -> Optional[float]:
    """Return account net liquidity in USD, if available."""
    try:
        account_summary = ib.accountSummary(account=IBKR_ACCOUNT)
        for item in account_summary:
            if item.tag == "NetLiquidation" and item.currency == "USD":
                net_liq = float(item.value)
                logger.info(
                    f"✅ Account net liquidity: ${net_liq:.2f} (Account: {IBKR_ACCOUNT})"
                )
                return net_liq
        logger.warning("⚠️ NetLiquidation not found in account summary.")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to fetch net liquidity: {str(e)}\n{traceback.format_exc()}")
        return None
def get_position_info(ib: IB, ticker: str) -> tuple[int, float]:
    """Return (quantity, average cost) for ``ticker``."""
    try:
        positions = ib.positions()
        for p in positions:
            contract = getattr(p, "contract", None)
            if (
                contract
                and getattr(contract, "symbol", "") == ticker
                and getattr(contract, "secType", "") == "STK"
            ):
                return int(p.position), float(getattr(p, "avgCost", 0.0))
        return 0, 0.0
    except Exception as e:
        logger.error(f"❌ Failed to fetch position info for {ticker}: {e}\n{traceback.format_exc()}")
        return 0, 0.0



# ========== Timeframe & Trade Limit Control ==========
def _iter_trade_log_rows():
    """Yield rows from the trade log as dictionaries."""
    if not os.path.exists(TRADE_LOG_PATH):
        return
    try:
        with open(TRADE_LOG_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return
            for row in reader:
                yield row
    except Exception as exc: # pragma: no cover - defensive logging only
        logger.error(f"Failed to read trade log at {TRADE_LOG_PATH}: {exc}\n{traceback.format_exc()}")
def _row_matches_today(row: dict) -> bool:
    """Return True if ``row`` has a timestamp dated today."""
    timestamp = (row or {}).get("timestamp")
    if not timestamp:
        return False
    try:
        row_datetime = datetime.fromisoformat(str(timestamp))
    except ValueError:
        try:
            row_datetime = datetime.strptime(str(timestamp), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return False
    return row_datetime.date() == datetime.now(timezone.utc).date()
def _row_matches_ticker(row: dict, ticker: str) -> bool:
    return (row or {}).get("ticker", "").upper() == ticker.upper()
def count_today_put_trades(ticker: str) -> int:
    count = 0
    for row in _iter_trade_log_rows() or []:
        if not _row_matches_today(row):
            continue
        if not _row_matches_ticker(row, ticker):
            continue
        decision = (row.get("decision") or "").upper()
        if decision == "PUT":
            count += 1
    return count
def count_today_call_trades(ticker: str) -> int:
    count = 0
    for row in _iter_trade_log_rows() or []:
        if not _row_matches_today(row):
            continue
        if not _row_matches_ticker(row, ticker):
            continue
        decision = (row.get("decision") or "").upper()
        if decision == "CALL":
            count += 1
    return count
def is_hk_market_open() -> bool:
    now_utc = datetime.now(timezone.utc)
    hkt_tz = pytz.timezone("Asia/Hong_Kong")
    now_hkt = now_utc.astimezone(hkt_tz)
    if now_hkt.weekday() >= 5:
        return False
    morning_open = datetime.strptime("09:00", "%H:%M").time()
    morning_close = datetime.strptime("12:00", "%H:%M").time()
    afternoon_open = datetime.strptime("13:00", "%H:%M").time()
    afternoon_close = datetime.strptime("16:00", "%H:%M").time()
    current_time = now_hkt.time()
    return (morning_open <= current_time <= morning_close) or (
        afternoon_open <= current_time <= afternoon_close
    )
def is_euronext_market_open() -> bool:
    now_utc = datetime.now(timezone.utc)
    cet_tz = pytz.timezone("Europe/Paris")
    now_cet = now_utc.astimezone(cet_tz)
    if now_cet.weekday() >= 5:
        return False
    market_open = datetime.strptime("09:00", "%H:%M").time()
    market_close = datetime.strptime("17:30", "%H:%M").time()
    current_time = now_cet.time()
    return market_open <= current_time <= market_close
def is_any_market_open(tickers: Sequence[str]) -> bool:
    for ticker in tickers:
        if is_market_open(ticker):
            return True
    return False


def _us_trading_session(now: Optional[datetime] = None) -> str:
    """Return the current U.S. session for equities.

    Sessions are defined as:
    * ``pre-market``: 04:00–09:30 ET
    * ``regular``: 09:30–16:00 ET
    * ``post-market``: 16:00–20:00 ET
    * ``overnight``: 20:00–04:00 ET (next day)
    * ``closed``: weekends when no U.S. equity trading session is available
    """

    if now is None:
        now = datetime.now(timezone.utc)

    est_tz = pytz.timezone("US/Eastern")
    now_est = now.astimezone(est_tz)
    weekday = now_est.weekday()

    if weekday >= 5:
        return "closed"

    current_time = now_est.time()
    pre_market_start = dtime(4, 0)
    regular_start = dtime(9, 30)
    regular_end = dtime(16, 0)
    post_market_end = dtime(20, 0)

    if pre_market_start <= current_time < regular_start:
        return "pre-market"
    if regular_start <= current_time < regular_end:
        return "regular"
    if regular_end <= current_time < post_market_end:
        return "post-market"
    return "overnight"


def is_market_open(ticker: str = "TSLA", include_extended: bool = True) -> bool:
    now_utc = datetime.now(timezone.utc)
    if ticker.isdigit():
        return is_hk_market_open()
    elif ticker == "MC":
        return is_euronext_market_open()

    session = _us_trading_session(now_utc)
    if session == "closed":
        return False
    if include_extended:
        return True

    return session == "regular"


def is_us_regular_trading_hours(now: Optional[datetime] = None) -> bool:
    """Return ``True`` when the U.S. equity session is open for trading."""

    return _us_trading_session(now) == "regular"


def is_us_overnight_hours(now: Optional[datetime] = None) -> bool:
    """Return ``True`` when the U.S. market is in its overnight session."""

    return _us_trading_session(now) == "overnight"
# NEW: Canonical feature names list (order matches your dict construction)
_BASE_PATTERN_SPECS: list[tuple[str, str, str]] = [
    (display_name, column_name, pattern_code)
    for display_name, column_name, pattern_code in zip(
        DEFAULT_CANDLESTICK_PATTERN_NAMES,
        DEFAULT_CANDLESTICK_PATTERN_COLUMNS,
        DEFAULT_CANDLESTICK_PATTERN_CODES,
    )
]
PATTERN_TIMEFRAME_SPECS: list[tuple[str, str]] = [
    ("1 hour", "1h"),
    ("4 hours", "4h"),
    ("1 day", "1d"),
]
PATTERN_FEATURE_SPECS: list[tuple[str, str, str]] = [
    (
        timeframe,
        display_name,
        base_feature if suffix == "1h" else f"{base_feature}_{suffix}",
    )
    for timeframe, suffix in PATTERN_TIMEFRAME_SPECS
    for display_name, base_feature, _ in _BASE_PATTERN_SPECS
]

# Mirror the canonical training schema to ensure inference uses the same
# feature ordering (including *_change columns) expected by every model.
FEATURE_NAMES = TRAINING_FEATURE_NAMES

TIMEFRAMES: Tuple[str, ...] = ("1 hour", "4 hours", "1 day")
_TIMEFRAME_TO_SUFFIX: Dict[str, str] = {
    "1 hour": "1h",
    "4 hours": "4h",
    "1 day": "1d",
}
_PATTERN_SUFFIX_ORDER: Tuple[str, ...] = tuple(
    _TIMEFRAME_TO_SUFFIX[tf] for tf in TIMEFRAMES
)
_BASE_PATTERN_COLUMNS: Tuple[str, ...] = tuple(
    DEFAULT_CANDLESTICK_PATTERN_COLUMNS
)
EXTENDED_PATTERN_COLUMNS: Tuple[str, ...] = (
    "pattern_hammer",
    "pattern_inverted_hammer",
    "pattern_engulfing",
    "pattern_piercing_line",
    "pattern_morning_star",
    "pattern_three_white_soldiers",
    "pattern_hanging_man",
    "pattern_shooting_star",
    "pattern_evening_star",
    "pattern_three_black_crows",
)
_PATTERN_SUFFIXES: Tuple[str, ...] = ("", "_1h", "_4h", "_1d")
FEATURE_SEQUENCE_WINDOW = 60
_IB_SEQUENCE_LOOKBACK = format_duration(365, "D")
FEATURE_HISTORY: defaultdict[str, deque[pd.Series]] = defaultdict(
    lambda: deque(maxlen=FEATURE_SEQUENCE_WINDOW)
)
RAW_FEATURE_HISTORY: defaultdict[str, deque[pd.Series]] = defaultdict(
    lambda: deque(maxlen=2)
)
_SEEDED_SEQUENCE_TICKERS: set[str] = set()
_HISTORICAL_FEATURE_CHUNK_SIZE = 50_000


def _json_safe_value(value: Any) -> Union[float, int, str, None]:
    """Return a JSON-serializable representation of ``value``.

    ``pandas`` and ``numpy`` routinely yield scalar extension types (for example
    ``np.float64``) that are not directly serialisable by ``json.dumps``. The
    helper normalises those values to their native Python equivalents while
    preserving ``None`` and falling back to ``str(value)`` for unusual objects.
    """

    if isinstance(value, (np.generic,)):
        value = value.item()
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def _sequence_dataframe(rows: Sequence[pd.Series]) -> pd.DataFrame:
    """Return a dataframe containing the most recent sequence window."""

    if not rows:
        return pd.DataFrame(columns=FEATURE_NAMES)

    # ``deque`` already enforces an upper bound but we still slice to make the
    # behaviour obvious if the maxlen is ever increased.
    window_rows = list(rows)[-FEATURE_SEQUENCE_WINDOW:]
    df = pd.DataFrame(window_rows)
    df = df.reindex(columns=FEATURE_NAMES, fill_value=0.0)
    if len(df) < FEATURE_SEQUENCE_WINDOW:
        missing = FEATURE_SEQUENCE_WINDOW - len(df)
        logging.debug(
            "Building ML features with %d/%d rows; padding %d defaults.",
            len(df),
            FEATURE_SEQUENCE_WINDOW,
            missing,
        )
        defaults = pd.Series(default_feature_values(FEATURE_NAMES))
        padding = pd.DataFrame([defaults] * missing)
        padding = padding.reindex(columns=FEATURE_NAMES, fill_value=0.0)
        df = pd.concat([padding, df], ignore_index=True)
    return df


def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """Map common timeframe labels to ``pandas`` timedeltas."""

    normalized = (timeframe or "1 hour").strip().lower()
    if normalized in {"1h", "1 hour", "hourly"}:
        return pd.Timedelta(hours=1)
    if normalized in {"4h", "4 hours"}:
        return pd.Timedelta(hours=4)
    if normalized in {"1d", "1 day", "daily"}:
        return pd.Timedelta(days=1)
    # Fall back to one hour if an unfamiliar label is supplied so live
    # predictions keep flowing instead of crashing.
    return pd.Timedelta(hours=1)


def _ensure_dataframe_for_prediction(
    features: Union[pd.DataFrame, pd.Series, Mapping[str, Any], np.ndarray, Sequence[float]],
    timeframe: str = "1 hour",
) -> pd.DataFrame:
    """Coerce ``features`` into a model-ready dataframe with a UTC index."""

    if features is None:
        return pd.DataFrame(columns=FEATURE_ORDER)

    if isinstance(features, pd.DataFrame):
        df = features.copy()
    elif isinstance(features, pd.Series):
        df = features.to_frame().T
    elif isinstance(features, Mapping):
        df = pd.DataFrame([features])
    else:
        array = np.atleast_2d(features)
        df = pd.DataFrame(array)

    if list(df.columns) != list(FEATURE_ORDER):
        df = df.reindex(columns=FEATURE_ORDER, fill_value=0.0)

    # Ensure a timezone-aware DatetimeIndex is present. Some upstream callers
    # may supply an already-correct index (e.g. during tests); detect and reuse
    # it to avoid clobbering historical timestamps.
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        ordered = df.sort_index()
    else:
        end = _last_completed_bar_timestamp(timeframe)
        step = _timeframe_to_timedelta(timeframe)
        periods = len(df)
        generated_index = pd.date_range(
            end=end,
            periods=max(periods, 1),
            freq=step,
            tz="UTC",
        )
        if periods == 0:
            return df.iloc[0:0].set_index(generated_index[0:0])
        ordered = df.copy()
        ordered.index = generated_index

    return ordered.sort_index()


def apply_live_feature_shift(
    df: pd.DataFrame, metadata_columns: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Replicate the ``X.shift(1)`` performed during training.

    During training every feature column was shifted forward one bar to remove
    look-ahead bias. The live pipeline must perform the identical transform so
    the models evaluate yesterday's features when producing today's signal.
    """

    if df is None or df.empty:
        return df

    protected = set(metadata_columns or ("timestamp", "ticker", "decision"))
    feature_cols = [col for col in df.columns if col not in protected]
    if not feature_cols:
        return df

    shifted = df.copy()
    shifted[feature_cols] = shifted[feature_cols].shift(1)
    shifted = shifted.dropna(subset=feature_cols, how="any")
    return shifted


def enforce_training_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Shift, align, and scale features to mirror the training pipeline."""

    if df is None or df.empty:
        return df

    shifted = apply_live_feature_shift(df)
    if shifted is None or shifted.empty:
        return pd.DataFrame(columns=FEATURE_ORDER)

    aligned = shifted.reindex(columns=FEATURE_ORDER, fill_value=0.0)
    aligned = aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = TRAINING_SCALER
    if scaler is not None:
        try:
            scaled = scaler.transform(aligned.values)
            return pd.DataFrame(scaled, columns=FEATURE_ORDER, index=aligned.index)
        except Exception as exc:
            logger.warning(
                "⚠️ Failed to scale live features with training scaler: %s. Using unscaled values.",
                exc,
            )

    return aligned


def _ibkr_cache_path(ticker: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in ticker)
    return os.path.join(IBKR_FEATURE_CACHE_DIR, f"{safe}.csv")


def _persist_ibkr_seed_cache(ticker: str, dataset: pd.DataFrame) -> None:
    """Write the downloaded IBKR feature frame to disk for future reuse."""

    if dataset.empty:
        return

    cache_path = _ibkr_cache_path(ticker)
    try:
        os.makedirs(IBKR_FEATURE_CACHE_DIR, exist_ok=True)
        dataset.to_csv(cache_path, index=False)
        logger.debug("💾 Cached IBKR feature seed for %s at %s", ticker, cache_path)
    except Exception as exc:  # pragma: no cover - filesystem issues are rare
        logger.warning(
            "⚠️ Failed to persist IBKR seed cache for %s at %s: %s",
            ticker,
            cache_path,
            exc,
        )


def _seed_feature_history_from_cache(ticker: str) -> bool:
    """Load previously cached IBKR features for ``ticker`` into the history deque."""

    cache_path = _ibkr_cache_path(ticker)
    if not os.path.exists(cache_path):
        return False

    try:
        cached = pd.read_csv(cache_path, parse_dates=["timestamp"])
    except Exception as exc:
        logger.error(
            "❌ Failed to read cached IBKR features for %s at %s: %s",
            ticker,
            cache_path,
            exc,
        )
        return False

    if cached.empty:
        logger.warning(
            "⚠️ Cached IBKR feature file %s for %s is empty.", cache_path, ticker
        )
        return False

    cached = cached.sort_values("timestamp")
    history = FEATURE_HISTORY[ticker]
    appended = 0
    for _, row in cached.tail(FEATURE_SEQUENCE_WINDOW).iterrows():
        try:
            history.append(row.loc[FEATURE_NAMES].copy())
        except KeyError as exc:
            logger.error(
                "❌ Cached IBKR data for %s is missing expected columns: %s",
                ticker,
                exc,
            )
            return False
        appended += 1

    if appended:
        logger.info(
            "💾 Seeded %d/%d rows for %s using cached IBKR data at %s.",
            appended,
            FEATURE_SEQUENCE_WINDOW,
            ticker,
            cache_path,
        )
        return True

    return False


def _seed_feature_history_from_ibkr(ticker: str, ib_client: Optional[IB]) -> bool:
    """Download hourly bars from IBKR and derive features for sequence seeding."""

    if ib_client is None:
        logger.warning(
            "⚠️ Cannot fetch IBKR history for %s because no IB client is available.",
            ticker,
        )
        return False
    if not ib_client.isConnected():
        logger.warning(
            "⚠️ IB client disconnected; unable to download historical data for %s.",
            ticker,
        )
        return False

    try:
        contract = _stock_contract(ib_client, ticker)
    except ValueError as exc:  # pragma: no cover - network interaction
        logger.error("❌ Failed to qualify IBKR contract for %s: %s", ticker, exc)
        return False

    end_time = _last_completed_bar_timestamp("1 hour")
    try:
        end_time_str = end_time.strftime("%Y%m%d %H:%M:%S UTC")
        bars = ib_client.reqHistoricalData(
            contract,
            endDateTime=end_time_str,
            durationStr=_IB_SEQUENCE_LOOKBACK,
            barSizeSetting="1 hour",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=2,
            keepUpToDate=False,
        )
        ib_client.sleep(1)
    except Exception as exc:  # pragma: no cover - network interaction
        logger.error("❌ IBKR historical download failed for %s: %s", ticker, exc)
        return False

    if not bars:
        logger.warning("⚠️ IBKR returned no historical bars for %s.", ticker)
        return False

    df = util.df(bars)
    if df.empty or "date" not in df:
        logger.warning("⚠️ Unable to build IBKR dataframe for %s.", ticker)
        return False

    df = df.rename(columns={"date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    price_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in price_cols if col not in df.columns]
    if missing_cols:
        logger.warning(
            "⚠️ IBKR history for %s is missing columns: %s",
            ticker,
            ", ".join(missing_cols),
        )
        return False

    price_df = df[price_cols].sort_index()
    features = _historical_augment_timeframe_features(price_df)
    if features.empty:
        logger.warning(
            "⚠️ Derived indicator frame for %s is empty after IBKR download.", ticker
        )
        return False

    start = price_df.index.min().to_pydatetime()
    end = price_df.index.max().to_pydatetime()
    dataset = _historical_finalise_feature_frame(features, ticker, start, end)
    dataset = add_golden_price_features(dataset)
    if dataset.empty:
        logger.warning(
            "⚠️ Processed IBKR dataset for %s produced no usable feature rows.",
            ticker,
        )
        return False

    dataset = dataset.sort_values("timestamp")
    window = dataset.tail(FEATURE_SEQUENCE_WINDOW)
    if window.empty:
        logger.warning(
            "⚠️ IBKR dataset for %s contained no rows within the sequence window.",
            ticker,
        )
        return False

    history = FEATURE_HISTORY[ticker]
    appended = 0
    for _, row in window.iterrows():
        history.append(row.loc[FEATURE_NAMES].copy())
        appended += 1

    _persist_ibkr_seed_cache(ticker, dataset)

    logger.info(
        "📥 Seeded %d/%d sequence rows for %s using IBKR historical data.",
        appended,
        FEATURE_SEQUENCE_WINDOW,
        ticker,
    )
    return True


def _seed_feature_history_from_historical_data(ticker: str, ib_client: Optional[IB]) -> None:
    """Populate ``FEATURE_HISTORY`` for *ticker* using cached or freshly downloaded data."""

    if ticker in _SEEDED_SEQUENCE_TICKERS:
        return

    feature_history = FEATURE_HISTORY[ticker]
    if feature_history:
        _SEEDED_SEQUENCE_TICKERS.add(ticker)
        return

    seeded = False

    if os.path.exists(HISTORICAL_DATA_FILE):
        feature_window: deque[pd.Series] = deque(maxlen=FEATURE_SEQUENCE_WINDOW)
        try:
            usecols = ["timestamp", "ticker", *FEATURE_NAMES]
            reader = pd.read_csv(
                HISTORICAL_DATA_FILE,
                usecols=usecols,
                parse_dates=["timestamp"],
                chunksize=_HISTORICAL_FEATURE_CHUNK_SIZE,
                low_memory=False,
            )
        except ValueError as exc:
            logger.error(
                "❌ Failed to prepare historical dataset iterator for %s: %s",
                ticker,
                exc,
            )
            reader = None
        except FileNotFoundError:
            logger.warning(
                "⚠️ Historical dataset %s disappeared before seeding %s.",
                HISTORICAL_DATA_FILE,
                ticker,
            )
            reader = None

        if reader is not None:
            for chunk in reader:
                ticker_rows = chunk[chunk["ticker"].astype(str) == ticker]
                if ticker_rows.empty:
                    continue
                ticker_rows = ticker_rows.sort_values("timestamp")
                for _, row in ticker_rows.iterrows():
                    feature_window.append(row.loc[FEATURE_NAMES].copy())

            if feature_window:
                feature_history.extend(feature_window)
                logger.info(
                    "📈 Seeded %d/%d sequence rows for %s from historical_data.csv.",
                    len(feature_history),
                    FEATURE_SEQUENCE_WINDOW,
                    ticker,
                )
                seeded = True
            else:
                logger.warning(
                    "⚠️ No historical feature rows found for %s in historical_data.csv.",
                    ticker,
                )
    else:
        logger.warning(
            "⚠️ Cannot seed %s feature history; %s is missing.",
            ticker,
            HISTORICAL_DATA_FILE,
        )

    if not seeded:
        seeded = _seed_feature_history_from_cache(ticker)

    if not seeded:
        seeded = _seed_feature_history_from_ibkr(ticker, ib_client)

    if seeded:
        _SEEDED_SEQUENCE_TICKERS.add(ticker)
    else:
        logger.warning(
            "⚠️ Unable to seed feature history for %s; will retry on next iteration.",
            ticker,
        )


def _suffix_for_timeframe(timeframe: str) -> str:
    return _TIMEFRAME_TO_SUFFIX[timeframe]


def build_feature_row(
    indicators: Dict[str, Dict[str, object]],
    price: float,
    iv: float,
    delta: float,
    sp500_pct: float,
    previous_features: Optional[pd.Series] = None,
) -> pd.Series:
    """Return a feature row populated with forward-filled indicator values."""

    defaults = default_feature_values(FEATURE_NAMES)
    if previous_features is not None and not previous_features.empty:
        for name in FEATURE_NAMES:
            prev_value = previous_features.get(name)
            if pd.notna(prev_value):
                defaults[name] = prev_value

    features = defaults
    dummy_fallbacks: list[tuple[str, float]] = []

    def _valid_numeric(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric):
            return None
        return numeric

    def _record_dummy(name: str, value: float) -> float:
        dummy_fallbacks.append((name, value))
        return value

    def _ffill_feature(name: str, neutral: float) -> float:
        if previous_features is not None and not previous_features.empty:
            prev_value = previous_features.get(name)
            prev_numeric = _valid_numeric(prev_value)
            if prev_numeric is not None:
                return prev_numeric
        return _record_dummy(name, neutral)

    for timeframe in TIMEFRAMES:
        suffix = _suffix_for_timeframe(timeframe)
        rsi = indicators.get("rsi", {}).get(timeframe, 0.0)
        macd = indicators.get("macd", {}).get(timeframe, 0.0)
        signal = indicators.get("signal", {}).get(timeframe, 0.0)
        ema10 = indicators.get("ema10", {}).get(timeframe, 0.0)
        ema10_dev = indicators.get("ema10_dev", {}).get(timeframe, 0.0)
        rsi_change = indicators.get("rsi_change", {}).get(timeframe, 0.0)
        macd_change = indicators.get("macd_change", {}).get(timeframe, 0.0)
        ema10_change = indicators.get("ema10_change", {}).get(timeframe, 0.0)
        price_above = indicators.get("price_above_ema10", {}).get(timeframe, False)
        boll = indicators.get("bollinger", {}).get(
            timeframe, {"upper": 0.0, "lower": 0.0, "mid": 0.0}
        )
        volume = indicators.get("volume", {}).get(timeframe, 0.0)
        tds_trend = indicators.get("tds_trend", {}).get(timeframe, 0)
        tds_signal = indicators.get("tds_signal", {}).get(timeframe, 0)
        td9 = encode_td9(indicators.get("td9_summary", {}).get(timeframe, ""))
        fib_summary = indicators.get("fib_summary", {}).get(timeframe, "")
        fib_levels, fib_zone_delta = derive_fibonacci_features(fib_summary, price)
        fib_time_count = count_fib_timezones(
            indicators.get("fib_time_zones", {}).get(timeframe, "")
        )
        zig = encode_zig(indicators.get("zig_zag_trend", {}).get(timeframe, ""))
        high_vol = int(bool(indicators.get("high_vol", {}).get(timeframe, False)))
        vol_spike = int(bool(indicators.get("vol_spike", {}).get(timeframe, False)))
        vol_cat = encode_vol_cat(
            indicators.get("vol_category", {}).get(timeframe, "")
        )
        atr = indicators.get("atr", {}).get(timeframe, 0.0)
        adx_value = indicators.get("adx", {}).get(timeframe)
        adx = _valid_numeric(adx_value)
        if adx is None:
            adx = _ffill_feature(f"adx_{suffix}", 25.0)

        obv_value = indicators.get("obv", {}).get(timeframe)
        obv = _valid_numeric(obv_value)
        if obv is None:
            obv = _ffill_feature(f"obv_{suffix}", 0.0)

        stochastic = indicators.get("stochastic", {}).get(timeframe) or {}
        stoch_k_val = _valid_numeric(stochastic.get("k"))
        stoch_k = (
            stoch_k_val
            if stoch_k_val is not None
            else _ffill_feature(f"stoch_k_{suffix}", 50.0)
        )
        stoch_d_val = _valid_numeric(stochastic.get("d"))
        stoch_d = (
            stoch_d_val
            if stoch_d_val is not None
            else _ffill_feature(f"stoch_d_{suffix}", 50.0)
        )

        features[f"rsi_{suffix}"] = rsi
        features[f"macd_{suffix}"] = macd
        features[f"signal_{suffix}"] = signal
        features[f"ema10_{suffix}"] = ema10
        features[f"ema10_dev_{suffix}"] = ema10_dev
        features[f"rsi_change_{suffix}"] = rsi_change
        features[f"macd_change_{suffix}"] = macd_change
        features[f"ema10_change_{suffix}"] = ema10_change
        features[f"price_above_ema10_{suffix}"] = int(bool(price_above))
        features[f"bb_upper_{suffix}"] = boll.get("upper", 0.0)
        features[f"bb_lower_{suffix}"] = boll.get("lower", 0.0)
        features[f"bb_mid_{suffix}"] = boll.get("mid", 0.0)
        features[f"volume_{suffix}"] = volume
        features[f"tds_trend_{suffix}"] = tds_trend
        features[f"tds_signal_{suffix}"] = tds_signal
        features[f"high_vol_{suffix}"] = high_vol
        features[f"vol_spike_{suffix}"] = vol_spike
        features[f"td9_{suffix}"] = td9
        features[f"zig_{suffix}"] = zig
        features[f"vol_cat_{suffix}"] = vol_cat
        features[f"fib_time_count_{suffix}"] = fib_time_count
        for idx, level in enumerate(fib_levels, start=1):
            features[f"fib_level{idx}_{suffix}"] = level
        features[f"fib_zone_delta_{suffix}"] = fib_zone_delta
        features[f"atr_{suffix}"] = atr
        features[f"adx_{suffix}"] = adx
        features[f"obv_{suffix}"] = obv
        features[f"stoch_k_{suffix}"] = stoch_k
        features[f"stoch_d_{suffix}"] = stoch_d

    level_weight_val = calculate_level_weight(indicators, price)
    features["level_weight"] = float(np.nan_to_num(level_weight_val, nan=0.0))
    features["iv"] = iv
    features["delta"] = delta
    features["sp500_above_20d"] = sp500_pct

    _derive_base_pattern_features(features)

    if dummy_fallbacks:
        feature_summary = ", ".join(
            f"{name}={value:.2f}" for name, value in dummy_fallbacks
        )
        logger.warning(
            "⚠️ Feeding dummy fallback values into live model features: %s",
            feature_summary,
        )

    return pd.Series(features).reindex(FEATURE_NAMES, fill_value=0.0)


_MAX_TIMEFRAME_BARS: Dict[str, int] = {
    "1 hour": 300,
    "4 hours": 200,
    "1 day": 400,
}


def _trim_live_frame(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    max_bars = _MAX_TIMEFRAME_BARS.get(timeframe, 1000)
    if len(df) > max_bars:
        original = len(df)
        df = df.iloc[-max_bars:]
        logger.debug(
            "Trimmed %s data to last %d bars (from %d)", timeframe, max_bars, original
        )

    return df


def _normalize_live_frame(
    df: Optional[pd.DataFrame],
    timeframe: str,
    reference: Optional[datetime] = None,
    trim: bool = True,
) -> pd.DataFrame:
    """Return a cleaned OHLCV frame aligned to the completed ``timeframe`` bar."""

    if df is None or df.empty:
        return pd.DataFrame()

    working = df.copy()

    if not isinstance(working.index, pd.DatetimeIndex):
        if "timestamp" in working.columns:
            idx = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
        else:
            idx = pd.to_datetime(working.index, utc=True, errors="coerce")
        working = working.set_index(idx)
    if working.index.tz is None:
        working.index = working.index.tz_localize("UTC")
    else:
        working.index = working.index.tz_convert("UTC")

    working = working.sort_index()
    keep_cols = [col for col in ["open", "high", "low", "close", "volume"] if col in working.columns]
    working = working[keep_cols]
    working = working.dropna(how="all")
    if working.empty:
        return pd.DataFrame()

    cutoff = _last_completed_bar_timestamp(timeframe, reference=reference)
    working = working[working.index <= cutoff]
    if working.empty:
        return pd.DataFrame()

    if trim:
        working = _trim_live_frame(working, timeframe)

    return working


def _resample_live_frame(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    agg = df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return agg.dropna(how="all")


def build_live_features_for_ml(
    ticker: str,
    df_1h: pd.DataFrame,
    df_4h: Optional[pd.DataFrame],
    df_1d: Optional[pd.DataFrame],
    iv: float,
    delta: float,
    sp500_above_20d: float,
    level_weight: float,
    vix: Optional[float] = None,
) -> pd.DataFrame:
    """Return a single-row dataframe matching the training feature schema."""

    def _resample_full(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """
        Resample dataframe to higher timeframe using standard pandas frequency alignment.
        This matches exactly how historical_data.csv was generated.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        return (
            df.resample(rule)
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna(how="all")
        )

    # --- MAIN FEATURE BUILDING LOGIC ---
    df_1h = _normalize_live_frame(df_1h, "1 hour")
    normalized_1h = df_1h.copy()

    if df_1h.empty or len(df_1h) < 100:
        logger.warning("Not enough 1h bars to build live features")
        return pd.DataFrame()

    # Resample higher timeframes if not provided
    if df_4h is None or df_4h.empty:
        df_4h = _resample_full(normalized_1h, "4h")
    df_4h = _normalize_live_frame(df_4h.copy(), "4 hours")

    if df_1d is None or df_1d.empty:
        df_1d = _resample_full(normalized_1h, "1d")
    df_1d = _normalize_live_frame(df_1d.copy(), "1 day")

    latest_frames = {
        "1h": df_1h.copy(),
        "4h": df_4h.copy() if not df_4h.empty else pd.DataFrame(),
        "1d": df_1d.copy() if not df_1d.empty else pd.DataFrame(),
    }

    start = df_1h.index.min().to_pydatetime()
    end = df_1h.index.max().to_pydatetime()

    augmented = _historical_augment_timeframe_features(normalized_1h)

    def _empty_pattern_frame(index: pd.Index, suffix: str) -> pd.DataFrame:
        columns = [f"{col}{suffix}" for col in DEFAULT_CANDLESTICK_PATTERN_COLUMNS]
        return pd.DataFrame(0, index=index, columns=columns)

    try:
        ohlc_1h = normalized_1h[["open", "high", "low", "close"]]
        ohlc_4h = df_4h[["open", "high", "low", "close"]] if not df_4h.empty else pd.DataFrame()
        ohlc_1d = df_1d[["open", "high", "low", "close"]] if not df_1d.empty else pd.DataFrame()

        # === CRITICAL FIX: Multi-timeframe candlestick patterns with correct suffixes ===
        pattern_1h = fast_candlestick_patterns(ohlc_1h, suffix="_1h") if not ohlc_1h.empty else _empty_pattern_frame(normalized_1h.index, "_1h")
        pattern_4h = fast_candlestick_patterns(ohlc_4h, suffix="_4h") if not ohlc_4h.empty else _empty_pattern_frame(normalized_1h.index, "_4h")
        pattern_daily = fast_candlestick_patterns(ohlc_1d, suffix="_1d") if not ohlc_1d.empty else _empty_pattern_frame(normalized_1h.index, "_1d")

        pattern_1h = pattern_1h.reindex(normalized_1h.index, method="ffill").fillna(0)
        pattern_4h = pattern_4h.reindex(normalized_1h.index, method="ffill").fillna(0)
        pattern_daily = pattern_daily.reindex(normalized_1h.index, method="ffill").fillna(0)

        augmented = pd.concat([augmented, pattern_1h, pattern_4h, pattern_daily], axis=1)
    except Exception as exc:
        logger.warning(
            "⚠️ Failed to compute live candlestick patterns with suffixes: %s",
            exc,
        )

    dataset = _historical_finalise_feature_frame(augmented, ticker, start, end)
    dataset = add_golden_price_features(dataset)
    if dataset.empty:
        logger.warning(
            "Historical finalisation produced no rows for live data on %s", ticker
        )
        return pd.DataFrame()

    latest_row = dataset.iloc[-1].copy()
    latest_row.name = None
    latest_row["ticker"] = ticker

    for base_name in EXTENDED_PATTERN_COLUMNS:
        for suffix in _PATTERN_SUFFIXES:
            column = f"{base_name}{suffix}"
            if column not in latest_row.index:
                latest_row[column] = 0.0

    def _set_feature(name: str, value: Optional[float], default: float = 0.0) -> None:
        latest_row[name] = (
            float(value) if value is not None and not pd.isna(value) else default
        )

    _set_feature("iv", iv, default=latest_row.get("iv", 0.0))
    _set_feature("delta", delta, default=latest_row.get("delta", 0.0))
    _set_feature(
        "sp500_above_20d", sp500_above_20d, latest_row.get("sp500_above_20d", 50.0)
    )
    _set_feature("level_weight", level_weight, latest_row.get("level_weight", 0.0))

    if vix is not None and "vix" in latest_row.index:
        latest_row["vix"] = float(vix)

    feature_row = latest_row.copy()

    # === 终极修复：让模型置信度从 0% 变成 90%（必须加！）===
    from self_learn import FEATURE_NAMES

    # 1. 补期权特征
    for col in ["delta_atm_call", "delta_atm_put", "iv_atm", "iv_rank_proxy"]:
        feature_row[col] = feature_row.get(col, 0.0)

    # 2. 补全所有训练时用到的 22×4=88 个扩展形态
    EXTENDED_PATTERNS = [
        "pattern_hammer", "pattern_inverted_hammer", "pattern_engulfing",
        "pattern_piercing_line", "pattern_morning_star", "pattern_three_white_soldiers",
        "pattern_hanging_man", "pattern_shooting_star", "pattern_evening_star",
        "pattern_three_black_crows"
    ]
    for base in EXTENDED_PATTERNS:
        for tf in ["", "_1h", "_4h", "_1d"]:
            col = f"{base}{tf}"
            feature_row[col] = feature_row.get(col, 0.0)

    # 3. 强制对齐训练时的特征顺序（最关键一步！）
    # 强制去重列名 + 转成一行 DataFrame（防止重复索引导致失败）
    feature_row = feature_row.copy()
    if feature_row.index.duplicated().any():
        cols = pd.Series(feature_row.index)
        cols = cols.groupby(cols).cumcount().astype(str).replace("0", "")
        cols = cols.where(cols == "", cols.str.replace("", "_dup"))
        feature_row.index = feature_row.index + cols

    feature_df = feature_row.to_frame().T
    feature_df = feature_df.infer_objects(copy=False)
    feature_df = add_legacy_candlestick_columns(feature_df)
    feature_df = feature_df.reindex(columns=FEATURE_NAMES, fill_value=0.0)
    # 用 feature_df 替代原来的 feature_row 传给模型
    live_df = feature_df

    timestamp = (
        pd.Timestamp(end, tz="UTC")
        if end.tzinfo is None
        else pd.Timestamp(end.astimezone(timezone.utc))
    )
    live_df.index = pd.DatetimeIndex([timestamp], name="timestamp")

    logger.info("Built live ML feature row for %s at %s", ticker, timestamp)
    return live_df


def _resample_full(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample dataframe to higher timeframe using standard pandas frequency alignment.
    This matches exactly how historical_data.csv was generated.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    return (
        df.resample(rule)
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna(how="all")
    )


def _derive_base_pattern_features(features: dict[str, Any]) -> None:
    # or even simpler:
    # features: dict
    """Populate non-suffixed candlestick pattern columns expected by training."""

    if not _BASE_PATTERN_COLUMNS:
        return

    for base_name in _BASE_PATTERN_COLUMNS:
        primary_key = f"{base_name}_{_TIMEFRAME_TO_SUFFIX['1 hour']}"
        base_value = features.get(primary_key)

        try:
            derived_val = float(base_value)
        except (TypeError, ValueError):
            derived_val = 0.0
        if math.isnan(derived_val):
            derived_val = 0.0

        if derived_val == 0.0:
            candidate_values: list[float] = []
            for suffix in _PATTERN_SUFFIX_ORDER:
                key = f"{base_name}_{suffix}"
                value = features.get(key)
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isnan(numeric):
                    continue
                candidate_values.append(numeric)
            if candidate_values:
                derived_val = max(candidate_values, key=lambda val: abs(val))

        features[base_name] = int(abs(derived_val) >= 1)

    if _BASE_PATTERN_COLUMNS:
        if all(int(features.get(base_name, 0) or 0) == 0 for base_name in _BASE_PATTERN_COLUMNS):
            logger.warning(
                "All base candlestick pattern features resolved to 0; upstream timeframe data may be missing."
            )
# ========== Market Data ==========
@retry(tries=5, delay=10, backoff=2, max_delay=120, jitter=2)
def get_historical_data(
    ib: IB,
    ticker: str,
    timeframe: str,
    duration: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    logger.info(
        f"Fetching {ticker} historical data for {timeframe} timeframe with duration {duration}."
    )
    _throttle_request(f"historical {ticker} {timeframe}")
    try:
        existing = load_curated_bars(ticker, timeframe)
        end_date = _last_completed_bar_timestamp(timeframe)
        tf_delta_map = {
            "1 hour": timedelta(hours=1),
            "4 hours": timedelta(hours=4),
            "1 day": timedelta(days=1),
        }
        delta = tf_delta_map.get(timeframe, timedelta(hours=1))
        if not existing.empty:
            last_ts = existing.index.max()
            start_date = last_ts + delta
            if start_date >= end_date:
                # Extend the window to fetch recent history in case of missed bars
                start_date = end_date - timedelta(days=10)
        else:
            days_needed = 10 * 365
            start_date = end_date - timedelta(days=days_needed)
        try:
            contract = _stock_contract(ib, ticker)
        except ValueError as exc:
            logger.error(f"❌ Could not qualify contract for {ticker}: {exc}")
            return existing if not existing.empty else None
        timeframe_map = {"1 hour": "1 hour", "4 hours": "4 hours", "1 day": "1 day"}
        ib_timeframe = timeframe_map.get(timeframe, "1 hour")
        duration_str = duration or get_optimal_duration(ticker, timeframe, existing)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_date,
            durationStr=duration_str,
            barSizeSetting=ib_timeframe,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
            timeout=30,
        )
        ib.sleep(5)
        if not bars:
            logger.warning(f"No data from IBKR for {ticker} on {timeframe}.")
            return existing if not existing.empty else None
        df = util.df(bars)
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index, utc=True)
        logger.info(
            f"✅ Fetched {len(df)} new rows from IBKR for {ticker} on {timeframe}."
        )
        source = "ibkr"
        if df is not None and not df.empty:
            save_raw_bars(df, ticker, timeframe, source, date_col_hint="date")
            save_curated_bars(df, ticker, timeframe, date_col_hint="date")
        return load_curated_bars(ticker, timeframe)
    except Exception as e:
        logger.error(
            f"❌ Failed to fetch historical data for {ticker} on {timeframe}: {e}\n{traceback.format_exc()}"
        )
        return existing if "existing" in locals() and not existing.empty else None
def _request_stock_market_data(ib: IB, ticker: str):
    """Fetch the latest ``ib.ticker`` snapshot for ``ticker``."""

    try:
        contract = _stock_contract(ib, ticker)
    except ValueError as exc:
        logger.warning(f"Failed to qualify contract for {ticker}: {exc}")
        return None
    _throttle_request(f"mkt data {ticker}")
    ib.reqMktData(contract, "", False, False)
    ib.sleep(2)
    return ib.ticker(contract)


def get_current_price(ib: IB, contract: Contract, timeout: float = 10.0) -> float:
    """Return the latest trade/close with a historical fallback."""

    ib.cancelMktData(contract)
    ib.sleep(0.1)
    ticker = ib.reqMktData(contract, "", False, False)
    deadline = time.time() + timeout

    last_price: Optional[float] = None
    close_price: Optional[float] = None

    while time.time() < deadline:
        ib.sleep(0.1)
        last_price = getattr(ticker, "last", None)
        if last_price is not None and not math.isnan(last_price):
            ib.cancelMktData(contract)
            return float(last_price)
        close_price = getattr(ticker, "close", None)
        if close_price is not None and not math.isnan(close_price):
            ib.cancelMktData(contract)
            return float(close_price)

    ib.cancelMktData(contract)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime=datetime.now(timezone.utc),
        durationStr="1 D",
        barSizeSetting="1 hour",
        whatToShow="TRADES",
        useRTH=True,
    )
    if bars:
        return float(bars[-1].close)

    raise ValueError("Could not get price")


def get_stock_price(ib: IB, ticker: str) -> Optional[float]:
    """Fetch the latest market price for ``ticker`` with fallback logic."""

    try:
        contract = _stock_contract(ib, ticker)
    except ValueError as exc:
        logger.error(f"❌ Failed to qualify contract for {ticker}: {exc}")
        return None

    try:
        price = get_current_price(ib, contract)
        logger.info(f"Current price for {ticker}: ${price:.2f}")
        return price
    except Exception as e:
        logger.error(
            f"Failed to get stock price for {ticker}: {e}\n{traceback.format_exc()}"
        )
        return None


def get_stock_quote(ib: IB, ticker: str) -> tuple[Optional[float], Optional[float]]:
    """Return the latest ask/bid quote for ``ticker``."""

    try:
        market_data = _request_stock_market_data(ib, ticker)
        if not market_data:
            return None, None
        ask = getattr(market_data, "ask", None)
        bid = getattr(market_data, "bid", None)
        ask_price = float(ask) if ask and ask > 0 else None
        bid_price = float(bid) if bid and bid > 0 else None
        return ask_price, bid_price
    except Exception as e:
        logger.error(
            f"Failed to get quote for {ticker}: {e}\n{traceback.format_exc()}"
        )
        return None, None
def get_daily_open_ibkr(ib: IB, ticker: str) -> Optional[float]:
    """Fetch the latest daily bar's open price directly from IBKR."""
    try:
        contract = _stock_contract(ib, ticker)
        end_time = _last_completed_bar_timestamp("1 day")
        end_time_str = end_time.strftime("%Y%m%d %H:%M:%S UTC")
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_time_str,
            durationStr="1 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
        )
        ib.sleep(2)
        if not bars:
            logger.warning(f"No daily bar data for {ticker} from IBKR.")
            return None
        return float(bars[-1].open)
    except Exception as e:
        logger.error(f"Failed to get daily open for {ticker}: {e}\n{traceback.format_exc()}")
        return None
def get_recent_hourly_bars_ibkr(ib: IB, ticker: str) -> Optional[pd.DataFrame]:
    """Fetch the last two 1-hour bars for ``ticker`` directly from IBKR."""
    try:
        contract = _stock_contract(ib, ticker)
        end_time = _last_completed_bar_timestamp("1 hour")
        end_time_str = end_time.strftime("%Y%m%d %H:%M:%S UTC")
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_time_str,
            durationStr=format_duration(1, "D"),
            barSizeSetting="1 hour",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
        )
        ib.sleep(2)
        if not bars:
            logger.warning(f"No hourly bar data for {ticker} from IBKR.")
            return None
        df = util.df(bars)[["date", "open", "high", "low", "close", "volume"]]
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    except Exception as e:
        logger.error(f"Failed to get hourly bars for {ticker}: {e}\n{traceback.format_exc()}")
        return None
_STRIKE_CACHE: Dict[Tuple[str, str, str], List[float]] = {}
def snap_to_strike_interval(price: float, desired_strike: float) -> float:
    """Snap ``desired_strike`` to the nearest valid increment."""
    if price < 100:
        interval = 1.0
    elif price <= 200:
        interval = 2.5
    else:
        interval = 5.0
    return round(desired_strike / interval) * interval
def calculate_strike(current_price: float, option_type: str) -> float:
    """Approximate a desired strike 1% OTM/ITM based on option side."""
    if option_type.upper() == "CALL":
        return current_price * 1.01
    return current_price * 0.99
def get_nearest_strike(
    ib: IB,
    ticker: str,
    current_price: float,
    option_type: str,
    expiry: str,
) -> Optional[float]:
    """Select nearest valid strike for ``ticker`` and ``expiry``.
    If no contracts are found for ``expiry`` the search advances to the next
    weekly expiry until a strike is located or a safeguard limit is reached.
    """
    right = "P" if option_type.upper() == "PUT" else "C"
    weeks_ahead = 0
    while weeks_ahead < 12: # limit retries to avoid infinite loops
        cache_key = (ticker, expiry, option_type)
        valid_strikes = _STRIKE_CACHE.get(cache_key)
        if valid_strikes is None:
            try:
                symbol, exchange, currency, _ = _resolve_contract_details(ticker)
                partial = Option(symbol, expiry, right=right, exchange=exchange, currency=currency)
                details = ib.reqContractDetails(partial)
                valid_strikes = sorted({d.contract.strike for d in details})
            except Exception as e:
                logger.error(f"❌ Failed to fetch strikes for {ticker}: {e}\n{traceback.format_exc()}")
                valid_strikes = []
            _STRIKE_CACHE[cache_key] = valid_strikes
        if valid_strikes:
            desired_strike = calculate_strike(current_price, option_type)
            desired_strike = snap_to_strike_interval(current_price, desired_strike)
            strike = min(valid_strikes, key=lambda s: abs(s - desired_strike))
            if abs(strike - desired_strike) > 2.5:
                logger.warning(
                    f"⚠️ Adjusted strike from {desired_strike} to nearest valid {strike}"
                )
            return strike
        weeks_ahead += 1
        expiry = get_valid_expiry(weeks_ahead)
    logger.error(
        f"❌ No valid {option_type} strikes for {ticker} after {weeks_ahead} weeks"
    )
    return None
def get_nearest_expiration(ib: IB, ticker: str) -> str:
    """Fetch the nearest available option expiration for ``ticker``.
    Tries to obtain the list of expirations from IBKR and returns the earliest
    one. Falls back to :func:`get_valid_expiry` if the request fails or no
    expirations are found.
    """
    if should_skip_option_chain(ticker):
        logger.info(
            "Skipping option chain lookup for %s; using fallback expiry", ticker
        )
        return get_valid_expiry()
    try:
        stock = _stock_contract(ib, ticker)
        chains = ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
        if chains:
            expirations = sorted(chains[0].expirations)
            if expirations:
                return expirations[0]
    except Exception as e:
        logger.warning(f"⚠️ Failed to get expirations for {ticker}: {e}\n{traceback.format_exc()}")
    return get_valid_expiry()
def get_valid_expiry(weeks_ahead: int = 0) -> str:
    """Return a contract expiry normalised to the nearest Friday."""
    raw = get_nearest_friday_expiry(weeks_ahead=weeks_ahead)
    return get_nearest_friday(datetime.strptime(raw, "%Y%m%d"))


def _next_weekly_option_expiry(reference: Optional[date] = None) -> str:
    """Return the next Friday expiry following ``reference`` (or today)."""

    if reference is None:
        reference = datetime.now().date()
    days_until_friday = (4 - reference.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    next_friday = reference + timedelta(days=days_until_friday)
    return next_friday.strftime("%Y%m%d")
def _fetch_option_greek(
    ib: IB, ticker: str, strike: float, expiration: str, greek: str
) -> float:
    """Request an option greek, retrying with earlier expiry on error 478."""
    contract = _option_contract(ticker, expiration, strike, "C")
    try:
        if not ib.qualifyContracts(contract):
            return 0.0
        ib.reqMktData(contract, "106", False, False) # 106 for greeks
        ib.sleep(2)
        ticker_data = ib.ticker(contract)
        if ticker_data.modelGreeks:
            return getattr(ticker_data.modelGreeks, greek) or 0.0
        return 0.0
    except RequestError as e:
        if e.code == 478:
            try:
                new_exp = (
                    datetime.strptime(expiration, "%Y%m%d") - timedelta(days=1)
                ).strftime("%Y%m%d")
                contract = _option_contract(ticker, new_exp, strike, "C")
                if not ib.qualifyContracts(contract):
                    return 0.0
                ib.reqMktData(contract, "106", False, False)
                ib.sleep(2)
                ticker_data = ib.ticker(contract)
                if ticker_data.modelGreeks:
                    return getattr(ticker_data.modelGreeks, greek) or 0.0
            except Exception as inner:
                logger.error(
                    f"Failed to get {greek} for {ticker} after retry: {inner}\n{traceback.format_exc()}"
                )
        else:
            logger.error(f"Failed to get {greek} for {ticker}: {e}\n{traceback.format_exc()}")
    except Exception as e:
        logger.error(f"Failed to get {greek} for {ticker}: {e}\n{traceback.format_exc()}")
    return 0.0
def get_iv(ib: IB, ticker: str, current_price: float) -> float:
    expiration = _next_weekly_option_expiry()
    strike = get_nearest_strike(ib, ticker, current_price, "CALL", expiration)
    if strike is None:
        expiration = get_valid_expiry()
        strike = get_nearest_strike(ib, ticker, current_price, "CALL", expiration)
    if strike is None:
        return 0.0
    return _fetch_option_greek(ib, ticker, strike, expiration, "impliedVol")


def get_delta(ib: IB, ticker: str, current_price: float) -> float:
    expiration = _next_weekly_option_expiry()
    strike = get_nearest_strike(ib, ticker, current_price, "CALL", expiration)
    if strike is None:
        expiration = get_valid_expiry()
        strike = get_nearest_strike(ib, ticker, current_price, "CALL", expiration)
    if strike is None:
        return 0.0
    return _fetch_option_greek(ib, ticker, strike, expiration, "delta")
def get_multi_timeframe_indicators(ib: IB, ticker: str) -> dict:
    """Build the full multi-timeframe indicator set used for live features."""

    indicators = {
        "rsi": {},
        "macd": {},
        "signal": {},
        "ema10": {},
        "ema10_dev": {},
        "price_above_ema10": {},
        "bollinger": {},
        "volume": {},
        "adx": {},
        "obv": {},
        "stochastic": {},
        "td9_summary": {},
        "tds_trend": {},
        "tds_signal": {},
        "fib_summary": {},
        "fib_time_zones": {},
        "pivot_points": {},
        "zig_zag_trend": {},
        "high_vol": {},
        "vol_spike": {},
        "vol_category": {},
        "atr": {},
        "rsi_change": {},
        "macd_change": {},
        "ema10_change": {},
        "candlestick_patterns": {},
    }

    timeframe_specs = PATTERN_TIMEFRAME_SPECS
    frame_cache: dict[str, pd.DataFrame] = {}

    for timeframe, _ in timeframe_specs:
        df = load_curated_bars(ticker, timeframe)
        if df.empty:
            frame_cache[timeframe] = pd.DataFrame()
            continue

        df = df.sort_index()
        if getattr(df.index, "tz", None) is None:
            df.index = df.index.tz_localize("UTC")

        cutoff_ts = _last_completed_bar_timestamp(timeframe)
        df = df[df.index <= cutoff_ts]
        if df.empty:
            logger.debug(
                "⚠️ No completed %s bars available for %s after alignment; skipping.",
                timeframe,
                ticker,
            )
            frame_cache[timeframe] = pd.DataFrame()
            continue

        frame_cache[timeframe] = df

    def _safe_latest(value: Optional[float], default: float = 0.0) -> float:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)

    MIN_ADX_ROWS = 15
    MIN_STOCH_ROWS = 14

    for timeframe, suffix in timeframe_specs:
        df = frame_cache.get(timeframe)
        if df is None or df.empty:
            continue

        row_count = len(df)

        def _has_min_rows(min_rows: int, label: str) -> bool:
            if row_count < min_rows:
                logger.warning(
                    "Insufficient %s data (%d rows) for %s calculation on %s.",
                    timeframe,
                    row_count,
                    label,
                    ticker,
                )
                return False
            return True

        rsi_source = df.tail(50)
        rsi_val = calculate_rsi(rsi_source)
        rsi_prev = None
        if len(rsi_source) > 1:
            rsi_prev = calculate_rsi(rsi_source.iloc[:-1])
        indicators["rsi"][timeframe] = _safe_latest(rsi_val)
        indicators["rsi_change"][timeframe] = _safe_latest(
            None
            if rsi_val is None or rsi_prev is None
            else rsi_val - rsi_prev
        )

        macd_df = df.tail(100)
        macd_val, signal_val = calculate_macd(macd_df)
        macd_prev = None
        if len(macd_df) > 1:
            macd_prev, _ = calculate_macd(macd_df.iloc[:-1])
        indicators["macd"][timeframe] = _safe_latest(macd_val)
        indicators["signal"][timeframe] = _safe_latest(signal_val)
        indicators["macd_change"][timeframe] = _safe_latest(
            None
            if macd_val is None or macd_prev is None
            else macd_val - macd_prev
        )

        if timeframe == "1 day":
            ema_series = df["close"].ewm(span=10, adjust=False).mean()
        else:
            daily_close = df["close"].resample("1D").last()
            daily_ema = daily_close.ewm(span=10, adjust=False).mean()
            ema_series = daily_ema.reindex(df.index, method="ffill")

        ema_last = ema_series.iloc[-1] if not ema_series.empty else 0.0
        ema_prev = ema_series.iloc[-2] if len(ema_series) > 1 else None
        price_last = df["close"].iloc[-1]
        indicators["ema10"][timeframe] = _safe_latest(ema_last)
        if ema_last:
            indicators["ema10_dev"][timeframe] = (price_last - ema_last) / ema_last
            indicators["price_above_ema10"][timeframe] = price_last > ema_last
        else:
            indicators["ema10_dev"][timeframe] = 0.0
            indicators["price_above_ema10"][timeframe] = False
        indicators["ema10_change"][timeframe] = _safe_latest(
            None if ema_prev is None else ema_last - ema_prev
        )

        bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(macd_df)
        indicators["bollinger"][timeframe] = {
            "upper": _safe_latest(bb_upper),
            "lower": _safe_latest(bb_lower),
            "mid": _safe_latest(bb_mid),
        }

        indicators["volume"][timeframe] = (
            float(df["volume"].iloc[-1]) if "volume" in df else 0.0
        )

        atr_val = calculate_atr(df)
        indicators["atr"][timeframe] = _safe_latest(atr_val)

        adx_val = calculate_adx(df) if _has_min_rows(MIN_ADX_ROWS, "ADX") else None
        indicators.setdefault("adx", {})[timeframe] = _safe_latest(adx_val, default=25.0)

        has_volume = "volume" in df and bool(df["volume"].abs().sum())
        if not has_volume:
            logger.warning(
                "No usable volume data for OBV calculation on %s timeframe for %s.",
                timeframe,
                ticker,
            )
            obv_val = None
        else:
            obv_val = calculate_obv(df)
        indicators.setdefault("obv", {})[timeframe] = _safe_latest(obv_val)

        if _has_min_rows(MIN_STOCH_ROWS, "Stochastic Oscillator"):
            stoch_k_val, stoch_d_val = calculate_stochastic_oscillator(df)
        else:
            stoch_k_val, stoch_d_val = None, None
        indicators["stochastic"][timeframe] = {
            "k": _safe_latest(stoch_k_val, default=50.0),
            "d": _safe_latest(stoch_d_val, default=50.0),
        }

        _, _, buy_count, sell_count = calculate_td_sequential(df)
        buy_count = int(buy_count or 0)
        sell_count = int(sell_count or 0)
        indicators["td9_summary"][timeframe] = summarize_td_sequential(
            buy_count, sell_count
        )

        tds_trend_val, tds_signal_val = calculate_tds_trend(df)
        indicators["tds_trend"][timeframe] = int(tds_trend_val or 0)
        indicators["tds_signal"][timeframe] = int(tds_signal_val or 0)

        pivots = detect_pivots(df)
        fib_levels = calculate_fib_levels_from_pivots(pivots)
        indicators["fib_summary"][timeframe] = str(fib_levels) if fib_levels else "N/A"
        indicators["fib_time_zones"][timeframe] = (
            calculate_fib_time_zones(pivots, df.index[-1]) or []
        )
        indicators["pivot_points"][timeframe] = calculate_pivot_points(df) or {}

        zig = calculate_zig_zag(df)
        if isinstance(zig, pd.Series) and not zig.empty:
            pivots = zig[zig == 1].index
            if len(pivots) >= 2:
                last_price = df.loc[pivots[-1], "close"] if "close" in df else 0
                prev_price = df.loc[pivots[-2], "close"] if "close" in df else 0
                if last_price > prev_price:
                    indicators["zig_zag_trend"][timeframe] = "Up"
                elif last_price < prev_price:
                    indicators["zig_zag_trend"][timeframe] = "Down"
                else:
                    indicators["zig_zag_trend"][timeframe] = "N/A"
            else:
                indicators["zig_zag_trend"][timeframe] = "N/A"
        else:
            indicators["zig_zag_trend"][timeframe] = "N/A"

        indicators["high_vol"][timeframe] = bool(
            calculate_high_volatility(df) or False
        )
        indicators["vol_spike"][timeframe] = bool(
            calculate_volume_spike(df) or False
        )
        indicators["vol_category"][timeframe] = calculate_volume_weighted_category(
            df
        ) or "Neutral Volume"

        suffix_key = f"_{suffix}" if suffix else ""
        pattern_df = get_candlestick_patterns(df.tail(100), suffix=suffix_key)
        if pattern_df.empty:
            timeframe_patterns = {
                name: 0 for name in DEFAULT_CANDLESTICK_PATTERN_NAMES
            }
        else:
            latest = pattern_df.iloc[-1]
            timeframe_patterns = {}
            for display_name, column_name, _ in _BASE_PATTERN_SPECS:
                column = f"{column_name}{suffix_key}"
                timeframe_patterns[display_name] = int(latest.get(column, 0))
        if not any(timeframe_patterns.values()):
            logger.warning(
                "No candlestick patterns detected for %s timeframe %s. Verify OHLC completeness.",
                ticker,
                timeframe,
            )
        indicators["candlestick_patterns"][timeframe] = timeframe_patterns

    indicators["sp500_above_20d"] = get_current_s5tw()
    return indicators
def ensure_connection(ib: IB) -> None:
    """Ensure the IB connection is active before placing orders."""
    try:
        if not getattr(ib, "isConnected", lambda: True)():
            ib.connect(ib.host, ib.port, clientId=ib.clientId)
    except Exception as e:
        logger.error(f"❌ Reconnection failed: {e}\n{traceback.format_exc()}")

def _build_stock_order(
    ib: IB,
    ticker: str,
    action: str,
    quantity: int,
    *,
    allow_after_hours: bool = False,
    reference_price: Optional[float] = None,
) -> object:
    """Create a market order at the nearest available market price.

    The helper now always builds ``MarketOrder`` objects so that trades execute
    immediately against the best available price. During extended hours the
    order is flagged with ``outsideRth`` so it can route after the regular
    session. ``reference_price`` is retained for potential logging and future
    safeguards.
    """

    action_upper = action.upper()
    in_regular_session = is_us_regular_trading_hours()

    order = MarketOrder(action_upper, quantity)

    if allow_after_hours:
        order.outsideRth = True

        if not in_regular_session:
            ask_price, bid_price = get_stock_quote(ib, ticker)
            nearest_price = ask_price if action_upper == "BUY" else bid_price
            if nearest_price:
                logger.info(
                    "Routing after-hours %s market order for %s near %.2f",
                    action_upper,
                    ticker,
                    nearest_price,
                )
            elif reference_price:
                logger.info(
                    "Routing after-hours %s market order for %s near reference %.2f",
                    action_upper,
                    ticker,
                    reference_price,
                )

    return order
def execute_stock_trade_ibkr(
    ib: IB,
    ticker: str,
    decision: str,
    price: float,
    *,
    equity_fraction: Optional[float] = None,
    source_label: str = "ML_STOCK",
) -> bool:
    """Execute a stock trade sized to a fraction of account equity.
    By default the function targets ``1%`` of total equity per trade. Hong Kong
    tickers continue to use a fixed ``100`` shares. The helper now enforces
    directional positioning:
    * ``"BUY"`` signals first cover any open short exposure before initiating a
      new long position.
    * ``"SELL"`` signals close existing long holdings before initiating or
      adding to short exposure.
    * If an open position is profitable, new orders in the same direction
      double the current holding.
    * If an open position is at a loss, new orders in the same direction are
      skipped.
    """
    if decision not in {"BUY", "SELL"}:
        return False

    if not ticker.isdigit() and ticker != "MC":
        try:
            session = _us_trading_session()
            if session != "regular":
                logger.info(
                    "Skipping trade — U.S. equities are outside regular hours (%s)",
                    session,
                )
                return False
            logger.info("Proceeding with %s U.S. session trade", session)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Failed trading session check: %s", exc)

    pos_qty, avg_cost = get_position_info(ib, ticker)
    try:
        net_liq = get_net_liquidity(ib) or 0.0
    except Exception as e:
        logger.error(f"Failed to fetch account equity: {e}\n{traceback.format_exc()}")
        net_liq = 0.0
    td9_trade = source_label == TD9_RULE_SOURCE
    last_pnl, last_size = get_last_trade_result(ticker, source_label)
    lot_size = get_round_lot_size(price)
    same_direction_as_position = (decision == "BUY" and pos_qty > 0) or (
        decision == "SELL" and pos_qty < 0
    )
    profitable_position = False
    losing_position = False
    if pos_qty != 0 and avg_cost > 0:
        if pos_qty > 0:
            profitable_position = price > avg_cost
            losing_position = price < avg_cost
        else:
            profitable_position = price < avg_cost
            losing_position = price > avg_cost
    if same_direction_as_position and losing_position:
        logger.info(
            "Skipping %s order for %s — existing position of %s shares is at a loss (avg %.2f vs. price %.2f).",
            decision,
            ticker,
            pos_qty,
            avg_cost,
            price,
        )
        return False

    if same_direction_as_position and profitable_position:
        qty_equity = _calculate_pyramiding_quantity(abs(pos_qty), lot_size, price, net_liq)
        if qty_equity <= 0:
            logger.info(
                "Insufficient equity to scale profitable %s position for %s. Skipping order.",
                ticker,
                decision,
            )
            return False
        logger.info(
            "🏗️ Pyramiding profitable position for %s: adding %s shares to double from %s.",
            ticker,
            qty_equity,
            abs(pos_qty),
        )
    elif ticker.isdigit():
        qty_equity = 100
    else:
        qty_equity, _ = _determine_position_size(
            ticker,
            price,
            net_liq,
            equity_fraction,
            last_pnl,
            last_size,
            allow_loss_doubling=False,
            max_equity_fraction=TD9_RULE_MAX_EQUITY_FRACTION if td9_trade else None,
        )
        if td9_trade and last_pnl is not None and last_pnl > 0 and qty_equity > 0:
            logger.info(
                "Previous TD9 trade for %s gained %.2f; reverting to base sizing (%s shares).",
                ticker,
                last_pnl,
                qty_equity,
            )
    if decision == "BUY":
        if pos_qty < 0:
            if not close_short_position(ib, ticker, price):
                logger.error(
                    f"❌ Unable to cover existing short for {ticker}; buy order aborted."
                )
                return False
            pos_qty = 0
        if qty_equity <= 0:
            logger.info("Calculated quantity 0; skipping buy.")
            return False
        try:
            contract = _stock_contract(ib, ticker)
            order = _build_stock_order(
                ib,
                ticker,
                "BUY",
                qty_equity,
                reference_price=price,
            )
            ensure_connection(ib)
            trade = ib.placeOrder(contract, order)
            ib.sleep(1)
            status = getattr(getattr(trade, "orderStatus", None), "status", "")
            if status in ("Filled", "Submitted"):
                now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                write_trade_csv(
                    {
                        "timestamp": now_utc,
                        "ticker": ticker,
                        "price": price,
                        "decision": "BUY",
                        "Volume": qty_equity,
                        "Source": source_label,
                    }
                )
                _record_open_trade(ticker, "LONG", qty_equity, price, source_label)
                log_trade_execution(
                    ticker=ticker,
                    action="BUY",
                    quantity=qty_equity,
                    price=price,
                    source=source_label,
                )
                logger.info(
                    f"✅ Executed BUY trade for {ticker}: {qty_equity} shares at {price}"
                )
                return True
            logger.error(f"❌ Order failed: {status}\n{traceback.format_exc()}")
        except Exception as e:
            logger.error(f"Failed to execute BUY trade for {ticker}: {e}\n{traceback.format_exc()}")
        return False
    if pos_qty > 0:
        if not close_stock_position(ib, ticker, price):
            logger.error(
                f"❌ Unable to close existing long for {ticker}; short sell aborted."
            )
            return False
        pos_qty = 0
    if qty_equity <= 0:
        logger.info("Calculated quantity 0; skipping short sell.")
        return False
    return open_short_position(
        ib,
        ticker,
        qty_equity,
        price,
        source_label=source_label,
    )
def execute_trade(
    ib: IB,
    ticker: str,
    option_type: str,
    current_price: float,
    order_size: int,
    side: str = "BUY",
    **kwargs,
) -> bool:
    try:
        if option_type == "CALL" and side.upper() == "SELL":
            position_size = get_position_size(ib, ticker)
            existing_calls = get_short_call_positions(ib, ticker)
            open_call_orders = get_open_short_call_orders(ib, ticker)
            _, exchange, _, _ = _resolve_contract_details(ticker)
            multiplier = int(get_option_multiplier(ticker, exchange))
            covered_shares = (existing_calls + open_call_orders) * multiplier
            available_shares = position_size - covered_shares
            if available_shares < order_size * multiplier:
                logger.warning(
                    f"⚠️ Insufficient holdings to sell {order_size} CALL contracts for {ticker}. "
                    f"Have {position_size} shares with {existing_calls + open_call_orders} CALLs already open."
                )
                return False
        if option_type == "PUT" and side.upper() == "SELL":
            cash_balance = get_cash_balance(ib) or 0
            if cash_balance <= 0:
                logger.warning(
                    f"⚠️ Insufficient cash to sell {order_size} PUT contracts for {ticker}. "
                    f"Available cash: ${cash_balance:.2f}"
                )
                return False
        if ticker.upper() == "TSLL":
            expiry = get_nearest_friday_expiry(days_ahead=2)  # use actual next expiry
        else:
            expiry = get_nearest_expiration(ib, ticker)
        logger.debug(f"Computed expiry: {expiry}")
        adjusted = get_nearest_friday(datetime.strptime(expiry, "%Y%m%d"))
        if adjusted != expiry:
            logger.debug(
                f"Adjusted expiry to nearest Friday: {adjusted}"
            )
            expiry = adjusted
        strike = get_nearest_strike(ib, ticker, current_price, option_type, expiry)
        if strike is None:
            return False
        right = "P" if option_type == "PUT" else "C"
        contract = _option_contract(ticker, expiry, strike, right)
        order = MarketOrder(side, order_size)
        trade = None
        for attempt in range(2):
            try:
                try:
                    qualified = ib.qualifyContracts(contract)
                    if not qualified:
                        return False
                    contract = qualified[0]
                    mismatches = []
                    if contract.lastTradeDateOrContractMonth != expiry:
                        mismatches.append(
                            f"expiry: requested {expiry}, got {contract.lastTradeDateOrContractMonth}"
                        )
                    if contract.strike != strike:
                        mismatches.append(
                            f"strike: requested {strike}, got {contract.strike}"
                        )
                    if contract.right.upper() != right:
                        mismatches.append(
                            f"right: requested {right}, got {contract.right}"
                        )
                    if mismatches:
                        expected_adjustment = (
                            len(mismatches) == 1
                            and "expiry" in mismatches[0]
                            and datetime.strptime(
                                contract.lastTradeDateOrContractMonth, "%Y%m%d"
                            )
                            == datetime.strptime(expiry, "%Y%m%d")
                            + timedelta(days=1)
                        )
                        if expected_adjustment:
                            logger.info(
                                f"✅ Expected expiry adjustment for {ticker}: {expiry} -> {contract.lastTradeDateOrContractMonth}"
                            )
                            # IB may return the Saturday expiry while the contract's conId
                            # still maps to the prior Friday. Override the expiry to the
                            # originally requested date to avoid conflicting parameters
                            # when placing the order.
                            contract.lastTradeDateOrContractMonth = expiry
                        else:
                            logger.error(
                                f"⚠️ Contract mismatch for {ticker}: {', '.join(mismatches)}\n{traceback.format_exc()}"
                            )
                            # Rebuild the contract using IB-provided details to ensure
                            # all fields match the contract's ``conId``.
                            expiry = contract.lastTradeDateOrContractMonth
                            strike = contract.strike
                            right = contract.right
                            contract = _option_contract(ticker, expiry, strike, right)
                            qualified = ib.qualifyContracts(contract)
                            if not qualified:
                                return False
                            contract = qualified[0]
                except RequestError as e:
                    if e.code == 200:
                        logger.warning(
                            f"⚠️ Invalid strike {strike} for {ticker}. Snapping to interval."
                        )
                        strike = snap_to_strike_interval(current_price, strike)
                        contract = _option_contract(ticker, expiry, strike, right)
                        qualified = ib.qualifyContracts(contract)
                        if not qualified:
                            return False
                        contract = qualified[0]
                    else:
                        raise
                ensure_connection(ib)
                trade = ib.placeOrder(contract, order)
                ib.sleep(2)
                break
            except RequestError as e:
                if e.code == 200:
                    logger.error(
                        f"❌ No valid contract after adjustment for {ticker}\n{traceback.format_exc()}"
                    )
                    return False
                if e.code == 478 and attempt == 0:
                    new_exp = (
                        datetime.strptime(
                            contract.lastTradeDateOrContractMonth, "%Y%m%d"
                        )
                        - timedelta(days=1)
                    ).strftime("%Y%m%d")
                    expiry = new_exp
                    contract = _option_contract(ticker, new_exp, strike, right)
                    continue
                logger.error(f"❌ Order failed: {e}\n{traceback.format_exc()}")
                return False
        else:
            return False
        if not trade or trade.orderStatus.status == "Cancelled":
            ib.sleep(1)
            ensure_connection(ib)
            # Create a fresh order to ensure a new order id is used on retry
            order = MarketOrder(side, order_size)
            trade = ib.placeOrder(contract, order)
            ib.sleep(2)
        status = getattr(getattr(trade, "orderStatus", None), "status", "")
        if status not in ("Filled", "Submitted"):
            message = (
                trade.log[-1].message
                if trade and getattr(trade, "log", None)
                else "Unknown error"
            )
            logger.error(f"❌ Order failed: {message}\n{traceback.format_exc()}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to execute {side} {option_type} trade for {ticker}: {e}\n{traceback.format_exc()}")
        return False
def close_stock_position(
    ib: IB, ticker: str, price: float, allow_after_hours: bool = False
) -> bool:
    """Sell all shares of ``ticker``.
    Parameters
    ----------
    ib
        Active ``ib_insync.IB`` connection.
    ticker
        Symbol whose position should be closed.
    price
        Reference price used for logging and trade history.
    allow_after_hours
        When ``True`` the resulting order is flagged to execute outside of
        regular trading hours if the venue permits it.
    Returns
    -------
    bool
        ``True`` if the position was closed successfully, otherwise
        ``False``.
    """
    if not ticker.isdigit() and ticker != "MC":
        try:
            session = _us_trading_session()
            if session != "regular":
                if not allow_after_hours:
                    logger.info(
                        "Skipping close — U.S. equities are outside regular hours (%s)",
                        session,
                    )
                    return False
                logger.info(
                    "Proceeding outside regular hours (%s) because allow_after_hours=True",
                    session,
                )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Failed trading session check: %s", exc)

    try:
        pos_qty, _ = get_position_info(ib, ticker)
        if pos_qty <= 0:
            logger.info(f"No stock position to close for {ticker}.")
            return False
        contract = _stock_contract(ib, ticker)
        order = _build_stock_order(
            ib,
            ticker,
            "SELL",
            pos_qty,
            allow_after_hours=allow_after_hours,
            reference_price=price,
        )
        ensure_connection(ib)
        trade = ib.placeOrder(contract, order)
        ib.sleep(1)
        status = getattr(getattr(trade, "orderStatus", None), "status", "")
        if status in ("Filled", "Submitted"):
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            write_trade_csv(
                {
                    "timestamp": now_utc,
                    "ticker": ticker,
                    "price": price,
                    "decision": "STOCK_CLOSE",
                    "Volume": pos_qty,
                    "Source": "ML_STOCK_CLOSE",
                }
            )
            _record_completed_trade(ticker, "LONG", price, pos_qty)
            log_trade_execution(
                ticker=ticker,
                action="STOCK_CLOSE",
                quantity=pos_qty,
                price=price,
                source="ML_STOCK_CLOSE",
            )
            logger.info(
                f"✅ Closed stock position for {ticker}: {pos_qty} shares at {price}"
            )
            return True
        logger.error(f"❌ Stock close order failed: {status}\n{traceback.format_exc()}")
        return False
    except Exception as e:
        logger.error(f"Failed to close stock position for {ticker}: {e}\n{traceback.format_exc()}")
        return False
def close_short_position(
    ib: IB, ticker: str, price: float, allow_after_hours: bool = False
) -> bool:
    """Buy-to-cover all short shares of ``ticker``."""
    if not ticker.isdigit() and ticker != "MC":
        try:
            session = _us_trading_session()
            if session != "regular":
                if not allow_after_hours:
                    logger.info(
                        "Skipping close — U.S. equities are outside regular hours (%s)",
                        session,
                    )
                    return False
                logger.info(
                    "Proceeding outside regular hours (%s) because allow_after_hours=True",
                    session,
                )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Failed trading session check: %s", exc)

    try:
        pos_qty, _ = get_position_info(ib, ticker)
        if pos_qty >= 0:
            logger.info(f"No short position to cover for {ticker}.")
            return False
        qty_to_cover = abs(pos_qty)
        contract = _stock_contract(ib, ticker)
        order = _build_stock_order(
            ib,
            ticker,
            "BUY",
            qty_to_cover,
            allow_after_hours=allow_after_hours,
            reference_price=price,
        )
        ensure_connection(ib)
        trade = ib.placeOrder(contract, order)
        ib.sleep(1)
        status = getattr(getattr(trade, "orderStatus", None), "status", "")
        if status in ("Filled", "Submitted"):
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            write_trade_csv(
                {
                    "timestamp": now_utc,
                    "ticker": ticker,
                    "price": price,
                    "decision": "SHORT_CLOSE",
                    "Volume": qty_to_cover,
                    "Source": "ML_SHORT_CLOSE",
                }
            )
            _record_completed_trade(ticker, "SHORT", price, qty_to_cover)
            log_trade_execution(
                ticker=ticker,
                action="SHORT_CLOSE",
                quantity=qty_to_cover,
                price=price,
                source="ML_SHORT_CLOSE",
            )
            logger.info(
                f"✅ Covered short position for {ticker}: {qty_to_cover} shares at {price}"
            )
            return True
        logger.error(f"❌ Short cover order failed: {status}\n{traceback.format_exc()}")
        return False
    except Exception as e:
        logger.error(f"Failed to cover short position for {ticker}: {e}\n{traceback.format_exc()}")
        return False
def open_short_position(
    ib: IB,
    ticker: str,
    quantity: int,
    price: float,
    allow_after_hours: bool = False,
    *,
    source_label: str = "ML_STOCK",
) -> bool:
    """Sell shares short at market for ``ticker``."""
    if not ticker.isdigit() and ticker != "MC":
        try:
            session = _us_trading_session()
            if session != "regular":
                if not allow_after_hours:
                    logger.info(
                        "Skipping short — U.S. equities are outside regular hours (%s)",
                        session,
                    )
                    return False
                logger.info(
                    "Proceeding outside regular hours (%s) because allow_after_hours=True",
                    session,
                )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Failed trading session check: %s", exc)

    if quantity <= 0:
        logger.info("Calculated quantity 0; skipping short sell.")
        return False
    try:
        contract = _stock_contract(ib, ticker)
        order = _build_stock_order(
            ib,
            ticker,
            "SELL",
            quantity,
            allow_after_hours=allow_after_hours,
            reference_price=price,
        )
        ensure_connection(ib)
        trade = ib.placeOrder(contract, order)
        ib.sleep(1)
        status = getattr(getattr(trade, "orderStatus", None), "status", "") if trade else ""
        if status in ("Filled", "Submitted"):
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            write_trade_csv(
                {
                    "timestamp": now_utc,
                    "ticker": ticker,
                    "price": price,
                    "decision": "SELL_SHORT",
                    "Volume": quantity,
                    "Source": source_label,
                }
            )
            _record_open_trade(ticker, "SHORT", quantity, price, source_label)
            log_trade_execution(
                ticker=ticker,
                action="SELL_SHORT",
                quantity=quantity,
                price=price,
                source=source_label,
            )
            logger.info(
                f"✅ Executed SHORT trade for {ticker}: {quantity} shares at {price}"
            )
            return True
        logger.error(f"❌ Short sell order failed: {status}\n{traceback.format_exc()}")
        return False
    except Exception as e:
        logger.error(f"Failed to execute short sell for {ticker}: {e}\n{traceback.format_exc()}")
        return False
def close_option_positions(
    ib: IB,
    ticker: str,
    *,
    rights: Optional[Iterable[str]] = None,
    only_short: bool = False,
    require_profit: bool = False,
    log_description: Optional[str] = None,
) -> bool:
    """Close option positions for ``ticker`` matching the provided filters."""
    rights_filter = {r.upper() for r in rights} if rights else None
    description_parts: List[str] = []
    if only_short:
        description_parts.append("short")
    if rights_filter:
        description_parts.append("/".join(sorted(rights_filter)))
    description_parts.append("option")
    if require_profit:
        description_parts.insert(0, "profitable")
    description = log_description or " ".join(description_parts)
    closed_any = False
    matched_any = False
    eligible_any = False
    try:
        for position in ib.positions():
            contract = getattr(position, "contract", None)
            if not contract:
                continue
            if getattr(contract, "symbol", "") != ticker:
                continue
            if getattr(contract, "secType", "") != "OPT":
                continue
            right = getattr(contract, "right", "").upper()
            if rights_filter and right not in rights_filter:
                continue
            quantity = int(getattr(position, "position", 0))
            if quantity == 0:
                continue
            if only_short and quantity >= 0:
                continue
            matched_any = True
            if require_profit:
                pnl = getattr(position, "unrealizedPNL", None)
                if pnl is None:
                    logger.info(
                        "Skipping %s option for %s: unable to determine unrealized PnL.",
                        right or "?",
                        ticker,
                    )
                    continue
                if float(pnl) <= 0:
                    logger.info(
                        "Skipping %s option for %s: unrealized PnL %.2f <= 0.",
                        right or "?",
                        ticker,
                        float(pnl),
                    )
                    continue
            eligible_any = True
            action = "BUY" if quantity < 0 else "SELL"
            order = MarketOrder(action, abs(quantity))
            expiry = getattr(contract, "lastTradeDateOrContractMonth", "")
            strike = getattr(contract, "strike", "")
            try:
                ensure_connection(ib)
                trade = ib.placeOrder(contract, order)
                ib.sleep(1)
                status = getattr(getattr(trade, "orderStatus", None), "status", "")
                if status in ("Filled", "Submitted"):
                    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    decision = "CALL_CLOSE" if right == "C" else "PUT_CLOSE"
                    write_trade_csv(
                        {
                            "timestamp": now_utc,
                            "ticker": ticker,
                            "price": 0.0,
                            "decision": decision,
                            "Source": "ML_OPTION_CLOSE",
                        }
                    )
                    logger.info(
                        "✅ Closed %s %s option for %s (expiry %s, strike %s) via %s of %d contracts.",
                        "short" if quantity < 0 else "long",
                        "CALL" if right == "C" else "PUT" if right == "P" else right,
                        ticker,
                        expiry,
                        strike,
                        action,
                        abs(quantity),
                    )
                    closed_any = True
                else:
                    logger.error(
                        "❌ Option close order failed for %s (%s %s @ %s): %s\n{traceback.format_exc()}",
                        ticker,
                        expiry,
                        strike,
                        right or "?",
                        status,
                    )
            except Exception as exc: # pragma: no cover - network interaction
                logger.error(
                    "Failed to close %s option for %s (expiry %s, strike %s): %s\n{traceback.format_exc()}",
                    "CALL" if right == "C" else "PUT" if right == "P" else right or "?",
                    ticker,
                    expiry,
                    strike,
                    exc,
                )
        if not closed_any:
            if not matched_any:
                logger.info("No %s positions to close for %s.", description, ticker)
            elif not eligible_any:
                logger.info(
                    "No %s positions met close criteria for %s.",
                    description,
                    ticker,
                )
            else:
                logger.warning(
                    "⚠️ Unable to close existing %s positions for %s.",
                    description,
                    ticker,
                )
        return closed_any
    except Exception as exc:
        logger.error(
            "Failed to check option positions for %s: %s\n{traceback.format_exc()}",
            ticker,
            exc,
        )
        return False
def close_all_option_positions(
    ib: IB,
    *,
    rights: Optional[Iterable[str]] = None,
    only_short: bool = False,
    require_profit: bool = False,
) -> bool:
    """Close option positions across all tickers in the IBKR account.
    Parameters
    ----------
    ib : IB
        Connected IBKR client instance.
    rights : Iterable[str], optional
        Option rights (e.g. {"C"}, {"P"}) to restrict which contracts are
        closed. When omitted all rights are considered.
    only_short : bool, optional
        If ``True`` only short positions are targeted. Defaults to ``False``.
    require_profit : bool, optional
        If ``True`` only positions with positive unrealized PnL are closed.
    Returns
    -------
    bool
        ``True`` if at least one option position was closed.
    """
    rights_filter = {r.upper() for r in rights} if rights else None
    description_parts: List[str] = []
    if only_short:
        description_parts.append("short")
    if rights_filter:
        description_parts.append("/".join(sorted(rights_filter)))
    description_parts.append("option")
    if require_profit:
        description_parts.insert(0, "profitable")
    description = " ".join(description_parts)
    try:
        positions = ib.positions()
    except Exception as exc: # pragma: no cover - network interaction
        logger.error("Failed to retrieve IBKR positions: %s\n{traceback.format_exc()}", exc)
        return False
    option_tickers: set[str] = set()
    for position in positions:
        contract = getattr(position, "contract", None)
        if not contract:
            continue
        if getattr(contract, "secType", "") != "OPT":
            continue
        if rights_filter:
            right = getattr(contract, "right", "").upper()
            if right not in rights_filter:
                continue
        symbol = getattr(contract, "symbol", "")
        if not symbol:
            continue
        option_tickers.add(symbol)
    if not option_tickers:
        logger.info(
            "No %s positions found to close across the account.",
            description,
        )
        return False
    closed_any = False
    for ticker in sorted(option_tickers):
        closed = close_option_positions(
            ib,
            ticker,
            rights=rights_filter,
            only_short=only_short,
            require_profit=require_profit,
            log_description=description,
        )
        closed_any = closed_any or closed
    if not closed_any:
        logger.info(
            "No %s positions met the close criteria across the account.",
            description,
        )
    return closed_any
def close_call_option_positions(ib: IB, ticker: str) -> bool:
    """Buy to close any short call option positions for ``ticker``."""
    return close_option_positions(
        ib,
        ticker,
        rights={"C"},
        only_short=True,
        log_description="short CALL",
    )
def _parse_cli_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the trading agent entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive Brokers trading agent controller",
    )
    subparsers = parser.add_subparsers(dest="command")
    close_parser = subparsers.add_parser(
        "close-options",
        help="Close option positions across the IBKR account and exit.",
    )
    close_parser.add_argument(
        "--rights",
        nargs="+",
        choices=("C", "P"),
        metavar="RIGHT",
        help="Restrict closing to specified option rights (C for calls, P for puts).",
    )
    close_parser.add_argument(
        "--only-short",
        action="store_true",
        help="Only close short option positions.",
    )
    close_parser.add_argument(
        "--require-profit",
        action="store_true",
        help="Only close positions with positive unrealized PnL.",
    )
    parser.add_argument(
        "--defer-initial-sync",
        action="store_true",
        help=(
            "Skip the startup data-lake seeding and rely on the post-decision "
            "sync that runs after each trading cycle."
        ),
    )
    subparsers.add_parser(
        "update-historical",
        help="Append newly generated rows to historical_data.csv and exit.",
    )
    return parser.parse_args(argv)
def process_single_ticker(
    ib: IB,
    ticker: str,
    market_context: Optional[dict[str, object]] = None,
    *,
    feature_order: Optional[Sequence[str]] = None,
    scaler: Optional[object] = None,
) -> None:
    ensure_event_loop()
    closed_bar = _last_completed_bar_timestamp("1 hour")
    previous_bar = last_processed_bar.get(ticker)
    if previous_bar and closed_bar <= previous_bar:
        logger.info(
            "⏩ No new closed 1H bar for %s (last processed %s). Skipping.",
            ticker,
            previous_bar,
        )
        return
    last_processed_bar[ticker] = closed_bar
    logger.info(f"{ticker} using closed 1H bar: {closed_bar.astimezone(pytz.timezone('Asia/Hong_Kong'))}")
    put_trades = count_today_put_trades(ticker)
    call_trades = count_today_call_trades(ticker)
    if put_trades >= MAX_DAILY_PUT_TRADES and call_trades >= MAX_DAILY_CALL_TRADES:
        logger.warning(
            f"⚠️ Max daily trades reached for {ticker} (PUT: {put_trades}/{MAX_DAILY_PUT_TRADES}, CALL: {call_trades}/{MAX_DAILY_CALL_TRADES}). Skipping."
        )
        return

    _seed_feature_history_from_historical_data(ticker, ib)
    try:
        indicators = get_multi_timeframe_indicators(ib, ticker)
    except Exception as e:
        logger.warning(
            f"⚠️ Failed to fetch indicators for {ticker}: {e}\n{traceback.format_exc()}. Using default values."
        )
        indicators = {}

    def _default_candlestick_frame() -> dict[str, int]:
        return {name: 0 for name in DEFAULT_CANDLESTICK_PATTERN_NAMES}

    indicator_defaults: dict[str, object] = {
        "rsi": 0.0,
        "macd": 0.0,
        "signal": 0.0,
        "ema10": 0.0,
        "ema10_dev": 0.0,
        "price_above_ema10": False,
        "bollinger": {"upper": 0.0, "lower": 0.0, "mid": 0.0},
        "volume": 0.0,
        "adx": 0.0,
        "obv": 0.0,
        "stochastic": {"k": 0.0, "d": 0.0},
        "td9_summary": "N/A",
        "tds_trend": 0,
        "tds_signal": 0,
        "fib_summary": "N/A",
        "fib_time_zones": [],
        "pivot_points": {},
        "zig_zag_trend": "N/A",
        "high_vol": False,
        "vol_spike": False,
        "vol_category": "Neutral Volume",
    }

    for key in ("rsi_change", "macd_change", "ema10_change"):
        indicator_defaults.setdefault(key, 0.0)

    indicators.setdefault("candlestick_patterns", {})
    for timeframe in TIMEFRAMES:
        tf_patterns = indicators["candlestick_patterns"].get(timeframe)
        if not isinstance(tf_patterns, dict):
            indicators["candlestick_patterns"][timeframe] = _default_candlestick_frame()
        else:
            missing = set(DEFAULT_CANDLESTICK_PATTERN_NAMES) - set(tf_patterns)
            for name in missing:
                tf_patterns[name] = 0

    for indicator_key, default_value in indicator_defaults.items():
        container = indicators.get(indicator_key)
        if not isinstance(container, dict):
            container = {}
            indicators[indicator_key] = container
        for timeframe in TIMEFRAMES:
            value = container.get(timeframe)
            if value is None:
                if isinstance(default_value, dict):
                    container[timeframe] = dict(default_value)
                elif isinstance(default_value, list):
                    container[timeframe] = list(default_value)
                else:
                    container[timeframe] = default_value

    if "sp500_above_20d" not in indicators or indicators["sp500_above_20d"] is None:
        indicators["sp500_above_20d"] = 50.0
    @retry(tries=3, delay=5, backoff=2)
    def fetch_stock_price(ib: IB, ticker: str) -> Optional[float]:
        return get_stock_price(ib, ticker)
    try:
        current_price = fetch_stock_price(ib, ticker)
        if current_price is None:
            logger.error(
                f"❌ Failed to get current price for {ticker} after retries. Skipping."
            )
            return
    except Exception as e:
        logger.error(
            f"❌ Failed to get current price for {ticker} after retries: {e}\n{traceback.format_exc()}. Skipping."
        )
        return
    market_context = market_context or fetch_iv_delta_spx(ib)
    sp500_context_default = 50.0
    spx_flag = market_context.get("spx_above_20d")
    if isinstance(spx_flag, bool):
        sp500_context_default = 60.0 if spx_flag else 40.0
    per_ticker = market_context.get("per_ticker") if isinstance(market_context, dict) else None
    ticker_snapshot = {}
    if isinstance(per_ticker, dict):
        ticker_snapshot = per_ticker.get(ticker, {}) or {}

    iv_market = float(ticker_snapshot.get("iv_atm") or market_context.get("iv_atm", 0.0) or 0.0)
    if iv_market > 0:
        iv = iv_market
        logger.debug("Using live ATM IV %.4f for %s from market context", iv_market, ticker)
    else:
        iv = get_iv(ib, ticker, current_price)
    delta_market = float(
        ticker_snapshot.get("delta_atm_call")
        or market_context.get("delta_atm_call", 0.0)
        or 0.0
    )
    if delta_market != 0.0:
        delta = delta_market
        logger.debug(
            "Using live ATM call delta %.4f for %s from market context",
            delta_market,
            ticker,
        )
    else:
        delta = get_delta(ib, ticker, current_price)
    missing_feature_inputs: set[str] = set()

    def _get_indicator_value(
        indicator_key: str,
        timeframe: Optional[str] = None,
        default: Optional[object] = None,
        feature_label: Optional[str] = None,
    ):
        """Return indicator value while tracking missing entries.

        When the requested indicator/timeframe pair is absent the helper returns
        ``default`` (invoked if it is callable) and records the feature label so
        we can log which model inputs were missing before invoking the
        predictors.
        """

        label = feature_label or (
            f"{indicator_key}_{timeframe}" if timeframe else indicator_key
        )
        container = indicators.get(indicator_key)
        value = container
        missing = False
        if timeframe is None:
            if value is None:
                missing = True
        else:
            if not isinstance(container, dict):
                missing = True
                value = None
            else:
                value = container.get(timeframe)
                if value is None:
                    missing = True
        if missing:
            missing_feature_inputs.add(label)
            if callable(default):
                return default()
            return default
        return value
    # Extract indicators for all timeframes
    # 1 hour
    rsi_1h = _get_indicator_value("rsi", "1 hour", 0.0, "rsi_1h")
    macd_1h = _get_indicator_value("macd", "1 hour", 0.0, "macd_1h")
    signal_1h = _get_indicator_value("signal", "1 hour", 0.0, "signal_1h")
    ema10_1h = _get_indicator_value("ema10", "1 hour", 0.0, "ema10_1h")
    ema10_dev_1h = _get_indicator_value("ema10_dev", "1 hour", 0.0, "ema10_dev_1h")
    rsi_change_1h = _get_indicator_value("rsi_change", "1 hour", 0.0, "rsi_change_1h")
    macd_change_1h = _get_indicator_value("macd_change", "1 hour", 0.0, "macd_change_1h")
    ema10_change_1h = _get_indicator_value(
        "ema10_change", "1 hour", 0.0, "ema10_change_1h"
    )
    price_above_ema10_1h = _get_indicator_value(
        "price_above_ema10", "1 hour", False, "price_above_ema10_1h"
    )
    bollinger_1h = _get_indicator_value(
        "bollinger",
        "1 hour",
        lambda: {"upper": 0.0, "lower": 0.0, "mid": 0.0},
        "bollinger_1h",
    )
    volume_1h = _get_indicator_value("volume", "1 hour", 0, "volume_1h")
    atr_1h = _get_indicator_value("atr", "1 hour", 0.0, "atr_1h")
    adx_1h = _get_indicator_value("adx", "1 hour", 0.0, "adx_1h")
    obv_1h = _get_indicator_value("obv", "1 hour", 0.0, "obv_1h")
    stochastic_1h = _get_indicator_value(
        "stochastic",
        "1 hour",
        lambda: {"k": 0.0, "d": 0.0},
        "stochastic_1h",
    )
    stoch_k_1h = stochastic_1h.get("k", 0.0)
    stoch_d_1h = stochastic_1h.get("d", 0.0)
    td9_summary_1h = _get_indicator_value("td9_summary", "1 hour", "N/A", "td9_1h")
    tds_trend_1h = _get_indicator_value("tds_trend", "1 hour", 0, "tds_trend_1h")
    tds_signal_1h = _get_indicator_value("tds_signal", "1 hour", 0, "tds_signal_1h")
    fib_summary_1h = _get_indicator_value("fib_summary", "1 hour", "N/A", "fib_1h")
    fib_time_zones_1h = _get_indicator_value(
        "fib_time_zones", "1 hour", lambda: [], "fib_time_zones_1h"
    )
    zig_zag_trend_1h = _get_indicator_value(
        "zig_zag_trend", "1 hour", "N/A", "zig_zag_trend_1h"
    )
    high_vol_1h = _get_indicator_value("high_vol", "1 hour", False, "high_vol_1h")
    vol_spike_1h = _get_indicator_value("vol_spike", "1 hour", False, "vol_spike_1h")
    vol_category_1h = _get_indicator_value(
        "vol_category", "1 hour", "Neutral Volume", "vol_category_1h"
    )
    # 4 hours
    rsi_4h = _get_indicator_value("rsi", "4 hours", 0.0, "rsi_4h")
    macd_4h = _get_indicator_value("macd", "4 hours", 0.0, "macd_4h")
    signal_4h = _get_indicator_value("signal", "4 hours", 0.0, "signal_4h")
    ema10_4h = _get_indicator_value("ema10", "4 hours", 0.0, "ema10_4h")
    ema10_dev_4h = _get_indicator_value("ema10_dev", "4 hours", 0.0, "ema10_dev_4h")
    rsi_change_4h = _get_indicator_value("rsi_change", "4 hours", 0.0, "rsi_change_4h")
    macd_change_4h = _get_indicator_value("macd_change", "4 hours", 0.0, "macd_change_4h")
    ema10_change_4h = _get_indicator_value(
        "ema10_change", "4 hours", 0.0, "ema10_change_4h"
    )
    price_above_ema10_4h = _get_indicator_value(
        "price_above_ema10", "4 hours", False, "price_above_ema10_4h"
    )
    bollinger_4h = _get_indicator_value(
        "bollinger",
        "4 hours",
        lambda: {"upper": 0.0, "lower": 0.0, "mid": 0.0},
        "bollinger_4h",
    )
    volume_4h = _get_indicator_value("volume", "4 hours", 0, "volume_4h")
    atr_4h = _get_indicator_value("atr", "4 hours", 0.0, "atr_4h")
    adx_4h = _get_indicator_value("adx", "4 hours", 0.0, "adx_4h")
    obv_4h = _get_indicator_value("obv", "4 hours", 0.0, "obv_4h")
    stochastic_4h = _get_indicator_value(
        "stochastic",
        "4 hours",
        lambda: {"k": 0.0, "d": 0.0},
        "stochastic_4h",
    )
    stoch_k_4h = stochastic_4h.get("k", 0.0)
    stoch_d_4h = stochastic_4h.get("d", 0.0)
    td9_summary_4h = _get_indicator_value("td9_summary", "4 hours", "N/A", "td9_4h")
    tds_trend_4h = _get_indicator_value("tds_trend", "4 hours", 0, "tds_trend_4h")
    tds_signal_4h = _get_indicator_value("tds_signal", "4 hours", 0, "tds_signal_4h")
    fib_summary_4h = _get_indicator_value("fib_summary", "4 hours", "N/A", "fib_4h")
    fib_time_zones_4h = _get_indicator_value(
        "fib_time_zones", "4 hours", lambda: [], "fib_time_zones_4h"
    )
    zig_zag_trend_4h = _get_indicator_value(
        "zig_zag_trend", "4 hours", "N/A", "zig_zag_trend_4h"
    )
    high_vol_4h = _get_indicator_value("high_vol", "4 hours", False, "high_vol_4h")
    vol_spike_4h = _get_indicator_value("vol_spike", "4 hours", False, "vol_spike_4h")
    vol_category_4h = _get_indicator_value(
        "vol_category", "4 hours", "Neutral Volume", "vol_category_4h"
    )
    # 1 day
    rsi_1d = _get_indicator_value("rsi", "1 day", 0.0, "rsi_1d")
    macd_1d = _get_indicator_value("macd", "1 day", 0.0, "macd_1d")
    signal_1d = _get_indicator_value("signal", "1 day", 0.0, "signal_1d")
    ema10_1d = _get_indicator_value("ema10", "1 day", 0.0, "ema10_1d")
    ema10_dev_1d = _get_indicator_value("ema10_dev", "1 day", 0.0, "ema10_dev_1d")
    rsi_change_1d = _get_indicator_value("rsi_change", "1 day", 0.0, "rsi_change_1d")
    macd_change_1d = _get_indicator_value("macd_change", "1 day", 0.0, "macd_change_1d")
    ema10_change_1d = _get_indicator_value(
        "ema10_change", "1 day", 0.0, "ema10_change_1d"
    )
    price_above_ema10_1d = _get_indicator_value(
        "price_above_ema10", "1 day", False, "price_above_ema10_1d"
    )
    bollinger_1d = _get_indicator_value(
        "bollinger",
        "1 day",
        lambda: {"upper": 0.0, "lower": 0.0, "mid": 0.0},
        "bollinger_1d",
    )
    volume_1d = _get_indicator_value("volume", "1 day", 0, "volume_1d")
    atr_1d = _get_indicator_value("atr", "1 day", 0.0, "atr_1d")
    adx_1d = _get_indicator_value("adx", "1 day", 0.0, "adx_1d")
    obv_1d = _get_indicator_value("obv", "1 day", 0.0, "obv_1d")
    stochastic_1d = _get_indicator_value(
        "stochastic",
        "1 day",
        lambda: {"k": 0.0, "d": 0.0},
        "stochastic_1d",
    )
    stoch_k_1d = stochastic_1d.get("k", 0.0)
    stoch_d_1d = stochastic_1d.get("d", 0.0)
    td9_summary_1d = _get_indicator_value("td9_summary", "1 day", "N/A", "td9_1d")
    tds_trend_1d = _get_indicator_value("tds_trend", "1 day", 0, "tds_trend_1d")
    tds_signal_1d = _get_indicator_value("tds_signal", "1 day", 0, "tds_signal_1d")
    fib_summary_1d = _get_indicator_value("fib_summary", "1 day", "N/A", "fib_1d")
    fib_time_zones_1d = _get_indicator_value(
        "fib_time_zones", "1 day", lambda: [], "fib_time_zones_1d"
    )
    zig_zag_trend_1d = _get_indicator_value(
        "zig_zag_trend", "1 day", "N/A", "zig_zag_trend_1d"
    )
    high_vol_1d = _get_indicator_value("high_vol", "1 day", False, "high_vol_1d")
    vol_spike_1d = _get_indicator_value("vol_spike", "1 day", False, "vol_spike_1d")
    vol_category_1d = _get_indicator_value(
        "vol_category", "1 day", "Neutral Volume", "vol_category_1d"
    )
    sp500_pct = _get_indicator_value(
        "sp500_above_20d",
        default=sp500_context_default,
        feature_label="sp500_above_20d",
    )
    # Encode non-numeric indicators
    td9_1h = encode_td9(td9_summary_1h)
    td9_4h = encode_td9(td9_summary_4h)
    td9_1d = encode_td9(td9_summary_1d)
    zig_1h = encode_zig(zig_zag_trend_1h)
    zig_4h = encode_zig(zig_zag_trend_4h)
    zig_1d = encode_zig(zig_zag_trend_1d)
    vol_cat_1h = encode_vol_cat(vol_category_1h)
    vol_cat_4h = encode_vol_cat(vol_category_4h)
    vol_cat_1d = encode_vol_cat(vol_category_1d)
    fib_levels_1h, fib_zone_delta_1h = derive_fibonacci_features(
        fib_summary_1h, current_price
    )
    fib_levels_4h, fib_zone_delta_4h = derive_fibonacci_features(
        fib_summary_4h, current_price
    )
    fib_levels_1d, fib_zone_delta_1d = derive_fibonacci_features(
        fib_summary_1d, current_price
    )
    fib_prices_1h = [lvl for lvl in fib_levels_1h if lvl]
    fib_prices_4h = [lvl for lvl in fib_levels_4h if lvl]
    fib_prices_1d = [lvl for lvl in fib_levels_1d if lvl]
    pivot_prices_all: list[float] = []
    for tf in ["1 hour", "4 hours", "1 day"]:
        pivot_dict = _get_indicator_value(
            "pivot_points",
            tf,
            lambda: {},
            feature_label=f"pivot_points_{tf.replace(' ', '')}",
        )
        pivot_prices_all.extend(list(pivot_dict.values()))
    all_fib_prices = fib_prices_1h + fib_prices_4h + fib_prices_1d
    level_weight = calculate_level_weight(
        current_price, all_fib_prices, pivot_prices_all
    )
    fib_time_count_1h = count_fib_timezones(fib_time_zones_1h)
    fib_time_count_4h = count_fib_timezones(fib_time_zones_4h)
    fib_time_count_1d = count_fib_timezones(fib_time_zones_1d)
    candlestick_patterns = indicators.get("candlestick_patterns", {})
    pattern_features_map: dict[str, int] = {}
    for timeframe, display_name, feature_name in PATTERN_FEATURE_SPECS:
        timeframe_patterns = candlestick_patterns.get(timeframe, {})
        match_val = timeframe_patterns.get(display_name, 0)
        pattern_features_map[feature_name] = 1 if bool(match_val) else 0
    timeframe_labels = {"1 hour": "1H", "4 hours": "4H", "1 day": "1D"}
    for timeframe, _ in PATTERN_TIMEFRAME_SPECS:
        tf_patterns = candlestick_patterns.get(timeframe, {})
        label = timeframe_labels.get(timeframe, timeframe)
        logger.info(f"🕯️ Candlestick Patterns ({label}): {tf_patterns}")
        active_patterns = [name for name, value in tf_patterns.items() if value]
        if active_patterns:
            logger.info(
                f"🕯️ Active Candlestick Patterns ({label}): {', '.join(active_patterns)}"
            )
        else:
            logger.info(f"🕯️ Active Candlestick Patterns ({label}): none")
    # Create expanded feature vector using the shared helper and keep a history
    df_live_1h = load_curated_bars(ticker, "1 hour")
    df_live_4h = load_curated_bars(ticker, "4 hours")
    df_live_1d = load_curated_bars(ticker, "1 day")
    live_feature_frame = build_live_features_for_ml(
        ticker=ticker,
        df_1h=df_live_1h,
        df_4h=df_live_4h,
        df_1d=df_live_1d,
        iv=iv,
        delta=delta,
        sp500_above_20d=sp500_pct,
        level_weight=level_weight,
        vix=market_context.get("vix"),
    )
    if live_feature_frame.empty:
        logger.warning("⚠️ Unable to build live features for %s; skipping.", ticker)
        return

    feature_history = FEATURE_HISTORY[ticker]
    feature_row_dict = live_feature_frame.iloc[0].to_dict()
    for key in (
        "iv_atm",
        "delta_atm_call",
        "delta_atm_put",
        "spx_above_20d",
        "vix",
        "iv_rank_proxy",
    ):
        if key in FEATURE_NAMES and key in market_context:
            feature_row_dict[key] = market_context[key]
    for key, value in pattern_features_map.items():
        feature_row_dict[key] = value

    # === FINAL FIX: Make live features 100% match training ===
    # 1. Add ALL 22 extended candlestick patterns (even if zero)
    EXTENDED_PATTERNS = [
        "pattern_hammer",
        "pattern_inverted_hammer",
        "pattern_engulfing",
        "pattern_piercing_line",
        "pattern_morning_star",
        "pattern_three_white_soldiers",
        "pattern_hanging_man",
        "pattern_shooting_star",
        "pattern_evening_star",
        "pattern_three_black_crows",
    ]

    for base in EXTENDED_PATTERNS:
        for tf in ["", "_1h", "_4h", "_1d"]:
            col = base + tf
            if col not in feature_row_dict:
                feature_row_dict[col] = 0.0

    # 2. Add missing option/IV features (they were in training)
    for col in ["delta_atm_call", "delta_atm_put", "iv_atm", "iv_rank_proxy"]:
        if col not in feature_row_dict:
            feature_row_dict[col] = 0.0

    # 3. Ensure old-style pattern names exist (backward compat)
    OLD_TO_NEW = {
        "pattern_bearish_engulfing_1d": "pattern_bearish_engulfing",
        "pattern_bullish_engulfing_1d": "pattern_bullish_engulfing",
        "pattern_hammer_1d": "pattern_hammer",
        "pattern_shooting_star_1d": "pattern_shooting_star",
    }
    for old, new in OLD_TO_NEW.items():
        if old in TRAINING_FEATURE_NAMES and old not in feature_row_dict:
            feature_row_dict[old] = feature_row_dict.get(new, 0.0)

    feature_df = pd.DataFrame([feature_row_dict])
    feature_df = feature_df.reindex(columns=FEATURE_NAMES, fill_value=0.0)
    feature_df = feature_df.astype(float)
    feature_row = feature_df.iloc[0]

    raw_history = RAW_FEATURE_HISTORY[ticker]
    raw_history.append(feature_row)
    raw_df = pd.DataFrame(raw_history)
    raw_df["ticker"] = ticker
    df = raw_df.copy()
    if df.empty:
        logger.info(
            "⏳ No confirmed previous bar yet for %s. Skipping this cycle until one is available.",
            ticker,
        )
        return

    confirmed_row = df.iloc[-1].drop(labels=["ticker"], errors="ignore")
    confirmed_row = confirmed_row.reindex(FEATURE_NAMES, fill_value=0.0)
    feature_history.append(confirmed_row)
    ml_features = _sequence_dataframe(feature_history)
    if missing_feature_inputs:
        logger.warning(
            "⚠️ Missing indicator inputs for %s: %s",
            ticker,
            ", ".join(sorted(missing_feature_inputs)),
        )
    prepared_features = enforce_training_consistency(ml_features)
    if prepared_features is None or prepared_features.empty:
        logger.info(
            "⏳ Awaiting sufficient shifted feature history for %s; skipping ML vote this cycle.",
            ticker,
        )
        return

    feature_payload: object = prepared_features
    if isinstance(feature_payload, pd.Series):
        temp_df = feature_payload.to_frame().T
    elif isinstance(feature_payload, Mapping):
        temp_df = pd.DataFrame([feature_payload])
    elif isinstance(feature_payload, (list, tuple, np.ndarray)):
        temp_array = np.atleast_2d(feature_payload)
        temp_df = pd.DataFrame(temp_array, columns=FEATURE_NAMES)
    elif isinstance(feature_payload, pd.DataFrame):
        temp_df = feature_payload.copy()
    else:
        temp_df = pd.DataFrame([feature_payload])

    for col in FEATURE_NAMES:
        if col not in temp_df.columns:
            temp_df[col] = 0.0
    temp_df = temp_df[FEATURE_NAMES]

    current_bar_time = _last_completed_bar_timestamp(
        "1 hour", reference=datetime.now(timezone.utc)
    ).astimezone(timezone.utc)
    if len(temp_df) == 1:
        temp_df.index = pd.DatetimeIndex([current_bar_time])
    else:
        temp_df.index = pd.date_range(
            end=current_bar_time,
            periods=len(temp_df),
            freq="1H",
            tz="UTC",
        )

    historical_path = Path("/app/data/historical_data.csv")
    if historical_path.exists():
        try:
            hist_df = pd.read_csv(historical_path)
        except Exception as exc:
            logger.warning(
                "⚠️ Unable to read historical data from %s: %s",
                historical_path,
                exc,
            )
            hist_df = pd.DataFrame()
        if not hist_df.empty and {"timestamp", "ticker"}.issubset(hist_df.columns):
            hist_df = hist_df[hist_df["ticker"] == ticker].copy()
            if not hist_df.empty:
                hist_df["timestamp"] = pd.to_datetime(
                    hist_df["timestamp"], utc=True, errors="coerce"
                )
                hist_df = hist_df.dropna(subset=["timestamp"]).set_index("timestamp")
                hist_df = hist_df.sort_index()
                hist_df = hist_df.reindex(columns=FEATURE_NAMES, fill_value=0.0)
                history_window = hist_df.tail(500)
                combined = pd.concat([history_window, temp_df])
            else:
                combined = temp_df
        else:
            combined = temp_df
    else:
        combined = temp_df

    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    if combined.empty:
        logger.info(
            "⏳ Unable to assemble model features for %s; skipping ML vote this cycle.",
            ticker,
        )
        return

    latest_timestamp = combined.index[-1]
    if latest_timestamp.tzinfo is None:
        latest_timestamp = latest_timestamp.tz_localize(timezone.utc)
    timestamp_str = latest_timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    latest_features = combined.iloc[-1].reindex(FEATURE_NAMES, fill_value=0.0)
    feature_snapshot = {
        column: _json_safe_value(latest_features[column]) for column in FEATURE_NAMES
    }
    logger.info(
        "📊 Live ML feature snapshot for %s @ %s: %s",
        ticker,
        timestamp_str,
        json.dumps(feature_snapshot, sort_keys=True),
    )

    sequence_df = combined

    # --- PREDICTION & MODEL DECISIONS ---
    ticker_settings = _get_ticker_settings(ticker)
    preds, ppo_meta = predict_with_all_models(sequence_df)
    current_position = get_position_size(ib, ticker)

    # After you get the decision from ml_predictor
    decision, info = ml_predictor.independent_model_decisions(
        (preds, ppo_meta), return_details=True
    )

    ppo_action = info.get("ppo_action", 1)

    if decision == "Hold":
        target_size = 0.0
    else:
        # ---- Pyramiding logic (100% → 150% → 175%) ----
        if info.get("last_direction") != decision:
            consecutive = 1
        else:
            consecutive = info.get("consecutive_count", 1) + 1

        # PPO decides if we are allowed to be greedy
        ppo_action = info.get("ppo_action", 1)
        ppo_value = info.get("ppo_value", 0.0)
        ppo_ma = info.get("ppo_value_ma100", 0.0)
        ppo_std = info.get("ppo_value_std100", 0.5)
        ppo_entropy = info.get("ppo_entropy", 0.3)

        allow_pyramid = (
            ppo_action == 2 and
            ppo_value > ppo_ma + 0.5 * ppo_std and
            ppo_entropy < 0.40
        )

        early_exit = (
            ppo_action == 0 or
            ppo_entropy > 0.60 or
            (current_position != 0 and info.get("prev_ppo_action", 1) == 2 and ppo_action == 0)
        )

        if early_exit:
            target_size = 0.0
        elif consecutive == 1:
            target_size = 1.00
        elif consecutive == 2 and allow_pyramid:
            target_size = 1.50
        elif consecutive == 3 and allow_pyramid:
            target_size = 1.75
        else:
            target_size = current_position  # stay at 175% max

        direction = 1.0 if decision == "Buy" else -1.0
        target_size *= direction

    # Store for next bar
    info["last_direction"] = decision
    info["consecutive_count"] = consecutive if decision != "Hold" else 0
    info["prev_ppo_action"] = ppo_action

    decision_detail: dict[str, object] = info if isinstance(info, dict) else {}
    trigger = decision_detail.get("trigger") or "unknown"
    confidence = float(decision_detail.get("confidence", 0.0))

    detail_votes = decision_detail.get("votes", {}) if isinstance(
        decision_detail, Mapping
    ) else {}
    detail_confidences = decision_detail.get("confidences", {}) if isinstance(
        decision_detail, Mapping
    ) else {}

    decision_row = {
        "timestamp": latest_timestamp.astimezone(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        ),
        "ticker": ticker,
        "price": current_price,
        "decision_price": current_price,
        "final_decision": decision,
        "reason": decision_detail.get("reason", "")
        if isinstance(decision_detail, Mapping)
        else "",
        "trigger_model": decision_detail.get("trigger")
        if isinstance(decision_detail, Mapping)
        else None,
        "trigger_confidence": decision_detail.get("confidence")
        if isinstance(decision_detail, Mapping)
        else None,
    }

    for name in MODEL_NAMES:
        decision_row[f"{name}_vote"] = detail_votes.get(name)
        decision_row[f"{name}_confidence"] = detail_confidences.get(name)

    write_model_decision_log(decision_row)

    # FINAL SACRED RULE — NEVER OVERRIDE ml_predictor.py
    if decision == "Hold":
        logger.info(
            "HOLD RESPECTED: %s",
            decision_detail.get("reason", "High-confidence disagreement"),
        )
        return

    if decision not in {"Buy", "Sell"}:
        logger.info(
            "No valid ML signal for %s (got: %s). Skipping trade.",
            ticker,
            decision,
        )
        return

    logger.info(
        "EXECUTING %s (trigger: %s, confidence: %.3f)",
        decision.upper(),
        trigger,
        confidence,
    )
    logger.info(
        "ML predictor details → LSTM: %s (%.6f) | TRANSFORMER: %s (%.6f) | Final decision: %s (trigger %s, confidence %.6f)",
        detail_votes.get("LSTM", "N/A")
        if isinstance(detail_votes, Mapping)
        else "N/A",
        detail_confidences.get("LSTM", 0.0)
        if isinstance(detail_confidences, Mapping)
        else 0.0,
        detail_votes.get("Transformer", "N/A")
        if isinstance(detail_votes, Mapping)
        else "N/A",
        detail_confidences.get("Transformer", 0.0)
        if isinstance(detail_confidences, Mapping)
        else 0.0,
        decision,
        decision_detail.get("trigger", "unknown")
        if isinstance(decision_detail, Mapping)
        else "unknown",
        float(decision_detail.get("confidence", 0.0))
        if isinstance(decision_detail, Mapping)
        else 0.0,
    )

    ml_vote = decision
    stock_mapping = {"Buy": "BUY", "Sell": "SELL", "Hold": "HOLD"}
    stock_decision = stock_mapping.get(ml_vote, "HOLD")
    td9_rule_decision = evaluate_td9_rule(td9_1h)
    td9_rule_active = td9_rule_decision is not None
    if td9_rule_active:
        if stock_decision != td9_rule_decision:
            logger.info(
                "📐 TD9 1H rule overriding ML stock decision for %s: %s -> %s (td9_1h=%s)",
                ticker,
                stock_decision,
                td9_rule_decision,
                td9_1h,
            )
        else:
            logger.info(
                "📐 TD9 1H rule confirmed %s signal for %s (td9_1h=%s)",
                stock_decision,
                ticker,
                td9_1h,
            )
        stock_decision = td9_rule_decision
    stock_equity_fraction = TD9_RULE_EQUITY_FRACTION if td9_rule_active else None
    stock_trade_source = TD9_RULE_SOURCE if td9_rule_active else "ML_STOCK"
    pos_qty, avg_cost = get_position_info(ib, ticker)
    logger.info(f"ML stock decision for {ticker}: {stock_decision}")
    log_model_decision(
        ticker=ticker,
        decision=stock_decision,
        detail=decision_detail,
    )
    # Log the decision using the new helper (logs HOLD as well)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    decision_log = stock_decision # "BUY", "SELL", or "HOLD"
    tds_str = f"{tds_trend_1h}/{tds_signal_1h}"
    td9_str = td9_summary_1h
    fib_str = fib_summary_1h
    source = "ML"
    ai_decision = decision_log
    ml_decision = str(ml_vote).upper()
    model_decisions = _format_model_decisions(vote_detail)
    trade_side = "LONG" if stock_decision == "BUY" else "SHORT" if stock_decision == "SELL" else None
    write_trade_csv(
        {
            "timestamp": now_utc,
            "ticker": ticker,
            "price": current_price,
            "decision": decision_log,
            "fib": fib_str,
            "tds": tds_str,
            "td9": td9_str,
            "RSI": rsi_1h,
            "MACD": macd_1h,
            "Signal": signal_1h,
            "Volume": volume_1h,
            "IV": iv,
            "Delta": delta,
            "Source": source,
            "ai_decision": ai_decision,
            "ml_decision": ml_decision,
            **model_decisions,
        }
    )
    if stock_decision in {"BUY", "SELL"}:
        execute_stock_trade_ibkr(
            ib,
            ticker,
            stock_decision,
            current_price,
            equity_fraction=stock_equity_fraction,
            source_label=stock_trade_source,
        )
    else:
        logger.info(
            f"⏳ No stock trade executed for {ticker}. Decision: {stock_decision}"
        )


def batch_historical_requests(
    ib: IB, tickers: Sequence[str], timeframe: str
) -> dict[str, Optional[pd.DataFrame]]:
    """Fetch historical bars for ``tickers`` sequentially for the same timeframe.

    Running IBKR requests inside thread pools can fail because ib_insync relies on
    the event loop that lives on the main thread. Sequential processing keeps all
    calls on that loop while still centralizing logging and error handling.
    """

    if not tickers:
        return {}

    results: dict[str, Optional[pd.DataFrame]] = {}
    logger.info(
        "🚀 Sequentially requesting %s historical data for %d tickers",
        timeframe,
        len(tickers),
    )

    for ticker in tickers:
        try:
            results[ticker] = get_historical_data(ib, ticker, timeframe)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "❌ Failed to batch sync %s for %s: %s\n%s",
                timeframe,
                ticker,
                exc,
                traceback.format_exc(),
            )
            results[ticker] = None

    return results
def sync_ticker_data(ib: IB, ticker: str) -> None:
    """Update the data lake for *ticker* across all timeframes.
    Runs after trading decisions to ensure fresh data is available for the
    next evaluation cycle rather than gating trades on the sync itself.
    """
    timeframes = ["1 hour", "4 hours", "1 day"]
    for tf in timeframes:
        _throttle_request(f"sync {ticker} {tf}")
        try:
            duration = get_optimal_duration(ticker, tf)
            logger.info(
                "Fetching %s %s with smart duration %s",
                ticker,
                tf,
                duration,
            )
            df = get_historical_data(ib, ticker, tf, duration)
            if df is None or df.empty:
                logger.warning(
                    f"⚠️ No historical data fetched for {ticker} ({tf}). Skipping."
                )
                continue
            logger.info(f"✅ Synced data lake for {ticker} ({tf})")
        except Exception as e:
            logger.error(f"❌ Failed to sync data lake for {ticker} ({tf}): {e}\n{traceback.format_exc()}")
def update_all_tickers_data(ib: IB, tickers: Sequence[str]) -> None:
    """Update historical data for all tickers during US off-market hours."""

    if not tickers:
        tickers = load_tickers()

    timeframes = ["1 hour", "4 hours", "1 day"]
    for timeframe in timeframes:
        batched = batch_historical_requests(ib, list(tickers), timeframe)
        for ticker, df in batched.items():
            if df is None or df.empty:
                logger.warning(
                    "⚠️ No historical data fetched for %s (%s) during batch update.",
                    ticker,
                    timeframe,
                )
            else:
                logger.info("✅ Synced data lake for %s (%s)", ticker, timeframe)
# ============================== 终极串行版 process_tickers（保持单一 event loop）==============================
def process_tickers(ib: IB, tickers: Sequence[str]) -> None:
    logger.info(f"开始串行处理 {len(tickers)} 只股票，避免跨线程的 asyncio event loop 问题")

    for ticker in tickers:
        try:
            process_single_ticker(ib, ticker)
        except Exception as e:
            logger.error(f"{ticker} 处理失败: {e}", exc_info=True)

    logger.info("本轮所有股票处理完毕！")
# ================================================================


def is_us_equity_session_open(
    now: Optional[datetime] = None, *, include_extended: bool = True
) -> bool:
    """Return ``True`` when U.S. equities can trade (regular or extended)."""

    now = (now or datetime.now(US_EASTERN)).astimezone(US_EASTERN)

    if now.weekday() >= 5:
        return False

    if now.date() in US_MARKET_HOLIDAYS:
        return False

    session = _us_trading_session(now)
    if include_extended:
        return session in {"pre-market", "regular", "post-market"}

    return session == "regular"


def run_precise_hourly_trading_cycle() -> None:
    if not is_us_equity_session_open():
        logger.info("U.S. equity session closed (weekend/holiday) → skipping run")
        return

    now_et = datetime.now(US_EASTERN).strftime("%H:%M:%S ET")
    logger.info(f"Precise hourly bar close +1 second trigger @ {now_et}")

    tickers = get_cached_tickers()
    process_tickers(ib, tickers)


def schedule_precise_trading_jobs() -> None:
    # Clear any old jobs first (in case of hot-reload)
    schedule.clear("precise-trading")

    # Pre-market through post-market (04:00 → 20:00 ET) on the hour +1 second
    for hour in range(4, 21):
        hh = f"{hour:02d}"
        schedule.every().day.at(f"{hh}:00:01", tz="America/New_York").do(
            run_precise_hourly_trading_cycle
        ).tag("precise-trading")

    # First decision one second after the regular session opens (09:30:01 ET)
    schedule.every().day.at("09:30:01", tz="America/New_York").do(
        run_precise_hourly_trading_cycle
    ).tag("precise-trading")

    logger.info(
        "Precise trading schedule installed: hourly at 04:00→20:00 ET (on :00:01) "
        "plus 09:30:01 for the regular open"
    )
def sync_data_lake(ib: IB, tickers: Optional[Sequence[str]] = None) -> None:
    """Synchronise the curated IBKR-backed data lake for ``tickers`` sequentially."""

    if not tickers:
        tickers = load_tickers()

    for ticker in tickers:
        try:
            sync_ticker_data(ib, ticker)
        except Exception as e:
            logger.error(
                f"❌ Failed to sync data lake for {ticker}: {e}\n{traceback.format_exc()}"
            )
# Generate a human readable summary of trade activity and model analyses.
def _calculate_daily_pnl(trades: pd.DataFrame) -> pd.Series:
    """Return realised daily PnL assuming one share per buy/sell pair."""
    if trades.empty:
        return pd.Series(dtype=float)
    required_columns = {"timestamp", "ticker", "price", "decision"}
    if not required_columns.issubset(trades.columns):
        return pd.Series(dtype=float)
    working = trades.copy()
    working["decision"] = working["decision"].astype(str).str.upper()
    working = working[working["decision"].isin({"BUY", "SELL"})]
    if working.empty:
        return pd.Series(dtype=float)
    working["price"] = pd.to_numeric(working["price"], errors="coerce")
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working = working.dropna(subset=["price", "timestamp", "ticker"])
    if working.empty:
        return pd.Series(dtype=float)
    working = working.sort_values("timestamp")
    positions = defaultdict(deque)
    pnl_per_day = defaultdict(float)
    for row in working.itertuples(index=False):
        ts = row.timestamp
        ticker = str(row.ticker)
        decision = row.decision
        price = float(row.price)
        if decision == "BUY":
            positions[ticker].append(price)
        elif decision == "SELL":
            entry_price = positions[ticker].popleft() if positions[ticker] else price
            pnl_per_day[ts.date()] += price - entry_price
    if not pnl_per_day:
        return pd.Series(dtype=float)
    pnl_series = pd.Series(pnl_per_day, dtype=float)
    pnl_series.sort_index(inplace=True)
    return pnl_series
def generate_trade_report() -> str:
    """Generate a trade report, append it to disk, and return the text."""
    try:
        df = pd.read_csv(TRADE_LOG_PATH, names=CSV_COLUMNS)
    except Exception as e:
        logger.error(f"❌ Failed to read trade log: {e}\n{traceback.format_exc()}")
        return ""
    if df.empty:
        logger.warning("⚠️ Trade log is empty; report will contain zeros.")
    df = df[df["timestamp"].notna()]
    df = df[df["timestamp"].astype(str).str.strip() != ""]
    df = df[~df["timestamp"].astype(str).str.lower().eq("timestamp")]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    decision_series = df["decision"].astype(str).str.upper()
    df["decision"] = decision_series
    total_trades = len(df)
    put_trades = int((decision_series == "PUT").sum())
    call_trades = int((decision_series == "CALL").sum())
    buy_trades = int((decision_series == "BUY").sum())
    sell_trades = int((decision_series == "SELL").sum())
    fib_trades = int(
        (
            df["fib"].notna()
            & df["fib"].astype(str).str.strip().str.upper().ne("N/A")
        ).sum()
    )
    daily_pnl_series = _calculate_daily_pnl(df[["timestamp", "ticker", "price", "decision"]])
    latest_date = None
    latest_daily_pnl = None
    if not daily_pnl_series.empty:
        latest_date = daily_pnl_series.index.max()
        latest_daily_pnl = float(daily_pnl_series.loc[latest_date])
    cumulative_pnl = float(daily_pnl_series.sum()) if not daily_pnl_series.empty else 0.0
    hk_tz = pytz.timezone("Asia/Hong_Kong")
    generated_at = datetime.now(timezone.utc).astimezone(hk_tz)
    daily_pnl_line = "Daily Profit/Loss: N/A"
    if latest_date is not None and latest_daily_pnl is not None:
        daily_pnl_line = (
            f"Daily Profit/Loss ({latest_date.isoformat()}): ${latest_daily_pnl:.2f}"
        )
    report = (
        f"=== Daily Trade Report ({generated_at.strftime('%Y-%m-%d %H:%M:%S %Z')}) ===\n"
        f"Total Trades: {total_trades}\n"
        f"PUT Trades: {put_trades}\n"
        f"CALL Trades: {call_trades}\n"
        f"Buy Decisions: {buy_trades}\n"
        f"Sell Decisions: {sell_trades}\n"
        f"Trades Near Fibonacci Levels: {fib_trades}\n"
        f"{daily_pnl_line}\n"
        f"Cumulative Profit/Loss: ${cumulative_pnl:.2f}\n"
    )
    try:
        os.makedirs(os.path.dirname(TRADE_REPORT_PATH), exist_ok=True)
        with open(TRADE_REPORT_PATH, "a", encoding="utf-8") as f:
            if f.tell() > 0:
                f.write("\n")
            f.write(report)
        logger.info(f"✅ Appended trade report entry: {TRADE_REPORT_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to write trade report: {e}\n{traceback.format_exc()}")
    return report
def update_historical_dataset() -> bool:
    """Append newly generated rows to ``historical_data.csv`` without rebuilding it."""

    hist_path = HISTORICAL_DATA_FILE
    logger.info("🔄 Historical dataset update triggered.")
    try:
        updated = append_historical_data(hist_path)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error(
            "❌ historical_data.csv update failed: %s\n%s",
            exc,
            traceback.format_exc(),
        )
        return False

    if updated:
        logger.info("✅ historical_data.csv append completed successfully.")
    else:
        logger.warning("⚠️ No new rows were appended to historical_data.csv.")
    return updated
if __name__ == "__main__":
    args = _parse_cli_args()
    hist_file = os.path.join("data", "historical_data.csv")

    def ensure_historical_dataset() -> None:
        if os.path.exists(hist_file):
            return
        logger.info(
            "historical_data.csv not found. Generating from locally cached raw data..."
        )
        if generate_historical_data(hist_file):
            logger.info("historical_data.csv created successfully.")
        else:
            logger.warning("Failed to create historical_data.csv.")

    if args.command == "update-historical":
        ensure_historical_dataset()
        success = update_historical_dataset()
        sys.exit(0 if success else 1)

    if args.command != "close-options":
        ensure_historical_dataset()

    ib = connect_ibkr()
    if ib is None or not ib.isConnected():
        logger.error("❌ Failed to connect to IBKR. Exiting.")
        sys.exit(1)

    if args.command == "close-options":
        try:
            closed = close_all_option_positions(
                ib,
                close_option_positions,
                rights=args.rights,
                only_short=args.only_short,
                require_profit=args.require_profit,
            )
            if closed:
                logger.info("✅ Closed at least one option position across the account.")
            else:
                logger.info("ℹ️ No option positions matched the close criteria.")
        finally:
            if ib.isConnected():
                ib.disconnect()
        sys.exit(0)

    defer_initial_sync = getattr(args, "defer_initial_sync", False)
    tickers = get_cached_tickers()

    if defer_initial_sync:
        logger.info(
            "⏭️ Skipping initial data lake seeding (--defer-initial-sync). "
            "Historical refresh will run after the first trading cycle."
        )
    else:
        logger.info(
            "🤖 Trading decisions now run before the first data sync; the post-decision "
            "sync inside each trading cycle will refresh the curated data lake."
        )

    schedule_precise_trading_jobs()
    schedule.every().day.at("04:00").do(generate_trade_report).tag("daily")

    logger.info("✅ Scheduled tasks installed (precise U.S. trading cadence + daily report).")

    if is_us_equity_session_open():
        logger.info(
            "🟢 U.S. equity session active at startup – running an immediate trading cycle before the scheduled cadence."
        )
        try:
            process_tickers(ib, tickers)
        except Exception as e:
            logger.error(
                f"❌ Startup trading cycle failed: {e}\n{traceback.format_exc()}"
            )
    else:
        logger.info(
            "ℹ️ Startup began while the U.S. equity session is closed; will await the first scheduled trading window."
        )

    try:
        subprocess.run(["python", "/app/self_learn.py"], check=True)
    except Exception as e:
        logger.error(f"❌ Initial model training failed: {e}\n{traceback.format_exc()}")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)

            # Reconnect IBKR if disconnected
            if ib and not ib.isConnected():
                logger.warning("⚠️ IBKR connection lost. Attempting to reconnect...")
                ib = connect_ibkr() or ib
                if ib and ib.isConnected():
                    logger.info("✅ Reconnected to IBKR successfully.")
                    reconnect_attempts = 0
                else:
                    reconnect_attempts += 1
                    backoff = min(
                        60 * (2**reconnect_attempts) + random.randint(0, 60),
                        max_backoff,
                    )
                    logger.error(
                        f"❌ Failed to reconnect (attempt {reconnect_attempts}). Waiting {backoff}s before retry.\n{traceback.format_exc()}"
                    )
                    time.sleep(backoff) # Sleep extra before next loop iteration
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt, disconnecting IBKR and exiting...")
        if ib and ib.isConnected():
            ib.disconnect()
        if os.path.exists(pid_file):
            os.remove(pid_file)
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Unexpected error in main loop: {e}\n{traceback.format_exc()}")
        if ib and ib.isConnected():
            ib.disconnect()
        if os.path.exists(pid_file):
            os.remove(pid_file)
        sys.exit(1)
