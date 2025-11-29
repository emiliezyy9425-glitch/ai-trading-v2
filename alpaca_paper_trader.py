import os
import time
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca_utils import get_alpaca_credentials

from tsla_ai_master_final_ready_backtest import (
    get_historical_data_backtest as get_historical_data,
    get_multi_timeframe_indicators_backtest,
    get_stock_price_backtest as get_stock_price,
    get_iv_and_delta_backtest as get_iv_and_delta,
    count_today_buy_trades,
    count_today_sell_trades,
    write_trade_csv,
    is_market_open_at,
    TIMEFRAMES,
    MAX_DAILY_BUY_TRADES,
    MAX_DAILY_SELL_TRADES,
)
from indicators import calculate_level_weight
from feature_engineering import (
    count_fib_timezones,
    derive_fibonacci_features,
    encode_td9,
    encode_vol_cat,
    encode_zig,
)
from sp500_above_20d import get_sp500_above_20d
from ml_predictor import (
    predict_with_all_models,
    ensemble_vote,
    MODEL_NAMES,
)
from self_learn import FEATURE_NAMES

logger = logging.getLogger("alpaca_paper_trader")
logging.basicConfig(level=logging.INFO)

# Hong Kong tickers (identified by numeric symbols) use fixed lot sizes


def _format_model_decisions(detail: dict | None) -> dict[str, str]:
    votes = {}
    if isinstance(detail, dict):
        votes = detail.get("votes", {}) if isinstance(detail.get("votes", {}), dict) else {}

    formatted: dict[str, str] = {}
    for name in MODEL_NAMES:
        column = f"{name}_decision"
        vote = votes.get(name)
        formatted[column] = str(vote).upper() if vote else "MISSING"
    return formatted


def place_stock_trade_alpaca(
    client: TradingClient,
    action: str,
    ticker: str,
    qty: int = 1,
    *,
    limit_price: float | None = None,
    extended_hours: bool = False,
) -> bool:
    """Place a limit order using Alpaca's paper trading API."""
    side = OrderSide.BUY if action.upper() == "BUY" else OrderSide.SELL
    if limit_price is None:
        logger.error("Limit price required for limit order; aborting trade.")
        return False
    try:
        client.submit_order(
            order_data=LimitOrderRequest(
                symbol=ticker,
                qty=qty,
                side=side,
                limit_price=limit_price,
                time_in_force=TimeInForce.DAY,
                extended_hours=extended_hours,
            )
        )
        return True
    except Exception as e:  # pragma: no cover - network interaction
        logger.error(f"❌ Failed to execute {action} order for {ticker}: {e}")
        return False
def _get_latest_quote(data_client: StockHistoricalDataClient, ticker: str) -> tuple[float | None, float | None]:
    """Fetch the latest ask and bid prices for a ticker."""
    try:
        request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        response = data_client.get_stock_latest_quote(request)
        if isinstance(response, dict):
            quote = response.get(ticker)
        else:
            quote = response
        if quote is None:
            return None, None
        ask_price = getattr(quote, "ask_price", None)
        bid_price = getattr(quote, "bid_price", None)
        return ask_price, bid_price
    except Exception as exc:  # pragma: no cover - network interaction
        logger.error(f"Failed to fetch latest quote for {ticker}: {exc}")
        return None, None


def _extract_features(
    indicators: dict,
    price: float,
    iv: float,
    delta: float,
    sp500_pct: float,
) -> pd.Series:
    """Build the feature vector matching the backtest logic."""
    # --- extract per-timeframe indicators (mirrors backtest) ---
    # 1 hour
    rsi_1h = indicators['rsi'].get('1 hour', 0.0)
    macd_1h = indicators['macd'].get('1 hour', 0.0)
    signal_1h = indicators['signal'].get('1 hour', 0.0)
    ema10_1h = indicators['ema10'].get('1 hour', 0.0)
    ema10_dev_1h = indicators['ema10_dev'].get('1 hour', 0.0)
    rsi_change_1h = indicators.get('rsi_change', {}).get('1 hour', 0.0)
    macd_change_1h = indicators.get('macd_change', {}).get('1 hour', 0.0)
    ema10_change_1h = indicators.get('ema10_change', {}).get('1 hour', 0.0)
    price_above_ema10_1h = indicators['price_above_ema10'].get('1 hour', False)
    bollinger_1h = indicators['bollinger'].get('1 hour', {'upper': 0.0, 'lower': 0.0, 'mid': 0.0})
    volume_1h = indicators['volume'].get('1 hour', 0)
    td9_summary_1h = indicators['td9_summary'].get('1 hour', "N/A")
    tds_trend_1h = indicators['tds_trend'].get('1 hour', 0)
    tds_signal_1h = indicators['tds_signal'].get('1 hour', 0)
    fib_summary_1h = indicators['fib_summary'].get('1 hour', "N/A")
    fib_time_zones_1h = indicators['fib_time_zones'].get('1 hour', [])
    zig_zag_trend_1h = indicators['zig_zag_trend'].get('1 hour', "N/A")
    high_vol_1h = indicators['high_vol'].get('1 hour', False)
    vol_spike_1h = indicators['vol_spike'].get('1 hour', False)
    vol_category_1h = indicators['vol_category'].get('1 hour', "Neutral Volume")

    # 4 hours
    rsi_4h = indicators['rsi'].get('4 hours', 0.0)
    macd_4h = indicators['macd'].get('4 hours', 0.0)
    signal_4h = indicators['signal'].get('4 hours', 0.0)
    ema10_4h = indicators['ema10'].get('4 hours', 0.0)
    ema10_dev_4h = indicators['ema10_dev'].get('4 hours', 0.0)
    rsi_change_4h = indicators.get('rsi_change', {}).get('4 hours', 0.0)
    macd_change_4h = indicators.get('macd_change', {}).get('4 hours', 0.0)
    ema10_change_4h = indicators.get('ema10_change', {}).get('4 hours', 0.0)
    price_above_ema10_4h = indicators['price_above_ema10'].get('4 hours', False)
    bollinger_4h = indicators['bollinger'].get('4 hours', {'upper': 0.0, 'lower': 0.0, 'mid': 0.0})
    volume_4h = indicators['volume'].get('4 hours', 0)
    td9_summary_4h = indicators['td9_summary'].get('4 hours', "N/A")
    tds_trend_4h = indicators['tds_trend'].get('4 hours', 0)
    tds_signal_4h = indicators['tds_signal'].get('4 hours', 0)
    fib_summary_4h = indicators['fib_summary'].get('4 hours', "N/A")
    fib_time_zones_4h = indicators['fib_time_zones'].get('4 hours', [])
    zig_zag_trend_4h = indicators['zig_zag_trend'].get('4 hours', "N/A")
    high_vol_4h = indicators['high_vol'].get('4 hours', False)
    vol_spike_4h = indicators['vol_spike'].get('4 hours', False)
    vol_category_4h = indicators['vol_category'].get('4 hours', "Neutral Volume")

    # 1 day
    rsi_1d = indicators['rsi'].get('1 day', 0.0)
    macd_1d = indicators['macd'].get('1 day', 0.0)
    signal_1d = indicators['signal'].get('1 day', 0.0)
    ema10_1d = indicators['ema10'].get('1 day', 0.0)
    ema10_dev_1d = indicators['ema10_dev'].get('1 day', 0.0)
    rsi_change_1d = indicators.get('rsi_change', {}).get('1 day', 0.0)
    macd_change_1d = indicators.get('macd_change', {}).get('1 day', 0.0)
    ema10_change_1d = indicators.get('ema10_change', {}).get('1 day', 0.0)
    price_above_ema10_1d = indicators['price_above_ema10'].get('1 day', False)
    bollinger_1d = indicators['bollinger'].get('1 day', {'upper': 0.0, 'lower': 0.0, 'mid': 0.0})
    volume_1d = indicators['volume'].get('1 day', 0)
    td9_summary_1d = indicators['td9_summary'].get('1 day', "N/A")
    tds_trend_1d = indicators['tds_trend'].get('1 day', 0)
    tds_signal_1d = indicators['tds_signal'].get('1 day', 0)
    fib_summary_1d = indicators['fib_summary'].get('1 day', "N/A")
    fib_time_zones_1d = indicators['fib_time_zones'].get('1 day', [])
    zig_zag_trend_1d = indicators['zig_zag_trend'].get('1 day', "N/A")
    high_vol_1d = indicators['high_vol'].get('1 day', False)
    vol_spike_1d = indicators['vol_spike'].get('1 day', False)
    vol_category_1d = indicators['vol_category'].get('1 day', "Neutral Volume")

    td9_1h = encode_td9(td9_summary_1h)
    td9_4h = encode_td9(td9_summary_4h)
    td9_1d = encode_td9(td9_summary_1d)
    zig_1h = encode_zig(zig_zag_trend_1h)
    zig_4h = encode_zig(zig_zag_trend_4h)
    zig_1d = encode_zig(zig_zag_trend_1d)
    vol_cat_1h = encode_vol_cat(vol_category_1h)
    vol_cat_4h = encode_vol_cat(vol_category_4h)
    vol_cat_1d = encode_vol_cat(vol_category_1d)

    fib_levels_1h, fib_zone_delta_1h = derive_fibonacci_features(fib_summary_1h, price)
    fib_levels_4h, fib_zone_delta_4h = derive_fibonacci_features(fib_summary_4h, price)
    fib_levels_1d, fib_zone_delta_1d = derive_fibonacci_features(fib_summary_1d, price)

    fib_prices_1h = [lvl for lvl in fib_levels_1h if lvl]
    fib_prices_4h = [lvl for lvl in fib_levels_4h if lvl]
    fib_prices_1d = [lvl for lvl in fib_levels_1d if lvl]

    pivot_prices_all = []
    for tf_name in ['1 hour', '4 hours', '1 day']:
        pivot_dict = indicators.get('pivot_points', {}).get(tf_name, {})
        pivot_prices_all.extend(list(pivot_dict.values()))
    all_fib_prices = fib_prices_1h + fib_prices_4h + fib_prices_1d
    level_weight = calculate_level_weight(price, all_fib_prices, pivot_prices_all)

    fib_time_count_1h = count_fib_timezones(fib_time_zones_1h)
    fib_time_count_4h = count_fib_timezones(fib_time_zones_4h)
    fib_time_count_1d = count_fib_timezones(fib_time_zones_1d)

    adx_1h = indicators.get('adx', {}).get('1 hour', 0.0)
    obv_1h = indicators.get('obv', {}).get('1 hour', 0.0)
    stochastic_1h = indicators.get('stochastic', {}).get('1 hour', {'k': 0.0, 'd': 0.0})
    stoch_k_1h = stochastic_1h.get('k', 0.0)
    stoch_d_1h = stochastic_1h.get('d', 0.0)

    adx_4h = indicators.get('adx', {}).get('4 hours', 0.0)
    obv_4h = indicators.get('obv', {}).get('4 hours', 0.0)
    stochastic_4h = indicators.get('stochastic', {}).get('4 hours', {'k': 0.0, 'd': 0.0})
    stoch_k_4h = stochastic_4h.get('k', 0.0)
    stoch_d_4h = stochastic_4h.get('d', 0.0)

    adx_1d = indicators.get('adx', {}).get('1 day', 0.0)
    obv_1d = indicators.get('obv', {}).get('1 day', 0.0)
    stochastic_1d = indicators.get('stochastic', {}).get('1 day', {'k': 0.0, 'd': 0.0})
    stoch_k_1d = stochastic_1d.get('k', 0.0)
    stoch_d_1d = stochastic_1d.get('d', 0.0)

    features = pd.Series(
        {
            'rsi_1h': rsi_1h,
            'macd_1h': macd_1h,
            'signal_1h': signal_1h,
            'ema10_1h': ema10_1h,
            'ema10_dev_1h': ema10_dev_1h,
            'rsi_change_1h': rsi_change_1h,
            'macd_change_1h': macd_change_1h,
            'ema10_change_1h': ema10_change_1h,
            'price_above_ema10_1h': int(price_above_ema10_1h),
            'bb_upper_1h': bollinger_1h['upper'],
            'bb_lower_1h': bollinger_1h['lower'],
            'bb_mid_1h': bollinger_1h['mid'],
            'volume_1h': volume_1h,
            'tds_trend_1h': tds_trend_1h,
            'tds_signal_1h': tds_signal_1h,
            'high_vol_1h': int(high_vol_1h),
            'vol_spike_1h': int(vol_spike_1h),
            'td9_1h': td9_1h,
            'zig_1h': zig_1h,
            'vol_cat_1h': vol_cat_1h,
            'fib_time_count_1h': fib_time_count_1h,
            'fib_level1_1h': fib_levels_1h[0],
            'fib_level2_1h': fib_levels_1h[1],
            'fib_level3_1h': fib_levels_1h[2],
            'fib_level4_1h': fib_levels_1h[3],
            'fib_level5_1h': fib_levels_1h[4],
            'fib_level6_1h': fib_levels_1h[5],
            'fib_zone_delta_1h': fib_zone_delta_1h,
            'atr_1h': indicators['atr'].get('1 hour', 0.0),
            'adx_1h': adx_1h,
            'obv_1h': obv_1h,
            'stoch_k_1h': stoch_k_1h,
            'stoch_d_1h': stoch_d_1h,

            'rsi_4h': rsi_4h,
            'macd_4h': macd_4h,
            'signal_4h': signal_4h,
            'ema10_4h': ema10_4h,
            'ema10_dev_4h': ema10_dev_4h,
            'rsi_change_4h': rsi_change_4h,
            'macd_change_4h': macd_change_4h,
            'ema10_change_4h': ema10_change_4h,
            'price_above_ema10_4h': int(price_above_ema10_4h),
            'bb_upper_4h': bollinger_4h['upper'],
            'bb_lower_4h': bollinger_4h['lower'],
            'bb_mid_4h': bollinger_4h['mid'],
            'volume_4h': volume_4h,
            'tds_trend_4h': tds_trend_4h,
            'tds_signal_4h': tds_signal_4h,
            'high_vol_4h': int(high_vol_4h),
            'vol_spike_4h': int(vol_spike_4h),
            'td9_4h': td9_4h,
            'zig_4h': zig_4h,
            'vol_cat_4h': vol_cat_4h,
            'fib_time_count_4h': fib_time_count_4h,
            'fib_level1_4h': fib_levels_4h[0],
            'fib_level2_4h': fib_levels_4h[1],
            'fib_level3_4h': fib_levels_4h[2],
            'fib_level4_4h': fib_levels_4h[3],
            'fib_level5_4h': fib_levels_4h[4],
            'fib_level6_4h': fib_levels_4h[5],
            'fib_zone_delta_4h': fib_zone_delta_4h,
            'atr_4h': indicators['atr'].get('4 hours', 0.0),
            'adx_4h': adx_4h,
            'obv_4h': obv_4h,
            'stoch_k_4h': stoch_k_4h,
            'stoch_d_4h': stoch_d_4h,

            'rsi_1d': rsi_1d,
            'macd_1d': macd_1d,
            'signal_1d': signal_1d,
            'ema10_1d': ema10_1d,
            'ema10_dev_1d': ema10_dev_1d,
            'rsi_change_1d': rsi_change_1d,
            'macd_change_1d': macd_change_1d,
            'ema10_change_1d': ema10_change_1d,
            'price_above_ema10_1d': int(price_above_ema10_1d),
            'bb_upper_1d': bollinger_1d['upper'],
            'bb_lower_1d': bollinger_1d['lower'],
            'bb_mid_1d': bollinger_1d['mid'],
            'volume_1d': volume_1d,
            'tds_trend_1d': tds_trend_1d,
            'tds_signal_1d': tds_signal_1d,
            'high_vol_1d': int(high_vol_1d),
            'vol_spike_1d': int(vol_spike_1d),
            'td9_1d': td9_1d,
            'zig_1d': zig_1d,
            'vol_cat_1d': vol_cat_1d,
            'fib_time_count_1d': fib_time_count_1d,
            'fib_level1_1d': fib_levels_1d[0],
            'fib_level2_1d': fib_levels_1d[1],
            'fib_level3_1d': fib_levels_1d[2],
            'fib_level4_1d': fib_levels_1d[3],
            'fib_level5_1d': fib_levels_1d[4],
            'fib_level6_1d': fib_levels_1d[5],
            'fib_zone_delta_1d': fib_zone_delta_1d,
            'atr_1d': indicators['atr'].get('1 day', 0.0),
            'adx_1d': adx_1d,
            'obv_1d': obv_1d,
            'stoch_k_1d': stoch_k_1d,
            'stoch_d_1d': stoch_d_1d,

            'iv': iv,
            'delta': delta,
            'sp500_above_20d': sp500_pct,
            'level_weight': level_weight,
        },
        name='features',
    )
    return features


def run_iteration(
    client: TradingClient,
    data_client: StockHistoricalDataClient,
    ticker: str,
) -> None:
    now = datetime.now(timezone.utc)
    regular_session = is_market_open_at(ticker, now, '1 hour')

    start = now - timedelta(days=30)
    data = {}
    for tf in TIMEFRAMES:
        try:
            df = get_historical_data(ticker, tf, start, now)
        except Exception as e:  # pragma: no cover - network interaction
            logger.error(f"Failed to fetch data for {tf}: {e}")
            df = None
        if df is not None and not df.empty:
            data[tf] = df
    if '1 hour' not in data:
        logger.warning("No 1h data available; skipping.")
        return

    indicators = get_multi_timeframe_indicators_backtest(ticker, now, data)
    current_df = data['1 hour']
    price = get_stock_price(ticker, now, current_df)
    if price is None:
        logger.warning("Price unavailable; skipping.")
        return

    ask_price, bid_price = _get_latest_quote(data_client, ticker)
    if ask_price is None or bid_price is None:
        logger.warning("Quote unavailable; falling back to last price for limit order.")
        ask_price = price if ask_price is None else ask_price
        bid_price = price if bid_price is None else bid_price

    iv, delta = get_iv_and_delta(ticker, now, current_df)
    try:
        sp500_pct, _, _ = get_sp500_above_20d()
    except Exception:  # pragma: no cover - network interaction
        sp500_pct = 0.0

    feature_series = _extract_features(indicators, price, iv, delta, sp500_pct)
    features = feature_series.to_frame().T
    predictions = predict_with_all_models(features)
    ml_raw, vote_detail = ensemble_vote(predictions, return_details=True)
    ml_decision = {"Buy": "BUY", "Sell": "SELL", "Hold": "HOLD"}.get(ml_raw, "HOLD")
    ai_decision = ml_decision
    model_decisions = _format_model_decisions(vote_detail)

    fib_summary = indicators['fib_summary'].get('1 hour', "N/A")
    tds_trend = indicators['tds_trend'].get('1 hour', 0)
    tds_signal = indicators['tds_signal'].get('1 hour', 0)
    td9_summary = indicators['td9_summary'].get('1 hour', "N/A")
    volume_1h = indicators['volume'].get('1 hour', 0)

    high_vol = indicators['high_vol'].get('1 hour', False)
    vol_spike = indicators['vol_spike'].get('1 hour', False)
    if high_vol or vol_spike:
        logger.warning("🚫 High volatility or volume spike detected; skipping trade.")
        decision_log = f"{ai_decision} (skipped)"
        executed = False
    else:
        buy_trades = count_today_buy_trades(ticker, now)
        sell_trades = count_today_sell_trades(ticker, now)
        executed = False
        decision = ai_decision
        try:
            account = client.get_account()
            equity = float(getattr(account, "equity", 0.0))
            cash = float(getattr(account, "cash", 0.0))
        except Exception as e:  # pragma: no cover - network interaction
            logger.error(f"Failed to fetch account equity: {e}")
            equity = 0.0
            cash = 0.0

        # Quantity corresponding to roughly 0.1% of account equity

        if ticker.isdigit():
            qty_equity = 100
        else:
            qty_equity = int((equity * 0.001) // price) if price > 0 else 0


        if decision == "BUY" and buy_trades < MAX_DAILY_BUY_TRADES:
            if cash <= 0:
                logger.info("Cash balance is non-positive; skipping buy.")
            elif qty_equity > 0:
                executed = place_stock_trade_alpaca(
                    client,
                    "BUY",
                    ticker,
                    qty_equity,
                    limit_price=float(ask_price),
                    extended_hours=not regular_session,
                )
            else:
                logger.info("Calculated quantity 0; skipping buy.")
        elif decision == "SELL" and sell_trades < MAX_DAILY_SELL_TRADES:
            try:
                position = client.get_open_position(ticker)
                pos_qty = int(float(position.qty))
                avg_cost = float(position.avg_entry_price)
            except Exception:
                position = None
                pos_qty = 0
                avg_cost = 0.0

            if position and price > avg_cost:
                qty_to_sell = min(pos_qty, qty_equity)
                if qty_to_sell > 0:
                    executed = place_stock_trade_alpaca(
                        client,
                        "SELL",
                        ticker,
                        qty_to_sell,
                        limit_price=float(bid_price),
                        extended_hours=not regular_session,
                    )
                else:
                    logger.info("Calculated quantity 0; skipping sell.")
            else:
                logger.info("No profitable position to sell; skipping.")
        else:
            logger.info(f"⏳ No trade executed. Decision: {decision}")
        decision_log = decision

    write_trade_csv(
        {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "price": price,
            "ai_decision": ai_decision,
            "ml_decision": ml_decision,
            "decision": decision_log,
            "executed": "Yes" if executed else "No",
            "fib": fib_summary,
            "tds": f"{tds_trend}/{tds_signal}",
            "td9": td9_summary,
            "RSI": indicators['rsi'].get('1 hour', 0.0),
            "MACD": indicators['macd'].get('1 hour', 0.0),
            "Signal": indicators['signal'].get('1 hour', 0.0),
            "Volume": volume_1h,
            "IV": iv,
            "Delta": delta,
            "Source": "ML",
            **model_decisions,
        }
    )
    if executed:
        logger.info(f"✅ Executed {ai_decision} trade for {ticker} at {price}")


def run_paper_trading(ticker: str = "TSLA") -> None:
    creds = get_alpaca_credentials()
    if not creds:
        raise RuntimeError(
            "Alpaca API credentials not configured. Set ALPACA_API_KEY/ALPACA_SECRET_KEY or "
            "APCA_API_KEY_ID/APCA_API_SECRET_KEY."
        )

    client = TradingClient(
        creds.api_key,
        creds.secret_key,
        paper=True,
    )
    data_client = StockHistoricalDataClient(
        creds.api_key,
        creds.secret_key,
    )
    while True:  # pragma: no cover - long running loop
        run_iteration(client, data_client, ticker)
        time.sleep(3600)


if __name__ == "__main__":  # pragma: no cover - manual execution
    run_paper_trading()
