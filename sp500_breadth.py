"""Utilities for computing S&P 500 breadth using IBKR data only."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional, Sequence

import pandas as pd
from ib_insync import IB, Stock

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

_SP500_TICKERS_RAW = """
A, AAPL, ABBV, ABNB, ABT, ACGL, ACN, ADBE, ADI, ADM, ADP, ADSK, AEE, AEP, AES, AFL, AIG, AIZ,
AJG, AKAM, ALB, ALGN, ALL, ALLE, AMAT, AMD, AME, AMGN, AMP, AMT, AMZN, ANET, ANSS, AON, AOS, APA,
APD, APH, APTV, ARE, ATO, AVB, AVGO, AVY, AWK, AXON, AXP, AZO, BAC, BALL, BAX, BBWI, BBY, BDX,
BEN, BF.B, BG, BIIB, BIO, BK, BKNG, BKR, BLDR, BLK, BMY, BR, BRK.B, BRO, BSX, BWA, BX, BXP, C,
CAG, CAH, CARR, CAT, CB, CBOE, CBRE, CCI, CCL, CDNS, CDW, CE, CEG, CF, CFG, CHD, CHRW, CHTR, CI,
CINF, CL, CLX, CMA, CMCSA, CME, CMG, CMI, CMS, CNC, CNP, COF, COO, COP, COR, COST, CPAY, CPB,
CPG, CPRT, CPT, CRWD, CSCO, CSGP, CSX, CTAS, CTLT, CTRA, CTSH, CTVA, CVX, CZR, D, DAL, DAY, DE,
DECK, DFS, DG, DGX, DHI, DHR, DLR, DLTR, DOC, DOV, DOW, DPZ, DRI, DTE, DUK, DVA, DVN, DXCM, EA,
EBAY, ECL, ED, EFX, EG, EIX, EL, ELV, EMN, EMR, ENPH, EOG, EPAM, EQIX, EQR, ES, ESS, ETN, ETR,
EVRG, EW, EXC, EXPD, EXPE, EXR, F, FANG, FAST, FCX, FDS, FDX, FE, FFIV, FI, FICO, FIS, FITB,
FMC, FOX, FOXA, FRT, FSLR, FTNT, FTV, GD, GE, GEHC, GEV, GILD, GIS, GL, GLW, GM, GNRC, GOOG,
GOOGL, GPC, GPN, GRMN, GS, GWW, HAL, HAS, HBAN, HCA, HD, HES, HIG, HII, HLT, HOLX, HON, HPE,
HPQ, HSY, HUBB, HUM, HWM, IBM, ICE, IDXX, IEX, IFF, ILMN, INCY, INTC, INTU, INVH, IP, IPG, IQV,
IR, IRM, ISRG, IT, ITW, IVZ, J, JBHT, JBL, JCI, JKHY, JNJ, JPM, K, KDP, KEY, KEYS, KHC, KIM,
KLAC, KMB, KMI, KMX, KO, KR, L, LDOS, LEN, LH, LHX, LIN, LKQ, LLY, LMT, LNT, LOW, LRCX, LULU,
LUV, LVS, LW, LYB, LYV, MA, MAA, MAR, MAS, MCD, MCHP, MCK, MCO, MDT, MDLZ, MET, META, MGM, MHK,
MKC, MKTX, MLM, MMC, MMM, MNST, MO, MOH, MOS, MPC, MPWR, MRK, MRNA, MS, MSCI, MSFT, MSI, MTB,
MU, NCLH, NDAQ, NDSN, NE, NEE, NEM, NFLX, NI, NKE, NOC, NOW, NRG, NSC, NTAP, NTRS, NVDA, NVDA,
NVR, NWSA, NWS, NXPI, O, ODFL, OGN, OKE, OMC, ON, ORCL, ORLY, OTIS, OXY, PANW, PARA, PAYC,
PAYX, PCAR, PCG, PEG, PEP, PFE, PG, PGR, PH, PHM, PKG, PLD, PM, PNC, PNR, PODD, POOL, PPG, PPL,
PRU, PSA, PSX, PTC, PTON, PWR, PYPL, QCOM, QRVO, RCL, REG, REGN, RF, RJF, RL, RMD, ROK, ROL,
ROP, ROST, RSG, RTX, RVTY, SBAC, SBUX, SCHW, SEDG, SHW, SJM, SLB, SMCI, SNA, SNPS, SO, SOLV,
SPG, SPGI, SRE, STE, STLD, STT, STX, STZ, SWK, SWKS, SYF, SYK, SYY, T, TAP, TDG, TDY, TECH, TEL,
TER, TESLA, TFC, TFX, TGT, TJX, TMO, TMUS, TPR, TRGP, TRMB, TROW, TRV, TSCO, TSLA, TSN, TT,
TTWO, TXN, TXT, TYL, UAL, UBER, UDR, UHS, ULTA, UNH, UNP, UPS, URI, USB, V, VFC, VICI, VLO,
VLTO, VRSK, VRSN, VRTX, VTR, VTRS, VXUS, WAB, WAL, WAT, WBA, WBD, WDC, WEC, WELL, WEN, WFC,
WM, WMB, WMT, WRB, WST, WTW, WY, WYNN, XEL, XOM, XYL, YUM, ZBH, ZBRA, ZTS
"""

SP500_TICKERS: Sequence[str] = tuple(
    ticker.strip() for ticker in _SP500_TICKERS_RAW.split(",") if ticker.strip()
)

DOT_TICKER_MAP = {
    "BRK.B": "BRK B",
    "BF.B": "BF B",
    "CPG": "CIGI",
    "FI": "FISV",
    "PARA": "PARAA",
    "TESLA": "TSLA",
    "ANSS": "ANSS",
    "CTLT": "CTLT",
    "DFS": "DFS",
    "HES": "HES",
    "WBA": "WBA",
}

_cached_s5tw: Optional[float] = None
_cache_time: Optional[datetime] = None
_CACHE_TTL_SECONDS = 1800
_MAX_CONCURRENT_REQUESTS = 25


async def _fetch_bars(
    ib: IB, contract: Stock, semaphore: asyncio.Semaphore
) -> tuple[Stock, Sequence]:
    async with semaphore:
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr="30 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
    return contract, bars


async def calculate_s5tw_ibkr(ib: IB) -> float:
    """Return the % of S&P 500 stocks trading above their 20-day SMA."""

    global _cached_s5tw, _cache_time

    now = datetime.now()
    if _cache_time and (now - _cache_time).total_seconds() < _CACHE_TTL_SECONDS:
        logger.debug("Using cached S5TW breadth: %.2f%%", _cached_s5tw or 50.0)
        return _cached_s5tw if _cached_s5tw is not None else 50.0

    if ib is None or not ib.isConnected():
        raise RuntimeError("IBKR connection required for breadth calculation")

    logger.info("Calculating live S5TW from IBKR (%d stocks)...", len(SP500_TICKERS))

    normalized_symbols = []
    for ticker in SP500_TICKERS:
        if ticker in DOT_TICKER_MAP:
            normalized_symbols.append(DOT_TICKER_MAP[ticker])
        elif "." in ticker:
            normalized_symbols.append(ticker.replace(".", " "))
        else:
            normalized_symbols.append(ticker)

    contracts = [Stock(symbol, "SMART", "USD") for symbol in normalized_symbols]
    await ib.qualifyContractsAsync(*contracts)

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)
    tasks = [_fetch_bars(ib, contract, semaphore) for contract in contracts]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    above_count = 0
    valid_count = 0

    for result in results:
        if isinstance(result, Exception):
            logger.debug("Skipping S5TW contract due to error: %s", result)
            continue

        contract, bars = result
        if not bars or len(bars) < 21:
            continue

        df = pd.DataFrame(b.__dict__ for b in bars)
        if df.empty or "close" not in df.columns:
            continue

        sma20 = df["close"].rolling(20).mean().iloc[-1]
        current = df["close"].iloc[-1]

        if pd.isna(sma20) or pd.isna(current):
            continue

        if current > sma20:
            above_count += 1
        valid_count += 1

    if valid_count == 0:
        raise RuntimeError("No valid breadth data returned from IBKR")

    percentage = round((above_count / valid_count) * 100, 2)
    _cached_s5tw = percentage
    _cache_time = now

    logger.info(
        "S5TW calculated: %d/%d stocks above 20d SMA → %.2f%%",
        above_count,
        valid_count,
        percentage,
    )
    return percentage


def get_sp500_above_20d(ib: Optional[IB] = None) -> tuple[float, None, None]:
    """Maintain backward compatibility with legacy callers."""

    if ib is None:
        logger.warning("IBKR connection missing for S5TW lookup; returning neutral 50%%")
        return 50.0, None, None

    try:
        value = ib.run(calculate_s5tw_ibkr(ib))
        return value, None, None
    except Exception as exc:  # pragma: no cover - network/IBKR errors
        logger.error("S5TW calculation failed via IBKR: %s", exc)
        return 50.0, None, None
