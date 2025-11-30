# AI Trading Agent Workflow Diagram

The diagram below shows how the repository's major scripts and modules interact to collect data, engineer features, train models, and run the live trading loop.

```mermaid
flowchart TD
    subgraph Data[Market Data Sources]
        IBKR[[ibkr_utils.py / Interactive Brokers]]
        Alpaca[[alpaca_utils.py / Alpaca]]
        Cache[[tickers_cache.py]]
        SP500[[sp500_breadth.py]]
    end

    subgraph Downloads[Historical Data + Indicators]
        RawBars[[scripts/download_historical_prices.py]]
        Indicators[[scripts/compute_historical_indicators.py]]
        Aggregator[[scripts/generate_historical_data.py]]
        Cleaner[[clean_historical_data.py]]
    end

    subgraph Features[Feature Engineering]
        IndicatorsCore[[indicators.py]]
        Fibs[[fibonacci_utils.py]]
        FeatEng[[feature_engineering.py]]
        Sequence[[sequence_utils.py]]
    end

    subgraph Training[Training Pipeline]
        SelfLearn[[self_learn.py]]
        AutoTrain[[auto_train.py]]
        Retrain[[retrain_models.py]]
        Runner[[scripts/run_training_pipeline.py]]
        Models[(models/*.pkl\nmodels/*.keras\nmodels/*.zip)]
    end

    subgraph Inference[Live Prediction]
        Predictor[[ml_predictor.py]]
        Decision[[ml_predictor.independent_model_decisions]]
    end

    subgraph Trading[Trading + Execution]
        Agent[[tsla_ai_master_final_ready.py]]
        Options[[option_chain_skip.py]]
        Risk[[training_lookback.py\nrule_based_trader.py]]
        Broker[[ib_insync orders]]
        Logger[[generate_trade_log.py\ndata/trade_log.csv]]
    end

    subgraph Automation[Scheduling]
        Cron[[hourly_cronjob.txt\nweekly_cronjob.txt\nai_agent_cron.sh]]
        Launch[[launch_all.sh]]
        Dashboard[[dashboard.py]]
    end

    Data -->|live bars| Agent
    Data -->|SPX breadth| Agent
    Data -->|historical bars| RawBars
    RawBars --> Indicators --> Aggregator --> Cleaner --> Training
    Cleaner -->|historical_data.csv| Training
    Features --> Training
    Training --> Models --> Predictor
    Features --> Predictor
    Predictor --> Decision --> Agent
    Agent --> Options
    Agent --> Risk
    Agent --> Broker --> Logger
    Logger --> SelfLearn
    SelfLearn --> Retrain
    AutoTrain --> Retrain
    Runner --> Retrain
    Cron --> Agent
    Cron --> Runner
    Launch --> Agent
    Automation --> Dashboard
```

## Component Notes

- **Data acquisition:** `ibkr_utils.py` and `alpaca_utils.py` fetch live bars and account data, while `tickers_cache.py` caches watchlists and `sp500_breadth.py` tracks the S&P 500 percentage-above-20-day breadth signal.
- **Historical pipeline:** `scripts/download_historical_prices.py` downloads raw bars; `scripts/compute_historical_indicators.py` computes indicators for each ticker; `scripts/generate_historical_data.py` aggregates the per-ticker files; `clean_historical_data.py` validates and fixes gaps.
- **Feature engineering:** Core indicator math lives in `indicators.py` and `fibonacci_utils.py`. `feature_engineering.py` builds model-ready frames from live bars, while `sequence_utils.py` pads sequences for LSTM/Transformer inputs.
- **Model training:** `self_learn.py` labels trades and prepares `FEATURE_NAMES`; `auto_train.py`, `retrain_models.py`, and `scripts/run_training_pipeline.py` orchestrate training for Random Forest, XGBoost, LightGBM, LSTM, PPO, and Transformer models stored under `models/`.
- **Inference and trading:** `ml_predictor.predict_with_all_models` loads the trained artifacts and `independent_model_decisions` selects the highest-confidence signal. `tsla_ai_master_final_ready.py` executes trades, consults `option_chain_skip.py` for liquid options, applies risk checks (`training_lookback.py`, `rule_based_trader.py`), and logs fills via `generate_trade_log.py`.
- **Automation:** Cron definitions (`hourly_cronjob.txt`, `weekly_cronjob.txt`) and launcher scripts (`launch_all.sh`, `ai_agent_cron.sh`) run the agent on schedule; `dashboard.py` surfaces runtime stats.
