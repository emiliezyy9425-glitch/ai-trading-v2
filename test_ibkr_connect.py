from ib_insync import IB
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connect():
    ib = IB()
    ibkr_host = os.getenv("IBKR_HOST", "127.0.0.1")
    ibkr_port = int(os.getenv("IBKR_PORT", "7497"))
    try:
        ib.connect(ibkr_host, ibkr_port, clientId=1, timeout=15)
        if ib.isConnected():
            logger.info("✅ TWS connectivity test successful.")
        else:
            logger.error("❌ Connection failed.")
    except Exception as e:
        logger.error(f"❌ Connection error: {e}")
    finally:
        ib.disconnect()

if __name__ == "__main__":
    test_connect()