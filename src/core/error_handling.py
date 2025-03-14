from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import MetaTrader5 as mt5


class MT5ConnectionManager:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True
    )
    def connect_mt5(self, secrets):
        if not mt5.initialize(**secrets):
            error = mt5.last_error()
            mt5.shutdown()
            raise ConnectionError(f"MT5 connection failed: {error}")
        return True
