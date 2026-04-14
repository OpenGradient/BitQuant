import logging
import asyncio

from web3 import AsyncWeb3
from web3.providers import AsyncHTTPProvider
from cachetools import TTLCache

from server.config import BASE_RPC_URL, OPG_TOKEN_ADDRESS, OPG_HOLDER_THRESHOLD

ERC20_BALANCE_OF_ABI = [
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    }
]


class OPGTokenGate:
    """Checks whether an EVM address holds enough $OPG tokens on Base."""

    def __init__(self):
        self.w3 = AsyncWeb3(AsyncHTTPProvider(BASE_RPC_URL))
        self.contract = self.w3.eth.contract(
            address=AsyncWeb3.to_checksum_address(OPG_TOKEN_ADDRESS),
            abi=ERC20_BALANCE_OF_ABI,
        )
        self.threshold = OPG_HOLDER_THRESHOLD
        self._cache: TTLCache = TTLCache(maxsize=4096, ttl=300)
        self._lock = asyncio.Lock()

    async def is_opg_holder(self, evm_address: str) -> bool:
        """Return True if evm_address holds >= threshold $OPG on Base.

        Fail-closed: returns False on any error so users get default limits.
        """
        try:
            # Validate EVM address
            checksum = AsyncWeb3.to_checksum_address(evm_address)
        except Exception:
            return False

        # Check cache
        cached = self._cache.get(checksum)
        if cached is not None:
            return cached

        async with self._lock:
            # Double-check after acquiring lock
            cached = self._cache.get(checksum)
            if cached is not None:
                return cached

            try:
                balance = await self.contract.functions.balanceOf(checksum).call()
                result = balance >= self.threshold
                self._cache[checksum] = result
                return result
            except Exception as e:
                logging.warning(f"OPG balance check failed for {checksum}: {e}")
                return False
