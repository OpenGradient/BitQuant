from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import asyncio
import logging
import time

from api.api_types import Pool, PoolQuery


class Protocol(ABC):

    @abstractmethod
    def get_pools(self) -> List[Pool]:
        """Return all pools supported by the protocol."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the protocol name."""
        pass


class ProtocolRegistry:

    def __init__(self):
        if self._initialized:
            return

        self.protocols: Dict[str, Protocol] = {}
        self.pools_cache: Dict[str, List[Pool]] = {}
        self.last_refresh: Dict[str, float] = {}
        self.refresh_interval = 60 * 60  # 1 hour in seconds

        self.logger = logging.getLogger("ProtocolRegistry")
        self._initialized = True

    def register_protocol(self, protocol: Protocol) -> None:
        """
        Register a new protocol.
        """
        protocol_name = protocol.name
        if protocol_name in self.protocols:
            self.logger.warning(
                f"Protocol {protocol_name} already registered. Overwriting."
            )

        self.protocols[protocol_name] = protocol
        self.logger.info(f"Registered protocol: {protocol_name}")

    async def refresh_pools(self, protocol_name: Optional[str] = None):
        """
        Refresh pools for a specific protocol or all protocols.
        """
        if protocol_name:
            if protocol_name not in self.protocols:
                self.logger.error(f"Protocol {protocol_name} not found")
                return
            protocols_to_refresh = {protocol_name: self.protocols[protocol_name]}
        else:
            protocols_to_refresh = self.protocols

        for name, protocol in protocols_to_refresh.items():
            try:
                self.logger.info(f"Refreshing pools for {name}")
                pools = await protocol.get_pools()
                self.pools_cache[name] = pools
                self.last_refresh[name] = time.time()
                self.logger.info(f"Refreshed {len(pools)} pools for {name}")
            except Exception as e:
                self.logger.error(f"Error refreshing pools for {name}: {str(e)}")

    async def ensure_fresh_data(self, protocol_name: Optional[str] = None):
        """Check if data needs refreshing and refresh if necessary"""
        current_time = time.time()

        if protocol_name:
            if protocol_name not in self.protocols:
                self.logger.error(f"Protocol {protocol_name} not found")
                return

            protocols_to_check = {protocol_name: self.protocols[protocol_name]}
        else:
            protocols_to_check = self.protocols

        refresh_tasks = []

        for name in protocols_to_check:
            last_refresh_time = self.last_refresh.get(name, 0)
            if (
                current_time - last_refresh_time > self.refresh_interval
                or name not in self.pools_cache
            ):
                refresh_tasks.append(self.refresh_pools(name))

        if refresh_tasks:
            await asyncio.gather(*refresh_tasks)

    async def get_pools(self, query: PoolQuery) -> List[Pool]:
        """
        Get pools that match the query criteria
        """
        # Collect all pools from relevant protocols
        all_pools = []
        for protocol_name, pools in self.pools_cache.items():
            if query.protocols and protocol_name not in query.protocols:
                continue
            all_pools.extend(pools)

        result = []
        for pool in all_pools:
            # Chain filter
            if query.chain is not None and pool.chain != query.chain:
                continue

            # Token filter (match any token by address or symbol)
            if query.tokens:
                pool_token_addresses = [t.address.lower() for t in pool.tokens]
                pool_token_symbols = [t.symbol.lower() for t in pool.tokens]

                token_match = False
                for token in query.tokens:
                    token_lower = token.lower()
                    if (
                        token_lower in pool_token_addresses
                        or token_lower in pool_token_symbols
                    ):
                        token_match = True
                        break

                if not token_match:
                    continue

            # Stablecoin filter
            if (
                query.isStableCoin is not None
                and pool.isStableCoin != query.isStableCoin
            ):
                continue

            # Impermanent loss risk filter
            if (
                query.impermanentLossRisk is not None
                and pool.impermanentLossRisk != query.impermanentLossRisk
            ):
                continue

            # If we got here, the pool matches all criteria
            result.append(pool)

        return result

    def get_all_protocols(self) -> List[str]:
        """Get list of all registered protocol names"""
        return list(self.protocols.keys())

    async def _background_refresh(self) -> None:
        """Background task to refresh pool data periodically"""
        while True:
            await asyncio.sleep(self.refresh_interval)
            self.logger.info("Starting scheduled pool refresh")
            try:
                await self.refresh_pools()
                self.logger.info("Scheduled pool refresh completed successfully")
            except Exception as e:
                self.logger.error(f"Error in scheduled pool refresh: {str(e)}")
