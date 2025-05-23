from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
import time
import asyncio

from api.api_types import Pool, PoolQuery, PoolType
from onchain.tokens.metadata import TokenMetadataRepo


class Protocol(ABC):
    @abstractmethod
    async def get_pools(self, token_metadata_repo: TokenMetadataRepo) -> List[Pool]:
        """Return all pools supported by the protocol."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the protocol name."""
        pass

    async def close(self):
        """Clean up any resources used by the protocol."""
        pass


class ProtocolRegistry:
    token_metadata_repo: TokenMetadataRepo
    protocols: Dict[str, Protocol] = {}
    pools_cache: Dict[str, List[Pool]] = {}
    last_refresh: Dict[str, float] = {}
    refresh_interval = 10 * 60  # Refresh every 10 mins
    _initialized = False
    _refresh_task: Optional[asyncio.Task] = None

    def __init__(self, token_metadata_repo: TokenMetadataRepo):
        self.logger = logging.getLogger("ProtocolRegistry")
        self.token_metadata_repo = token_metadata_repo

    def register_protocol(self, protocol: Protocol) -> None:
        """
        Register a new protocol and initialize it.
        """
        protocol_name = protocol.name
        if protocol_name in self.protocols:
            self.logger.warning(
                f"Protocol {protocol_name} already registered. Overwriting."
            )

        self.protocols[protocol_name] = protocol
        self.logger.info(f"Registered protocol: {protocol_name}")

    async def refresh_pools(self, protocol_name: Optional[str] = None) -> None:
        """Refresh pools for a specific protocol or all protocols"""
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
                pools = await protocol.get_pools(self.token_metadata_repo)
                self.pools_cache[name] = pools
                self.last_refresh[name] = time.time()
                self.logger.info(f"Refreshed {len(pools)} pools for {name}")
            except Exception as e:
                self.logger.error(f"Error refreshing pools for {name}: {str(e)}")

    async def get_pools(self, query: PoolQuery) -> List[Pool]:
        """
        Get pools that match the query criteria.
        For AMM pools, only return pools where the user has both tokens in the pair.
        """
        # Collect all pools from relevant protocols
        all_pools: List[Pool] = []
        for protocol_name, pools in self.pools_cache.items():
            if query.protocols and protocol_name not in query.protocols:
                continue
            all_pools.extend(pools)

        result = []
        for pool in all_pools:
            # Chain filter
            if query.chain is not None and pool.chain != query.chain:
                continue

            # Token filter (match any token by address OR symbol)
            if query.tokens:
                token_matches = False
                for pool_token in pool.tokens:
                    if (
                        pool_token.symbol in query.tokens
                        or pool_token.address in query.tokens
                    ):
                        token_matches = True
                        break
                if not token_matches:
                    continue

            # For AMM pools, check if user has both tokens
            if pool.type == PoolType.AMM:
                # Get all token addresses in the pool
                pool_token_addresses = {token.address for token in pool.tokens}
                # Get all token addresses from user's holdings
                user_token_addresses = {token.address for token in query.user_tokens}
                # Only include if user has all tokens needed for the pool
                if not pool_token_addresses.issubset(user_token_addresses):
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

    async def _background_refresh(self) -> None:
        """Background task to refresh pool data periodically"""
        self.logger.info("Background refresh task started")

        # Do initial refresh
        try:
            await self.refresh_pools()
        except Exception as e:
            self.logger.error(f"Error in initial pool refresh: {str(e)}")

        # Periodic refresh loop
        while True:
            await asyncio.sleep(self.refresh_interval)

            # Refresh pools
            self.logger.info("Starting scheduled pool refresh")
            try:
                await self.refresh_pools()
                self.logger.info("Scheduled pool refresh completed successfully")
            except Exception as e:
                self.logger.error(f"Error in scheduled pool refresh: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the registry with default protocols and start refresh task"""
        if self._initialized:
            return

        for protocol in self.protocols.values():
            await protocol.initialize()

        # Perform initial refresh if not already done
        if not self.pools_cache:
            await self.refresh_pools()

        # Start background refresh task if not already running
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(self._background_refresh())
            self.logger.info("Started background refresh task")

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the registry and stop background task"""
        if self._refresh_task and not self._refresh_task.done():
            self.logger.info("Stopping background refresh task...")
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                self.logger.info("Background refresh task stopped successfully")
            except Exception as e:
                self.logger.error(f"Error stopping background refresh task: {str(e)}")

        # Close all protocol sessions
        for protocol in self.protocols.values():
            try:
                await protocol.close()
            except Exception as e:
                self.logger.error(f"Error closing protocol session: {str(e)}")

        self._initialized = False

    def get_pools_by_ids(self, pool_ids: List[str]) -> List[Pool]:
        """Get full pool objects by their IDs."""
        # Collect all pools from all protocols
        all_pools: List[Pool] = []
        for pools in self.pools_cache.values():
            all_pools.extend(pools)

        # Return pools that match the requested IDs
        return [pool for pool in all_pools if pool.id in pool_ids]
