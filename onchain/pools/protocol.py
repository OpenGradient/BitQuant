from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
import time
import threading

from api.api_types import Pool, PoolQuery, PoolType
from onchain.tokens.metadata import TokenMetadataRepo


class Protocol(ABC):
    @abstractmethod
    def get_pools(self, token_metadata_repo: TokenMetadataRepo) -> List[Pool]:
        """Return all pools supported by the protocol."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the protocol name."""
        pass


class ProtocolRegistry:
    token_metadata_repo: TokenMetadataRepo
    protocols: Dict[str, Protocol] = {}
    pools_cache: Dict[str, List[Pool]] = {}
    last_refresh: Dict[str, float] = {}
    refresh_interval = 10 * 60  # Refresh every 10 mins
    _initialized = False

    def __init__(self, token_metadata_repo: TokenMetadataRepo):
        self.logger = logging.getLogger("ProtocolRegistry")
        self.token_metadata_repo = token_metadata_repo
        self._refresh_thread = None
        self._stop_event = threading.Event()

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

    def refresh_pools(self, protocol_name: Optional[str] = None) -> None:
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
                pools = protocol.get_pools(self.token_metadata_repo)
                self.pools_cache[name] = pools
                self.last_refresh[name] = time.time()
                self.logger.info(f"Refreshed {len(pools)} pools for {name}")
            except Exception as e:
                self.logger.error(f"Error refreshing pools for {name}: {str(e)}")

    def ensure_fresh_data(self) -> None:
        """Check if data needs refreshing and refresh if necessary"""
        current_time = time.time()
        protocols_to_check = self.protocols

        for name in protocols_to_check:
            last_refresh_time = self.last_refresh.get(name, 0)
            if (
                current_time - last_refresh_time > self.refresh_interval
                or name not in self.pools_cache
            ):
                self.refresh_pools(name)

    def get_pools(self, query: PoolQuery) -> List[Pool]:
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

    def get_all_protocols(self) -> List[str]:
        """Get list of all registered protocol names"""
        return list(self.protocols.keys())

    def _background_refresh(self) -> None:
        """Background thread to refresh pool data periodically"""
        self.logger.info("Background refresh thread started")

        # Do initial refresh
        try:
            self.refresh_pools()
        except Exception as e:
            self.logger.error(f"Error in initial pool refresh: {str(e)}")

        # Periodic refresh loop
        while not self._stop_event.is_set():
            # Sleep for the refresh interval, but check for stop event periodically
            for _ in range(
                36
            ):  # Check every 100 seconds (36 * 100 = 3600 seconds = 1 hour)
                if self._stop_event.is_set():
                    break
                time.sleep(100)

            if self._stop_event.is_set():
                break

            # Refresh pools
            self.logger.info("Starting scheduled pool refresh")
            try:
                self.refresh_pools()
                self.logger.info("Scheduled pool refresh completed successfully")
            except Exception as e:
                self.logger.error(f"Error in scheduled pool refresh: {str(e)}")

        self.logger.info("Background refresh thread stopped")

    def initialize(self) -> None:
        """Initialize the registry with default protocols and start refresh thread"""
        if self._initialized:
            return

        # Perform initial refresh if not already done
        if not self.pools_cache:
            self.refresh_pools()

        # Start background refresh thread if not already running
        if self._refresh_thread is None or not self._refresh_thread.is_alive():
            self._stop_event.clear()
            self._refresh_thread = threading.Thread(
                target=self._background_refresh,
                daemon=True,
                name="ProtocolRegistryThread",
            )
            self._refresh_thread.start()
            self.logger.info("Started background refresh thread")

        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the registry and stop background thread"""
        if self._refresh_thread and self._refresh_thread.is_alive():
            self.logger.info("Stopping background refresh thread...")
            self._stop_event.set()
            self._refresh_thread.join(timeout=10)  # Wait up to 10 seconds
            if self._refresh_thread.is_alive():
                self.logger.warning("Background refresh thread did not stop gracefully")
            else:
                self.logger.info("Background refresh thread stopped successfully")
        self._initialized = False

    def get_pools_by_ids(self, pool_ids: List[str]) -> List[Pool]:
        """Get full pool objects by their IDs."""
        # Collect all pools from all protocols
        all_pools: List[Pool] = []
        for pools in self.pools_cache.values():
            all_pools.extend(pools)

        # Return pools that match the requested IDs
        return [pool for pool in all_pools if pool.id in pool_ids]
