import requests
from typing import Dict, List, Optional, Union, Any


class DefiLlama:
    """
    Client for interacting with the DeFi Llama API.
    Documentation: https://defillama.com/docs/api
    """

    BASE_URL = "https://api.llama.fi"
    YIELDS_URL = "https://yields.llama.fi"

    def __init__(self, timeout: int = 30):
        """
        Initialize the DeFi Llama API client.
        
        Args:
            timeout (int): Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
    
    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Make a GET request to the API.
        
        Args:
            url (str): The URL to request
            params (Dict[str, Any], optional): Query parameters
            
        Returns:
            Dict: The JSON response
        """
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    # Protocol methods
    
    def get_protocols(self) -> Dict:
        """
        Get all protocols on DeFi Llama.
        
        Returns:
            Dict: Protocol data
        """
        return self._get(f"{self.BASE_URL}/protocols")
    
    def get_protocol(self, protocol: str) -> Dict:
        """
        Get detailed information about a protocol.
        
        Args:
            protocol (str): Protocol slug as used in the URL on the website
            
        Returns:
            Dict: Protocol details
        """
        return self._get(f"{self.BASE_URL}/protocol/{protocol}")
    
    # TVL methods
    
    def get_tvl(self) -> Dict:
        """
        Get historical TVL (excludes liquid staking and double counted tvl).
        
        Returns:
            Dict: TVL data
        """
        return self._get(f"{self.BASE_URL}/charts")
    
    def get_tvl_by_chain(self, chain: str) -> Dict:
        """
        Get historical TVL for a specific chain.
        
        Args:
            chain (str): Chain name
            
        Returns:
            Dict: Chain TVL data
        """
        return self._get(f"{self.BASE_URL}/charts/{chain}")
    
    # Pools methods
    
    def get_pools(self, **params) -> Dict:
        """
        Get all pools data.
        
        Args:
            **params: Optional parameters to filter pools
            
        Returns:
            Dict: Pools data
        """
        return self._get(f"{self.YIELDS_URL}/pools", params=params)
    
    def get_pool(self, pool_id: str) -> Dict:
        """
        Get specific pool data.
        
        Args:
            pool_id (str): Pool ID
            
        Returns:
            Dict: Pool data
        """
        return self._get(f"{self.YIELDS_URL}/pool/{pool_id}")
    
    # Coins/tokens methods
    
    def get_all_coins(self) -> Dict:
        """
        Get all coins/tokens.
        
        Returns:
            Dict: All coins data
        """
        return self._get(f"{self.BASE_URL}/coins")
    
    def get_token_prices(self, coins: Union[str, List[str]], search_width: str = "4h") -> Dict:
        """
        Get historical prices for one or multiple tokens.
        
        Args:
            coins (Union[str, List[str]]): Comma-separated list of coins or list of coins
            search_width (str): Search width (default: "4h")
            
        Returns:
            Dict: Token price data
        """
        if isinstance(coins, list):
            coins = ",".join(coins)
        
        return self._get(f"{self.BASE_URL}/prices/current/{coins}", params={"searchWidth": search_width})
    
    # Chain methods
    
    def get_chains(self) -> List[str]:
        """
        Get all chains.
        
        Returns:
            List[str]: List of all chains
        """
        return self._get(f"{self.BASE_URL}/chains")
    
    # Additional methods
    
    def get_airdrop_protocols(self) -> List[Dict]:
        """
        Get protocols with airdrops.
        
        Returns:
            List[Dict]: Protocol airdrop information
        """
        return self._get(f"{self.BASE_URL}/airdrops")
    
    def get_bridges(self) -> Dict:
        """
        Get all bridges data.
        
        Returns:
            Dict: Bridges data
        """
        return self._get(f"{self.BASE_URL}/bridges")
    
    def get_dexes(self) -> Dict:
        """
        Get all DEXes/DEX aggregators volume data.
        
        Returns:
            Dict: DEX volume data
        """
        return self._get(f"{self.BASE_URL}/dexs")
    
    def get_dex(self, dex_id: str) -> Dict:
        """
        Get a DEX volume data.
        
        Args:
            dex_id (str): DEX ID
            
        Returns:
            Dict: DEX volume data
        """
        return self._get(f"{self.BASE_URL}/dex/{dex_id}")
    
    def get_fees(self) -> Dict:
        """
        Get fees and revenue data for all protocols.
        
        Returns:
            Dict: Fees and revenue data
        """
        return self._get(f"{self.BASE_URL}/summary/fees") 