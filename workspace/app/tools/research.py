import logging
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

logger = logging.getLogger(__name__)

class ResearchTool:
    """
    Tool for performing research and fetching data from external APIs.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.timeout = 15  # Standard request timeout in seconds

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout)),
        reraise=True
    )
    def _make_request(self, url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Internal helper to handle network requests with built-in retry logic
        for Connection and Timeout errors.
        """
        try:
            response = requests.get(
                url, 
                params=params, 
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout as e:
            logger.warning(f"Request timed out for {url}, retrying... Error: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error occurred while fetching {url}: {e}")
            raise

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for information based on a query.
        """
        # Example API endpoint for research
        url = "https://api.research-provider.com/v1/search"
        params = {"q": query}
        
        try:
            data = self._make_request(url, params)
            return data.get("results", [])
        except Exception as e:
            logger.error(f"Research search failed after retries: {str(e)}")
            return []

    def get_details(self, identifier: str) -> Dict[str, Any]:
        """
        Fetch detailed information for a specific identifier.
        """
        url = f"https://api.research-provider.com/v1/details/{identifier}"
        
        try:
            return self._make_request(url)
        except Exception as e:
            logger.error(f"Research detail retrieval failed after retries: {str(e)}")
            return {}
