import requests
import json
import streamlit as st
from typing import List, Dict, Any, Optional
from config import SERPER_API_URL, DEFAULT_SEARCH_RESULTS


class WebSearcher:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.enabled = api_key is not None

    def search(
        self, query: str, num_results: int = DEFAULT_SEARCH_RESULTS
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            st.warning("Web search is disabled. Please add Serper API key.")
            return []

        try:
            payload = json.dumps({"q": query, "num": num_results})

            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            }

            response = requests.post(SERPER_API_URL, headers=headers, data=payload)

            if response.status_code == 200:
                data = response.json()
                results = []

                # Extract organic search results
                if "organic" in data:
                    for result in data["organic"]:
                        results.append(
                            {
                                "title": result.get("title", ""),
                                "snippet": result.get("snippet", ""),
                                "link": result.get("link", ""),
                                "source": "web_search",
                            }
                        )

                return results
            else:
                st.error(f"Web search failed with status: {response.status_code}")
                return []

        except Exception as e:
            st.error(f"Web search error: {str(e)}")
            return []

    def format_search_results_for_context(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""

        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_result = f"""
Web Result {i}:
Title: {result["title"]}
Content: {result["snippet"]}
Source: {result["link"]}
"""
            formatted_results.append(formatted_result)

        return "\n".join(formatted_results)
