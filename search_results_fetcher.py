import requests
import os
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY = os.getenv("SERP_API_KEY")

def serpapi_search(question: str, api_key: str):
    """Executes a web search using SerpApi for the given question."""
    params = {
        "q": question,
        "api_key": api_key,
        "engine": "google"
    }

    try:
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()  # Will raise an exception for HTTP error responses
        data = response.json()


        # Extract organic results
        organic_results = data.get("organic_results", [])
        if organic_results:
            # Get top 3 snippets from organic results
            summaries = [item["snippet"] for item in organic_results[:3] if "snippet" in item]
        else:
            summaries = []

        # If no organic results, check other sections (e.g., top_stories)
        if not summaries:
            top_stories = data.get("top_stories", [])
            summaries = [f"Top story: {item['link']}" for item in top_stories[:3] if "link" in item]

        # If no top stories, check related searches
        if not summaries:
            related_searches = data.get("related_searches", [])
            summaries = [f"Related search: {item['link']}" for item in related_searches[:3] if "link" in item]

        return summaries

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {response.text}")
    except Exception as err:
        print(f"An error occurred: {err}")

    return []

# ----------------------
# 🔽 Run this block to test
# ----------------------

if __name__ == "__main__":
    
    question = input("Enter your question: ")
    summaries = serpapi_search(question, SERPAPI_KEY)

    if summaries:
        print("\n🔍 Top Search Results:")
        for i, summary in enumerate(summaries, 1):
            print(f"{i}. {summary}")
    else:
        print("No results found or an error occurred.")
