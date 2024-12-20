from typing import Dict, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def process_search_results(keyword: str, search_results: List[Dict]) -> pd.DataFrame:
    """
    Process raw search results into a structured DataFrame format.

    Args:
        keyword: The search keyword used to obtain results
        search_results: List of search result dictionaries from SearchClient

    Returns:
        pd.DataFrame: Processed results with columns:
            - keyword: search term used
            - product_id: unique identifier for the product
            - title: product title
            - description: product description
            - rank: position in search results (1-based)
            - score: search relevance score
    """
    category_info = pd.read_csv("data/category_info.csv")
    category_info = dict(zip(category_info["category_id"], category_info["category_name_ko"]))
    try:
        processed_results = []

        for rank, result in enumerate(search_results, start=1):
            processed_result = {
                "keyword": keyword,
                "product_id": result.get("_id"),
                "title": result.get("_source", {}).get("title", ""),
                "category": result.get("_source", {}).get("category_name_0", ""),
                "depth1_category": category_info.get(result.get("_source", {}).get("llm_category_depth_1_id", "")),
                "depth2_category": category_info.get(result.get("_source", {}).get("llm_category_depth_2_id", "")),
                "depth3_category": category_info.get(result.get("_source", {}).get("llm_category_depth_3_id", "")),
                "rank": rank,
                "score": result.get("_score", 0.0),
            }
            processed_results.append(processed_result)

        df_results = pd.DataFrame(processed_results)

        if df_results.empty:
            logger.warning(f"No results found for keyword: {keyword}")

        return df_results

    except Exception as e:
        logger.error(
            f"Error processing search results for keyword '{keyword}': {str(e)}"
        )
        raise
