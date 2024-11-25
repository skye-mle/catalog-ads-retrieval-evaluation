import pandas as pd
import numpy as np
from typing import Dict


def calculate_precision(df: pd.DataFrame) -> float:
    """Calculate precision@k for binary relevance"""
    relevant = df["label"].sum()
    total = len(df)
    return relevant / total if total > 0 else 0


def calculate_ndcg(df: pd.DataFrame) -> float:
    """Calculate NDCG@k for binary relevance"""
    relevance = df["label"].values

    # Calculate DCG
    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))

    # Calculate IDCG
    ideal_relevance = np.sort(relevance)[::-1]
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))

    return dcg / idcg if idcg > 0 else 0


def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate evaluation metrics per keyword and average them

    Args:
        df: DataFrame with columns ['keyword', 'label', ...]
    Returns:
        Dictionary containing averaged metrics across keywords
    """
    # Calculate per-keyword metrics
    keyword_metrics = df.groupby("keyword").apply(
        lambda x: pd.Series(
            {
                "precision": calculate_precision(x),
                "ndcg": calculate_ndcg(x),
                "relevant_count": x["label"].sum(),
                "total_count": x["num_results"].mean(),
            }
        )
    )

    # Calculate query volume weighted metrics
    if "query_count" in df.columns:
        query_volumes = df.groupby("keyword")["query_count"].first()
        total_volume = query_volumes.sum()

        weighted_metrics = {
            "weighted_precision": (
                keyword_metrics["precision"] * query_volumes / total_volume
            ).sum(),
            "weighted_ndcg": (
                keyword_metrics["ndcg"] * query_volumes / total_volume
            ).sum(),
        }
    else:
        weighted_metrics = {}

    # Combine all metrics
    metrics = {
        # Simple averages across keywords
        "avg_precision": keyword_metrics["precision"].mean(),
        "avg_ndcg": keyword_metrics["ndcg"].mean(),
        "avg_relevant_per_keyword": keyword_metrics["relevant_count"].mean(),
        "avg_results_per_keyword": keyword_metrics["total_count"].mean(),
        # Overall statistics
        "total_keywords": len(keyword_metrics),
        "total_results": keyword_metrics["total_count"].mean(),
        "total_relevant": keyword_metrics["relevant_count"].mean(),
        # Distribution statistics
        "precision_std": keyword_metrics["precision"].std(),
        "ndcg_std": keyword_metrics["ndcg"].std(),
        "precision_median": keyword_metrics["precision"].median(),
        "ndcg_median": keyword_metrics["ndcg"].median(),
    }

    # Add weighted metrics if query counts are available
    metrics.update(weighted_metrics)

    return metrics
