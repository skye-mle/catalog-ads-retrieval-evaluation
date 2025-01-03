import logging
from typing import Dict
import argparse
import os
from datetime import datetime

from config import SearchConfig
from evaluator.llm_evaluator import LLMEvaluator
from evaluator.metrics import calculate_metrics
from search.client import SearchClient
from utils.data_processor import process_search_results
import pandas as pd
import time
from utils.logging_config import setup_logging

# Define logger at module level
logger = logging.getLogger(__name__)


def create_results_dir(dsl_filter: str, dsl_ranking: str) -> str:
    """Create results directory with timestamp"""
    base_dir = "results"
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(base_dir, f"{dsl_filter}_{dsl_ranking}_{timestamp}")
    os.makedirs(result_dir)

    return result_dir


def load_keywords(csv_path: str) -> pd.DataFrame:
    """Load keywords from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        required_columns = ["keyword", "top_category_name", "query_count"]

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")

        return df

    except Exception as e:
        logger.error(f"Error loading keywords from CSV: {str(e)}")
        raise


def run_evaluation(
    keywords_df: pd.DataFrame, dsl_filter: str, dsl_ranking: str, config: SearchConfig
) -> Dict:
    """Run evaluation pipeline for search results"""
    try:
        search_client = SearchClient(
            es_url=config.ES_URL,
            es_index=config.ES_INDEX,
            feature_platform_config={
                "endpoint": config.FEATURE_PLATFORM_ENDPOINT,
                "service": config.FEATURE_PLATFORM_SERVICE,
                "method": config.FEATURE_PLATFORM_METHOD,
            },
        )
        llm_evaluator = LLMEvaluator(
            api_key=config.OPENAI_API_KEY,
            model_name=config.OPENAI_MODEL,
            prompt_template_path=config.PROMPT_TEMPLATE_PATH,
            num_requests=config.NUM_LLM_REQUESTS,
            max_workers=config.NUM_WORKERS,
        )

        results = []
        for i, row in keywords_df.iterrows():
            keyword = row["keyword"]
            top_category_name = row["top_category_name"]
            query_count = row["query_count"]

            logger.info(
                f"Processing keyword ({i+1}/{len(keywords_df)}): {keyword} (category: {top_category_name}), count: {query_count})"
            )

            search_results = search_client.search(keyword=keyword, dsl_filter=dsl_filter, dsl_ranking=dsl_ranking)
            logger.info(f"Number of search results: {len(search_results)}")
            df_results = process_search_results(
                keyword=keyword, search_results=search_results
            )

            llm_results = llm_evaluator.evaluate_batch(df_results)
            llm_results["keyword"] = keyword
            llm_results["num_results"] = len(df_results)
            llm_results["top_category_name"] = top_category_name
            llm_results["query_count"] = query_count

            results.append(llm_results)

            time.sleep(1)

        df_all = pd.concat(results, ignore_index=True)
        df_all = df_all[
            [
                "keyword",
                "title",
                "label",
                "core_intent",
                "ads_core_intent",
                "score",
                "num_results",
                "query_count",
                "depth1_category",
                "depth2_category",
                "depth3_category",
            ]
        ]
        metrics = calculate_metrics(df_all)

        return {"metrics": metrics, "detailed_results": df_all}

    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {str(e)}")
        raise


def save_results(results: Dict, result_dir: str):
    """Save evaluation results to files"""
    results["detailed_results"].to_csv(
        os.path.join(result_dir, "detailed_results.csv"), index=False
    )
    result = results["detailed_results"][["keyword", "title", "label", "score"]]
    result.to_csv(os.path.join(result_dir, "results.csv"), index=False)

    metrics_df = pd.DataFrame([results["metrics"]])
    metrics_df.to_json(
        os.path.join(result_dir, "metrics.json"), orient="records", indent=2
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate search DSLs")
    parser.add_argument(
        "--dsl-filter",
        type=str,
        required=True,
        choices=["fasttext", "llm_depth1", "llm_depth2", "llm_depth3"],
        help="Filter DSLs by prefix (e.g. llm_category_match)",
    )
    parser.add_argument(
        "--dsl-ranking",
        type=str,
        required=True,
        choices=["fasttext", "llm_depth123_score123", "llm_depth123_score12", "llm_depth23_score123", "llm_depth23_score12", "llm_depth3_score123", "llm_depth3_score12"],
        help="Ranking DSLs by category depth",
    )
    parser.add_argument(
        "--keywords-file",
        type=str,
        required=True,
        help="Path to CSV file containing keywords",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    keywords_df = load_keywords(args.keywords_file)
    result_dir = create_results_dir(args.dsl_filter, args.dsl_ranking)

    setup_logging(logs_dir=result_dir)
    logger.info(f"Results will be saved to: {result_dir}")
    keywords_df.to_csv(os.path.join(result_dir, "input_keywords.csv"), index=False)

    config = SearchConfig()
    logger.info(f"\nEvaluating DSL: filter {args.dsl_filter}, ranking {args.dsl_ranking}")
    results = run_evaluation(
        keywords_df=keywords_df, dsl_filter=args.dsl_filter, dsl_ranking=args.dsl_ranking, config=config
    )

    logger.info("\nMetrics:")
    for metric_name, value in results["metrics"].items():
        logger.info(f"{metric_name}: {value:.4f}")

    save_results(results, result_dir)


if __name__ == "__main__":
    main()
