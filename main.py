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

def create_results_dir() -> str:
    """Create results directory with timestamp"""
    base_dir = "results"
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(base_dir, timestamp)
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
    keywords_df: pd.DataFrame, dsl_name: str, config: SearchConfig
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
        for _, row in keywords_df.iterrows():
            keyword = row["keyword"]
            top_category_name = row["top_category_name"]
            query_count = row["query_count"]

            logger.info(
                f"Processing keyword: {keyword} (category: {top_category_name}), count: {query_count})"
            )

            search_results = search_client.search(keyword=keyword, dsl_name=dsl_name)
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
            ]
        ]
        metrics = calculate_metrics(df_all)

        return {"metrics": metrics, "detailed_results": df_all}

    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {str(e)}")
        raise


def save_results(results: Dict, dsl_name: str, result_dir: str):
    """Save evaluation results to files"""
    results["detailed_results"].to_csv(
        os.path.join(result_dir, f"detailed_results_{dsl_name}.csv"), index=False
    )
    result = results["detailed_results"][["keyword", "title", "label", "score"]]
    result.to_csv(os.path.join(result_dir, f"results_{dsl_name}.csv"), index=False)

    metrics_df = pd.DataFrame([results["metrics"]])
    metrics_df.to_json(
        os.path.join(result_dir, f"metrics_{dsl_name}.json"), orient="records", indent=2
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate search DSLs")
    parser.add_argument(
        "--dsls",
        type=str,
        required=True,
        help="Comma-separated list of DSL names to evaluate",
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

    dsl_names = [dsl.strip() for dsl in args.dsls.split(",")]
    keywords_df = load_keywords(args.keywords_file)
    result_dir = create_results_dir()

    setup_logging(logs_dir=result_dir)
    logger.info(f"Results will be saved to: {result_dir}")
    keywords_df.to_csv(os.path.join(result_dir, "input_keywords.csv"), index=False)

    config = SearchConfig()
    for dsl_name in dsl_names:
        logger.info(f"\nEvaluating DSL: {dsl_name}")
        results = run_evaluation(
            keywords_df=keywords_df, dsl_name=dsl_name, config=config
        )

        logger.info("\nMetrics:")
        for metric_name, value in results["metrics"].items():
            logger.info(f"{metric_name}: {value:.4f}")

        save_results(results, dsl_name, result_dir)


if __name__ == "__main__":
    main()
