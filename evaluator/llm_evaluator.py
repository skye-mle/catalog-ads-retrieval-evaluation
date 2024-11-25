import pandas as pd
from typing import Dict
import json
from openai import OpenAI
from jinja2 import Template
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class LLMEvaluator:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        prompt_template_path: str,
        num_requests: int = 10,
        max_workers: int = 4,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.num_requests = num_requests
        self.max_workers = max_workers

        # Load prompt template
        with open(prompt_template_path, "r") as f:
            self.template = Template(f.read())

    def evaluate_single(self, row: Dict) -> Dict:
        """Evaluate a single search result using LLM"""
        # Prepare prompt
        prompt = self.template.render(
            {
                "query": row["keyword"],
                "query_category": row.get("top_category_name", ""),
                "title": row["title"],
                "category": row["category"],
            }
        )

        # Get LLM response
        response = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": prompt}]
        )

        # Parse response
        content = response.choices[0].message.content
        result = json.loads(content)

        return {
            "label": int(result["Score"]),
            "core_intent": result["Core_intent"],
            "ads_core_intent": result["Ads_core_intent"],
        }

    def evaluate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate multiple search results in parallel"""
        target = df.iloc[: self.num_requests]
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_row = {
                executor.submit(self.evaluate_single, row): i
                for i, row in target.iterrows()
            }

            for future in as_completed(future_to_row):
                row_idx = future_to_row[future]
                try:
                    result = future.result()
                    result.update({"product_id": target.iloc[row_idx]["product_id"]})
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing row {row_idx}: {str(e)}")
                    continue

        df_results = pd.DataFrame(results)
        df_results = pd.merge(df_results, target, on="product_id", how="left")
        return df_results
