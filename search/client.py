import requests
import json
import re
from typing import Dict, Any, Optional
import logging
import ast
from grpc_requests import Client

logger = logging.getLogger(__name__)


class SearchClient:
    def __init__(self, es_url: str, es_index: str, feature_platform_config: Dict):
        self.es_url = es_url
        self.es_index = es_index
        self.feature_platform_config = feature_platform_config
        self.fp_client = Client.get_by_endpoint(feature_platform_config["endpoint"])

    def get_keyword_category_weights(
        self, keyword: str, depth: int = 1
    ) -> Optional[Dict]:
        """Get category weights from feature platform"""
        try:
            keyword = re.sub(r"[^ㄱ-ㅎ가-힣a-zA-Z0-9]", "", keyword)

            # Set feature selector based on depth
            if depth == 1:
                feature_selector = {
                    "resolve_fleamarketarticle_searchcategoryweight_v1_features": {
                        "resolve_category_weights": True,
                        "resolve_run_date": True,
                    }
                }
            elif depth == 3:
                feature_selector = {
                    "resolve_fleamarketarticle_searchkeywordllmcategoryweight_v1_features": {
                        "resolve_category_1_weights": True,
                        "resolve_category_2_weights": True,
                        "resolve_category_3_weights": True,
                    }
                }
            else:
                raise ValueError(f"Unsupported depth: {depth}")

            # Request to feature platform
            request_body = {"keyword": keyword, "feature_selector": feature_selector}
            headers = [
                ("fp-client-name", "ads-catalog-product-category-match-test"),
            ]

            response = self.fp_client.request(
                self.feature_platform_config["service"],
                self.feature_platform_config["method"],
                request_body,
                metadata=headers,
            )

            # Extract category weights based on depth
            if depth == 1:
                category_weights = response["searchkeyword_view_entity"][
                    "fleamarketarticle_searchcategoryweight_v1_features"
                ]
            else:
                category_weights = response["searchkeyword_view_entity"][
                    "fleamarketarticle_searchkeywordllmcategoryweight_v1_features"
                ]

            return category_weights

        except Exception as e:
            logger.error(f"Error getting category weights: {str(e)}")
            return None

    def get_dsl_params(self, keyword: str, dsl_name: str) -> Dict[str, Any]:
        """Get parameters for DSL based on DSL name"""
        if dsl_name == "fasttext_category_match":
            category_weights = self.get_keyword_category_weights(keyword, depth=1)
            if category_weights:
                category_list = [
                    info["hoian_category_name"]
                    for info in json.loads(category_weights["category_weights"])
                    if info["is_boost"] == 1
                ]
                return {"category_list": category_list}
            else:
                return {"category_list": []}
        elif dsl_name.startswith("llm_category_match"):
            category_weights = self.get_keyword_category_weights(keyword, depth=3)
            if category_weights:
                category_1_weights = {
                    weight["category_id"]: weight["score"] 
                    for weight in ast.literal_eval(category_weights["category_1_weights"])}
                category_2_weights = {
                    weight["category_id"]: weight["score"]
                    for weight in ast.literal_eval(category_weights["category_2_weights"])
                }
                category_3_weights = {
                    weight["category_id"]: weight["score"]
                    for weight in ast.literal_eval(category_weights["category_3_weights"])
                }
                return {
                    "category_1_weights": category_1_weights,
                    "category_2_weights": category_2_weights,
                    "category_3_weights": category_3_weights,
                }
            else:
                return {
                    "category_1_weights": {},
                    "category_2_weights": {},
                    "category_3_weights": {},
                }

    def search(self, keyword: str, dsl_name: str) -> Dict[str, Any]:
        """Execute search with specified DSL"""
        try:
            # Get DSL parameters
            dsl_params = self.get_dsl_params(keyword, dsl_name)

            # Get DSL for the keyword
            dsl = self._get_dsl(keyword, dsl_name, dsl_params)

            # Execute search
            url = f"{self.es_url}/{self.es_index}/_search"
            headers = {"Content-Type": "application/json"}

            response = requests.post(url, headers=headers, data=json.dumps(dsl))
            response.raise_for_status()

            return response.json()["hits"]["hits"]

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise

    def _get_dsl(
        self, keyword: str, dsl_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get DSL query based on name and keyword"""
        # Import DSL module dynamically to avoid circular imports
        import dsls

        if dsl_name.startswith("llm_category_match"):
            if not params["category_1_weights"] or not params["category_2_weights"] or not params["category_3_weights"]:
                return dsls.get_keyword_match_dsl(keyword)
            
            filter_option, ranking_option = dsl_name.replace("get_llm_category_match_", "").replace("_dsl", "").split("_")
            if filter_option == "depth1filter":
                filter_query = dsls.get_filter_query(depth=1, category_weights=params["category_1_weights"])
            elif filter_option == "depth2filter":
                filter_query = dsls.get_filter_query(depth=2, category_weights=params["category_2_weights"])
            elif filter_option == "depth3filter":
                filter_query = dsls.get_filter_query(depth=3, category_weights=params["category_3_weights"])
            else:
                raise ValueError(f"Unsupported filter option: {filter_option}")
            if ranking_option == "depth1ranking":
                category_1_weights = {k: v * 100 for k, v in params["category_1_weights"].items()}
                category_2_weights = {k: v * 10 for k, v in params["category_2_weights"].items()}
                category_3_weights = {k: v * 1 for k, v in params["category_3_weights"].items()}
            elif ranking_option == "depth3ranking":
                category_1_weights = {k: v * 1 for k, v in params["category_1_weights"].items()}
                category_2_weights = {k: v * 10 for k, v in params["category_2_weights"].items()}
                category_3_weights = {k: v * 100 for k, v in params["category_3_weights"].items()}
            elif ranking_option == "equalranking":
                category_1_weights = params["category_1_weights"]
                category_2_weights = params["category_2_weights"]
                category_3_weights = params["category_3_weights"]
            else:
                raise ValueError(f"Unsupported ranking option: {ranking_option}")
            ranking_query = dsls.get_ranking_query(
                category_1_weights, category_2_weights, category_3_weights
            )
            return dsls.get_llm_category_match_dsl(keyword, filter_query, ranking_query)
        else:
            dsl_func = getattr(dsls, f"get_{dsl_name}_dsl")
            if params:
                return dsl_func(keyword, **params)
            return dsl_func(keyword)
