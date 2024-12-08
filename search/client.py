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

    def get_dsl_params(self, keyword: str) -> Dict[str, Any]:
        """Get parameters for DSL based on DSL name"""
        fasttext_category_weights = self.get_keyword_category_weights(keyword, depth=1)
        if fasttext_category_weights:
            fasttext_category_list = [
                info["hoian_category_name"]
                for info in json.loads(fasttext_category_weights["category_weights"])
                if info["is_boost"] == 1
            ]
        else:
            fasttext_category_list = []
            
        llm_category_weights = self.get_keyword_category_weights(keyword, depth=3)
        if llm_category_weights:
            category_1_weights = {
                weight["category_id"]: weight["score"] 
                for weight in ast.literal_eval(llm_category_weights["category_1_weights"])}
            category_2_weights = {
                weight["category_id"]: weight["score"]
                for weight in ast.literal_eval(llm_category_weights["category_2_weights"])
            }
            category_3_weights = {
                weight["category_id"]: weight["score"]
                for weight in ast.literal_eval(llm_category_weights["category_3_weights"])
            }
            return {
                "fasttext_category_list": fasttext_category_list,
                "category_1_weights": category_1_weights,
                "category_2_weights": category_2_weights,
                "category_3_weights": category_3_weights,
            }
        else:
            return {
                "fasttext_category_list": fasttext_category_list,
                "category_1_weights": {},
                "category_2_weights": {},
                "category_3_weights": {},
            }

    def search(self, keyword: str, dsl_filter: str, dsl_ranking: str) -> Dict[str, Any]:
        """Execute search with specified DSL"""
        try:
            # Get DSL parameters
            dsl_params = self.get_dsl_params(keyword)

            # Get DSL for the keyword
            dsl = self._get_dsl(keyword, dsl_filter, dsl_ranking, dsl_params)

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
        self, keyword: str, dsl_filter: str, dsl_ranking: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        filter_dsl = get_filter_dsl(keyword, dsl_filter, params)
        ranking_dsl = get_ranking_dsl(keyword, dsl_ranking, params)
        dsl = {
            "size": 1000,
            "query": {
                "function_score": {
                    "boost_mode": "replace",
                    "query": {
                        "bool": { "filter": filter_dsl }
                    },
                    "functions": ranking_dsl,
                    "score_mode": "sum",
                }
            },
            "_source": [
                "original_id",
                "product_id",
                "catalog_id",
                "title",
                "category_name_0",
                "fast_text_category_name",
                "llm_category_depth_1_id",
                "llm_category_depth_2_id",
                "llm_category_depth_3_id",
            ]
        }
        return dsl


def get_filter_dsl(keyword: str, dsl_filter: str, params: Dict[str, Any]) -> Dict[str, Any]:
    filter_dsl = [
        {"exists": {"field": "catalog_product_set_ids"}},
        {"term": {"is_live": {"value": True}}},
        {"term": {"availability": {"value": "IN_STOCK"}}},
        {
            "match": {
                "serving_title": {
                    "query": keyword,
                    "operator": "and",
                }
            }
        }        
    ]
    if dsl_filter == "fasttext_category_match" and params["fasttext_category_list"]:
        filter_dsl.append({
            "bool": {
                "should": [
                    {"terms": {"fast_text_category_name": params["fasttext_category_list"]}},
                    {"bool": {"must_not": {"exists": {"field": "fast_text_category_name"}}}},
                ]
            }
        })
    elif dsl_filter.startswith("llm_depth"):
        target_depth = dsl_filter.split("_")[1].replace("depth", "")
        if params["fasttext_category_list"]:
            if params[f"category_{target_depth}_weights"]:
                filter_dsl.append({
                        "bool": {
                        "should": [
                            {
                                "terms": {f"llm_category_depth_{target_depth}_id": [int(k) for k in params[f"category_{target_depth}_weights"].keys()]},
                            },
                            {
                                "bool": {
                                "filter": [
                                        {"bool": {"must_not": {"exists": {"field": f"llm_category_depth_{target_depth}_id"}}}},
                                        {
                                            "bool": {
                                                "should": [
                                                    {"terms": {"fast_text_category_name": params["fasttext_category_list"]}},
                                                    {"bool": {"must_not": {"exists": {"field": "fast_text_category_name"}}}},
                                                ]
                                            }
                                        },                                
                                    ]                            
                                }
                            }
                        ]
                    }                
                })
            else:
                filter_dsl.append({
                    "bool": {
                        "should": [
                            {"terms": {"fast_text_category_name": params["fasttext_category_list"]}},
                            {"bool": {"must_not": {"exists": {"field": "fast_text_category_name"}}}},
                        ]
                    }
                })
        elif params[f"category_{target_depth}_weights"]:
            filter_dsl.append({
                "bool": {
                    "should": [
                        {"terms": {f"llm_category_depth_{target_depth}_id": [int(k) for k in params[f"category_{target_depth}_weights"].keys()]}},
                        {"bool": {"must_not": {"exists": {"field": f"llm_category_depth_{target_depth}_id"}}}},
                    ]
                }
            })
    return filter_dsl


def get_ranking_dsl(keyword: str, dsl_ranking: str, params: Dict[str, Any]) -> Dict[str, Any]:
    ranking_dsl = [{"weight": 1, "random_score": {}}]
    if params["fasttext_category_list"]:
        ranking_dsl.append(
            {
                "weight": 1,
                "filter": {
                    "terms": {"fast_text_category_name": params["fasttext_category_list"]}
                },
            }
        )
    if dsl_ranking == "fasttext_prior":
        return ranking_dsl

    if dsl_ranking == "llm_depth1_prior":
        category_1_weights = {k: v * 1000 for k, v in params["category_1_weights"].items()}
        category_2_weights = {k: v * 100 for k, v in params["category_2_weights"].items()}
        category_3_weights = {k: v * 10 for k, v in params["category_3_weights"].items()}
    elif dsl_ranking == "llm_depth3_prior":
        category_1_weights = {k: v * 10 for k, v in params["category_1_weights"].items()}
        category_2_weights = {k: v * 100 for k, v in params["category_2_weights"].items()}
        category_3_weights = {k: v * 1000 for k, v in params["category_3_weights"].items()}
    elif dsl_ranking == "llm_equal_prior":
        category_1_weights = {k: v * 10 for k, v in params["category_1_weights"].items()}
        category_2_weights = {k: v * 10 for k, v in params["category_2_weights"].items()}
        category_3_weights = {k: v * 10 for k, v in params["category_3_weights"].items()}
    ranking_dsl.append(
        {
            "script_score": {
                "script": {
                    "source": """
                        def category1Weights = params.category1Weights;
                        def category2Weights = params.category2Weights;
                        def category3Weights = params.category3Weights;
                        def totalScore = 0;
                                        
                        if (doc['llm_category_depth_1_id'].size() != 0) {
                            def categoryId1 = doc['llm_category_depth_1_id'].value.toString();
                            totalScore += (category1Weights.containsKey(categoryId1) ? category1Weights.get(categoryId1) : 0);
                        }

                        if (doc['llm_category_depth_1_id'].size() != 0) {
                            def categoryId2 = doc['llm_category_depth_2_id'].value.toString();
                            totalScore += (category2Weights.containsKey(categoryId2) ? category2Weights.get(categoryId2) : 0);
                        }

                        if (doc['llm_category_depth_1_id'].size() != 0) {
                            def categoryId3 = doc['llm_category_depth_3_id'].value.toString();
                            totalScore += (category3Weights.containsKey(categoryId3) ? category3Weights.get(categoryId3) : 0);
                        }

                        return totalScore;
                """,
                    "params": {
                        "category1Weights": category_1_weights,
                        "category2Weights": category_2_weights,
                        "category3Weights": category_3_weights,
                    },
                }
            }
        }        
    )
    return ranking_dsl
