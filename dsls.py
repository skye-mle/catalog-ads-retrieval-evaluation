from typing import Dict, Any, List


def get_keyword_match_dsl(keyword: str) -> Dict[str, Any]:
    return {
        "size": 1000,
        "query": {
            "function_score": {
                "boost_mode": "replace",
                "query": {
                    "bool": {
                        "filter": [
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
                            },
                        ]
                    }
                },
                "functions": [{"weight": 1, "random_score": {}}],
                "score_mode": "sum",
            }
        },
        "_source": [
            "original_id",
            "product_id",
            "catalog_id",
            "title",
            "category_name_0",
        ],
    }


def get_fasttext_category_match_dsl(
    keyword: str, category_list: List[str]
) -> Dict[str, Any]:
    if category_list:
        return {
            "size": 1000,
            "query": {
                "function_score": {
                    "boost_mode": "replace",
                    "query": {
                        "bool": {
                            "filter": [
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
                                },
                                {
                                    "bool": {
                                        "should": [
                                            {
                                                "terms": {
                                                    "fast_text_category_name": category_list
                                                }
                                            },
                                            {
                                                "bool": {
                                                    "must_not": {
                                                        "exists": {
                                                            "field": "fast_text_category_name"
                                                        }
                                                    }
                                                }
                                            },
                                        ]
                                    }
                                },
                            ]
                        }
                    },
                    "functions": [
                        {
                            "weight": 10,
                            "filter": {
                                "terms": {"fast_text_category_name": category_list}
                            },
                        },
                        {"weight": 1, "random_score": {}},
                    ],
                    "score_mode": "sum",
                }
            },
            "_source": [
                "original_id",
                "product_id",
                "catalog_id",
                "title",
                "category_name_0",
            ],
        }
    else:
        return get_keyword_match_dsl(keyword)


def get_llm_category_match_dsl(
    keyword: str,
    filter_query: Dict[str, Any],
    ranking_query: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "query": {
            "function_score": {
                "score_mode": "sum",
                "boost_mode": "replace",
                "query": {
                    "bool": {
                        "filter": [
                            {"match": {"title": {"query": keyword, "operator": "and"}}},
                            filter_query,
                        ]
                    }
                },
                "functions": [ranking_query],
            }
        }
    }


def get_filter_query(depth, category_weights):
    category_list = [int(category_id) for category_id in category_weights.keys()]
    return {
        "bool": {
            "should": [
                {"terms": {f"llm_category_depth_{depth}_id": category_list}},
                {
                    "bool": {
                        "must_not": {
                            "exists": {"field": f"llm_category_depth_{depth}_id"}
                        }
                    }
                },
            ]
        }
    }


def get_ranking_query(category_1_weights, category_2_weights, category_3_weights):
    return {
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
