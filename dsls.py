from typing import Dict, Any, List


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
