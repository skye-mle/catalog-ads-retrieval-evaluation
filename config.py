from dataclasses import dataclass
import os


@dataclass
class SearchConfig:
    # Elasticsearch settings
    ES_URL: str = "https://ads-searching.kr.krmt.io"
    ES_INDEX: str = "ads-catalog-product-v3"

    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Paths
    PROMPT_TEMPLATE_PATH: str = "prompts/v1.txt"

    # Search settings
    MAX_RESULTS: int = 100

    # Feature platform settings
    FEATURE_PLATFORM_ENDPOINT: str = "feature-platform-grpc.kr.krmt.io:80"
    FEATURE_PLATFORM_SERVICE: str = (
        "featureplatform.featureserving.rpc.v1.FeatureServingService"
    )
    FEATURE_PLATFORM_METHOD: str = "GetSearchKeywordViewEntity"

    # LLM settings
    NUM_LLM_REQUESTS: int = 64
    NUM_WORKERS: int = 16
