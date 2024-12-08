python main.py --dsl-filter fasttext_category_match --dsl-ranking fasttext_prior --keywords-file keywords/test_keyword.csv
python main.py --dsl-filter fasttext_category_match --dsl-ranking llm_equal_prior --keywords-file keywords/test_keyword.csv
python main.py --dsl-filter fasttext_category_match --dsl-ranking llm_depth1_prior --keywords-file keywords/test_keyword.csv
python main.py --dsl-filter fasttext_category_match --dsl-ranking llm_depth3_prior --keywords-file keywords/test_keyword.csv

python main.py --dsl-filter llm_depth1_match --dsl-ranking fasttext_prior --keywords-file keywords/test_keyword.csv
python main.py --dsl-filter llm_depth1_match --dsl-ranking llm_equal_prior --keywords-file keywords/test_keyword.csv
python main.py --dsl-filter llm_depth1_match --dsl-ranking llm_depth1_prior --keywords-file keywords/test_keyword.csv
python main.py --dsl-filter llm_depth1_match --dsl-ranking llm_depth3_prior --keywords-file keywords/test_keyword.csv

python main.py --dsl-filter llm_depth2_match --dsl-ranking fasttext_prior --keywords-file keywords/test_keyword.csv
python main.py --dsl-filter llm_depth2_match --dsl-ranking llm_equal_prior --keywords-file keywords/test_keyword.csv
python main.py --dsl-filter llm_depth2_match --dsl-ranking llm_depth1_prior --keywords-file keywords/test_keyword.csv
python main.py --dsl-filter llm_depth2_match --dsl-ranking llm_depth3_prior --keywords-file keywords/test_keyword.csv

python main.py --dsl-filter llm_depth3_match --dsl-ranking fasttext_prior --keywords-file keywords/test_keyword.csv
python main.py --dsl-filter llm_depth3_match --dsl-ranking llm_equal_prior --keywords-file keywords/test_keyword.csv
python main.py --dsl-filter llm_depth3_match --dsl-ranking llm_depth1_prior --keywords-file keywords/test_keyword.csv
python main.py --dsl-filter llm_depth3_match --dsl-ranking llm_depth3_prior --keywords-file keywords/test_keyword.csv