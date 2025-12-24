"""
unipaper_search.py（入口脚本）

保持历史运行方式不变：
  uv run python unipaper_search.py --query "xxx" --no-llm

真实实现位于：
  pipelines/run_unipaper_search.py
"""

from pipelines.run_unipaper_search import main


if __name__ == "__main__":
    main()



