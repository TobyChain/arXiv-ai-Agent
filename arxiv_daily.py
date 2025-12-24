"""
arxiv_daily.py（入口脚本）

为保持历史运行方式不变，本文件仅作为薄封装入口：
  uv run python arxiv_daily.py

真实实现位于：
  pipelines/run_arxiv_daily.py
"""

from pipelines.run_arxiv_daily import main


if __name__ == "__main__":
    main()



