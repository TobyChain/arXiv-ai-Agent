from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from clients.call_llm import LLMClient
from clients.fetch_crossref import fetch_bibtex_by_doi
from clients.fetch_openalex import reconstruct_abstract, search_works
from clients.fetch_unpaywall import extract_best_oa_urls, fetch_unpaywall_record
from renderers.render_markdown import papers_to_markdown


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _slugify(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[^a-z0-9]+", "-", t)
    t = re.sub(r"-{2,}", "-", t).strip("-")
    return t[:60] or "query"


def _pick_domain_from_work(work: Dict[str, Any]) -> str:
    concepts = work.get("concepts") or []
    if not concepts:
        return "综合"
    top = concepts[0] or {}
    return (top.get("display_name") or "综合").strip() or "综合"


def _authors_from_work(work: Dict[str, Any]) -> List[str]:
    authors: List[str] = []
    for a in work.get("authorships") or []:
        author = (a.get("author") or {}).get("display_name")
        if author:
            authors.append(str(author))
    return authors


def _best_urls_from_openalex(work: Dict[str, Any]) -> Dict[str, Optional[str]]:
    best = work.get("best_oa_location") or {}
    pdf_url = best.get("pdf_url")
    landing_page_url = best.get("landing_page_url")
    if not landing_page_url:
        primary = work.get("primary_location") or {}
        landing_page_url = primary.get("landing_page_url")
    return {"pdf_url": pdf_url, "landing_page_url": landing_page_url}


def _recommendation_order() -> Dict[str, int]:
    return {"极度推荐": 5, "很推荐": 4, "推荐": 3, "一般推荐": 2, "不推荐": 1}


def _extract_arxiv_id(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"(\d{4}\.\d{4,5})", text)
    return m.group(1) if m else None


def _fallback_bibtex(
    *,
    title: str,
    authors: List[str],
    year: Optional[str],
    url: Optional[str],
    doi: Optional[str],
    key_hint: Optional[str] = None,
) -> str:
    key = key_hint or (doi or title or "paper")
    key = _slugify(key).replace("-", "")
    author_text = " and ".join([a for a in authors if a])
    y = (year or "").strip()[:4] if year else ""
    doi_line = f"  doi={{{doi}}},\n" if doi else ""
    url_line = f"  url={{{url}}},\n" if url else ""
    return (
        f"@article{{{key},\n"
        f"  title={{{title}}},\n"
        f"  author={{{author_text}}},\n"
        f"  year={{{y}}},\n"
        f"{url_line}"
        f"{doi_line}"
        f"}}"
    )


def _dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key_of(r: Dict[str, Any]) -> str:
        doi = (r.get("doi") or "").strip().lower()
        if doi and "arxiv." in doi:
            arx_from_doi = _extract_arxiv_id(doi)
            if arx_from_doi:
                return f"arxiv:{arx_from_doi}"
        if doi:
            return f"doi:{doi}"
        arx = _extract_arxiv_id(r.get("abs_url") or "") or _extract_arxiv_id(
            r.get("doi_url") or ""
        )
        if arx:
            return f"arxiv:{arx}"
        title = (r.get("title") or "").strip().lower()
        year = (r.get("published") or "")[:4]
        return f"t:{title}|y:{year}"

    def score(r: Dict[str, Any]) -> int:
        s = 0
        if r.get("pdf_url"):
            s += 10
        if r.get("doi"):
            s += 5
        s += min(len(r.get("summary") or ""), 2000) // 200
        s += min(len(r.get("authors") or []), 20)
        return s

    best: Dict[str, Dict[str, Any]] = {}
    for r in records:
        k = key_of(r)
        if k not in best or score(r) > score(best[k]):
            best[k] = r
    return list(best.values())


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="检索关键词（英文/中文均可）")
    parser.add_argument("--from-date", default=None, help="起始发布日期 YYYY-MM-DD")
    parser.add_argument("--to-date", default=None, help="结束发布日期 YYYY-MM-DD")
    parser.add_argument("--max", type=int, default=50, help="最大返回数量（<=200）")
    parser.add_argument("--no-llm", action="store_true", help="跳过 LLM 分析（更快）")
    args = parser.parse_args(argv)

    query = args.query.strip()
    max_n = max(1, min(int(args.max), 200))

    works, _next = search_works(
        query=query,
        from_date=args.from_date,
        to_date=args.to_date,
        per_page=max_n,
    )
    if not works:
        logger.warning("未检索到结果")
        return

    llm = LLMClient()
    processed: List[Dict[str, Any]] = []

    for i, w in enumerate(works, start=1):
        title = (w.get("title") or "").strip()
        doi = (w.get("doi") or "").replace("https://doi.org/", "").strip() or None
        published = (w.get("publication_date") or "").strip()
        authors = _authors_from_work(w)
        domain = _pick_domain_from_work(w)

        abstract = reconstruct_abstract(w.get("abstract_inverted_index"))

        urls = _best_urls_from_openalex(w)
        abs_url = urls.get("landing_page_url") or (w.get("id") or "")
        pdf_url = urls.get("pdf_url")

        if (not pdf_url) and doi:
            rec = fetch_unpaywall_record(doi)
            if rec:
                best_urls = extract_best_oa_urls(rec)
                pdf_url = pdf_url or best_urls.get("pdf_url")
                abs_url = abs_url or best_urls.get("landing_page_url")

        analysis: Dict[str, Any] = {
            "trans_abs": "",
            "compressed": "",
            "keywords": [],
            "sub_topic": domain,
            "recommendation": "一般推荐",
        }
        if (not args.no_llm) and abstract:
            logger.info(f"[{i}/{len(works)}] LLM 分析: {title}")
            analysis_json = llm.translate_abstract(
                f"Title: {title}\nAbstract: {abstract}",
                domain=domain,
            )
            try:
                analysis = json.loads(analysis_json)
            except Exception as e:
                logger.warning(f"解析 LLM JSON 失败，使用回退值: {e}")
                analysis["trans_abs"] = analysis_json

        bibtex = fetch_bibtex_by_doi(doi) if doi else None
        if not bibtex:
            arxiv_id = _extract_arxiv_id(abs_url) or _extract_arxiv_id(doi or "")
            bibtex = _fallback_bibtex(
                title=title,
                authors=authors,
                year=published[:4] if published else None,
                url=abs_url,
                doi=doi,
                key_hint=arxiv_id or doi or title,
            )

        processed.append(
            {
                "title": title,
                "authors": authors,
                "abs_url": abs_url,
                "pdf_url": pdf_url,
                "trans_abs": analysis.get("trans_abs", ""),
                "keywords": analysis.get("keywords", []),
                "sub_topic": analysis.get("sub_topic", domain),
                "recommendation": analysis.get("recommendation", "一般推荐"),
                "bibtex": bibtex,
                "doi": doi,
                "doi_url": f"https://doi.org/{doi}" if doi else None,
                "published": published,
                "summary": abstract,
                "source": "openalex",
                "openalex_id": w.get("id"),
            }
        )

    processed = _dedupe_records(processed)

    if not args.no_llm:
        order = _recommendation_order()
        processed.sort(
            key=lambda x: order.get(x.get("recommendation") or "", 0), reverse=True
        )

    today = datetime.now().strftime("%Y-%m-%d")
    slug = _slugify(query)
    out_dir = os.path.join(_project_root(), "database", "unipaper")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"{today}_{slug}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    logger.success(f"JSON 已保存: {json_path}")

    md_text = papers_to_markdown(today, processed)
    md_path = os.path.join(out_dir, f"{today}_{slug}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    logger.success(f"Markdown 已保存: {md_path}")


