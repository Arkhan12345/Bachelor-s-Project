from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import xml.etree.ElementTree as ET
import uuid
import sys

import pandas as pd
import requests
from flask import Flask, jsonify, redirect, render_template, request, session

LLM_URL = "http://localhost:8000/generate"

THIS_DIR = Path(__file__).resolve().parent
APP_DIR = THIS_DIR.parent
PROJECT_ROOT = APP_DIR.parent
ARCHIVE_DIR = PROJECT_ROOT / "Archive"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from pipeline import (
    find_ic,
    find_pathway_ics,
    genes,
    generate_ic_enrichment_plot,
    generate_ic_sample_annotation_plots,
    get_top_pathways_for_ic,
)

app = Flask(__name__, template_folder=str(THIS_DIR / "templates"), static_folder=str(THIS_DIR / "static"))
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
CHAT_HISTORY_MAX_TURNS = int(os.environ.get("CHAT_HISTORY_MAX_TURNS", "20"))

app.config["LLM_URL"] = LLM_URL
app.jinja_env.globals["LLM_URL"] = LLM_URL

GENESET_DATABASES = [
    {"id": "gsea_default", "name": "GSEA"},
]


def _find_gene(
    gene_query: Optional[str] = None,
    entrez: Optional[str] = None,
    symbol: Optional[str] = None,
    genetitle: Optional[str] = None,
):
    df = genes.copy()

    def _to_str(x):
        return str(x) if x is not None else None

    entrez = _to_str(entrez)
    symbol = _to_str(symbol)
    genetitle = _to_str(genetitle)
    gene_query = _to_str(gene_query)

    row = None

    if gene_query and not entrez and not symbol and not genetitle:
        match = re.match(r"^\s*([^()\s]+)\s*\((\d+)\)\s*[\u2014\-]\s*(.+)$", gene_query)
        if match:
            symbol = match.group(1)
            entrez = match.group(2)
            genetitle = match.group(3)
        else:
            match2 = re.match(r"^\s*([^()\s]+)\s*\((\d+)\)\s*$", gene_query)
            if match2:
                symbol = match2.group(1)
                entrez = match2.group(2)

    if entrez:
        try:
            eid = int(float(entrez))
            m = df[df["ENTREZID"] == eid]
            if not m.empty:
                row = m.iloc[0]
        except Exception:
            pass

    if row is None and symbol:
        m = df[df["SYMBOL"].astype(str).str.upper() == symbol.upper()]
        if not m.empty:
            row = m.iloc[0]

    if row is None and genetitle:
        m = df[df["GENETITLE"].astype(str).str.upper() == genetitle.upper()]
        if not m.empty:
            row = m.iloc[0]

    if row is None and gene_query:
        if gene_query.replace(".", "", 1).isdigit():
            try:
                eid = int(float(gene_query))
                m = df[df["ENTREZID"] == eid]
                if not m.empty:
                    row = m.iloc[0]
            except Exception:
                pass

        if row is None:
            m = df[df["SYMBOL"].astype(str).str.upper() == gene_query.upper()]
            if not m.empty:
                row = m.iloc[0]

        if row is None:
            m = df[df["GENETITLE"].astype(str).str.contains(gene_query, case=False, na=False)]
            if not m.empty:
                row = m.iloc[0]

    if row is None:
        return None, None

    row_dict = {
        "ENTREZID": str(int(row["ENTREZID"])) if pd.notna(row["ENTREZID"]) else "",
        "SYMBOL": str(row["SYMBOL"]) if pd.notna(row["SYMBOL"]) else "",
        "GENETITLE": str(row["GENETITLE"]) if pd.notna(row["GENETITLE"]) else "",
    }
    display = f"{row_dict['SYMBOL']} ({row_dict['ENTREZID']}) — {row_dict['GENETITLE']}"
    return row_dict, display


def fetch_publications(query: str, max_results: int = 10):
    query = (query or "").strip()
    if not query:
        return []

    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    es_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(max_results),
        "sort": "date",
    }
    r = requests.get(esearch_url, params=es_params, timeout=15)
    r.raise_for_status()
    es = r.json()
    pmids = es.get("esearchresult", {}).get("idlist") or []
    if not pmids:
        return []

    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    ef_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    r2 = requests.get(efetch_url, params=ef_params, timeout=20)
    r2.raise_for_status()

    root = ET.fromstring(r2.text)
    results = []

    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//PMID") or "").strip()
        title = (article.findtext(".//ArticleTitle") or "").strip()

        abs_parts = []
        for ab in article.findall(".//Abstract/AbstractText"):
            label = ab.attrib.get("Label")
            txt = "".join(ab.itertext()).strip()
            if txt:
                abs_parts.append(f"{label}: {txt}" if label else txt)
        abstract = "\n".join(abs_parts).strip()

        journal = (article.findtext(".//Journal/Title") or "").strip()

        year = (article.findtext(".//JournalIssue/PubDate/Year") or "").strip()
        if not year:
            medline = (article.findtext(".//JournalIssue/PubDate/MedlineDate") or "").strip()
            year = medline[:4] if medline[:4].isdigit() else ""

        author_list = []
        for a in article.findall(".//AuthorList/Author"):
            last = (a.findtext("LastName") or "").strip()
            initials = (a.findtext("Initials") or "").strip()
            collective = (a.findtext("CollectiveName") or "").strip()
            if collective:
                author_list.append(collective)
            elif last:
                author_list.append(f"{last} {initials}".strip())
        authors = ", ".join(author_list[:12])

        doi = ""
        for aid in article.findall(".//ArticleIdList/ArticleId"):
            if aid.attrib.get("IdType") == "doi":
                doi = (aid.text or "").strip()
                break

        results.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": year,
                "source": journal,
                "doi": doi,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            }
        )

    return results


def summarize_publications(pub_list, max_papers_for_llm=5):
    pubs_for_prompt = [p for p in pub_list if p.get("abstract")][:max_papers_for_llm]

    if not pubs_for_prompt:
        return "No abstracts available to summarize."

    prompt_parts = [
        "You are an expert biomedical summarizer.",
        "Given the following publications:",
        "1) Write a short paragraph describing the shared biological theme.",
        "2) Provide a bullet list summarizing each paper in one sentence, including year and title.",
        "Publications:",
    ]

    for i, p in enumerate(pubs_for_prompt, start=1):
        title = p.get("title", "")[:300]
        year = p.get("year") or p.get("pubYear") or ""
        abstract_snip = p.get("abstract", "")[:2000]
        prompt_parts.append(f"{i}. Title: {title} ({year})")
        prompt_parts.append(f"Abstract: {abstract_snip}")
        prompt_parts.append("")

    prompt = "\n".join(prompt_parts)

    payload = {
        "prompt": prompt,
        "raw_prompt": True,
        "max_new_tokens": 600,
        "temperature": 0.2,
        "top_p": 0.9,
    }
    resp = requests.post(LLM_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json().get("output", "")


def summarize_publication_hits(pub_list, max_papers=5):
    """Create deterministic publication context when the LLM summary is empty."""
    if not pub_list:
        return ""

    lines = [f"Publication search found {len(pub_list)} paper(s). Relevant examples:"]
    for p in pub_list[:max_papers]:
        title = (p.get("title") or "Untitled").strip()
        year = (p.get("year") or "").strip()
        source = (p.get("source") or "").strip()
        abstract = " ".join((p.get("abstract") or "").split())
        details = ", ".join(x for x in [year, source] if x)
        prefix = f"{title} ({details})" if details else title
        if abstract:
            lines.append(f"- {prefix}: {abstract[:700]}")
        else:
            lines.append(f"- {prefix}")

    return "\n".join(lines)


def _load_ic_annotation_merged(ic_name: str):
    sample_path = ARCHIVE_DIR / "sample_annotations.txt"
    mixing_path = ARCHIVE_DIR / "mixing_matrix.txt"

    # sample annotations: first column is sample ID
    sample_ann = pd.read_csv(sample_path, sep="\t", dtype=str, index_col=0)
    mixing = pd.read_csv(mixing_path, sep="\t", index_col=0)

    # Clean labels
    sample_ann.columns = sample_ann.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    sample_ann.index = sample_ann.index.astype(str).str.strip()
    mixing.columns = mixing.columns.astype(str).str.strip()
    mixing.index = mixing.index.astype(str).str.strip()

    if ic_name not in mixing.index:
        return None, f"{ic_name}: IC scores not found in mixing matrix."

    # Turn sample annotation index into a real column
    sample_ann = sample_ann.reset_index().rename(columns={"index": "sample_id"})

    # Extract scores for this IC
    ic_scores = mixing.loc[ic_name].reset_index()
    ic_scores.columns = ["sample_id", "ic_score"]

    # Clean IDs
    sample_ann["sample_id"] = (
        sample_ann["sample_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    ic_scores["sample_id"] = (
        ic_scores["sample_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    merged = sample_ann.merge(ic_scores, on="sample_id", how="inner")

    if merged.empty:
        return None, "No overlapping sample annotation data found."

    merged["ic_score"] = pd.to_numeric(merged["ic_score"], errors="coerce")
    merged = merged.dropna(subset=["ic_score"])

    if merged.empty:
        return None, "Merged data exists, but IC scores could not be parsed as numeric."

    return merged, None


def summarize_annotation_patterns(ic_name, threshold):
    summaries = []

    try:
        merged, err = _load_ic_annotation_merged(ic_name)
        if err:
            return [err]

        high_cut = merged["ic_score"].quantile(0.80)
        low_cut = merged["ic_score"].quantile(0.20)

        high = merged[merged["ic_score"] >= high_cut].copy()
        low = merged[merged["ic_score"] <= low_cut].copy()

        if high.empty or low.empty:
            return ["Could not define high- and low-scoring IC sample groups."]

        if "Age" in merged.columns:
            high_age = pd.to_numeric(high["Age"], errors="coerce").dropna()
            low_age = pd.to_numeric(low["Age"], errors="coerce").dropna()
            if not high_age.empty and not low_age.empty:
                summaries.append(
                    f"Age: high-IC samples have mean age {high_age.mean():.1f}, versus {low_age.mean():.1f} in low-IC samples."
                )

        if "Stage" in merged.columns:
            high_stage = high["Stage"].dropna().astype(str)
            low_stage = low["Stage"].dropna().astype(str)
            if not high_stage.empty and not low_stage.empty:
                high_top = high_stage.value_counts(normalize=True)
                low_top = low_stage.value_counts(normalize=True)
                if not high_top.empty:
                    summaries.append(
                        f"Stage: the most common stage among high-IC samples is {high_top.index[0]} ({high_top.iloc[0]*100:.1f}%)."
                    )
                if not low_top.empty:
                    summaries.append(
                        f"Stage comparison: the most common stage among low-IC samples is {low_top.index[0]} ({low_top.iloc[0]*100:.1f}%)."
                    )

        if "Grade" in merged.columns:
            high_grade = high["Grade"].dropna().astype(str)
            if not high_grade.empty:
                top_grade = high_grade.value_counts(normalize=True)
                summaries.append(
                    f"Grade: the most common grade among high-IC samples is {top_grade.index[0]} ({top_grade.iloc[0]*100:.1f}%)."
                )

        if "Subtype" in merged.columns:
            high_sub = high["Subtype"].dropna().astype(str)
            if not high_sub.empty:
                top_sub = high_sub.value_counts(normalize=True)
                summaries.append(
                    f"Subtype: the most common subtype among high-IC samples is {top_sub.index[0]} ({top_sub.iloc[0]*100:.1f}%)."
                )

        if "Type_updated" in merged.columns:
            high_type = high["Type_updated"].dropna().astype(str)
            if not high_type.empty:
                top_type = high_type.value_counts(normalize=True)
                summaries.append(
                    f"Tumor type: the most common tumor category among high-IC samples is {top_type.index[0]} ({top_type.iloc[0]*100:.1f}%)."
                )

        if "Survival.status" in merged.columns:
            high_surv = high["Survival.status"].dropna().astype(str)
            if not high_surv.empty:
                top_surv = high_surv.value_counts(normalize=True)
                summaries.append(
                    f"Survival status: among high-IC samples, the most common status is {top_surv.index[0]} ({top_surv.iloc[0]*100:.1f}%)."
                )

        if "Recurrence.status" in merged.columns:
            high_rec = high["Recurrence.status"].dropna().astype(str)
            if not high_rec.empty:
                top_rec = high_rec.value_counts(normalize=True)
                summaries.append(
                    f"Recurrence: among high-IC samples, the most common recurrence category is {top_rec.index[0]} ({top_rec.iloc[0]*100:.1f}%)."
                )

        for col in ["Platinum", "Taxol", "Debulking"]:
            if col in merged.columns:
                vals = high[col].dropna().astype(str)
                if not vals.empty:
                    top_val = vals.value_counts(normalize=True)
                    summaries.append(
                        f"{col}: among high-IC samples, the most common category is {top_val.index[0]} ({top_val.iloc[0]*100:.1f}%)."
                    )

        if not summaries:
            summaries.append("No structured annotation trends could be computed for this IC.")

    except Exception as e:
        summaries.append(f"Sample annotation summary unavailable: {e}")

    return summaries


def get_plot_conclusions(ic_name, threshold):
    conclusions = []

    try:
        merged, err = _load_ic_annotation_merged(ic_name)
        if err:
            return [err]

        high_cut = merged["ic_score"].quantile(0.80)
        high = merged[merged["ic_score"] >= high_cut].copy()
        if high.empty:
            return ["No high-scoring sample group could be defined."]

        top_pathways = get_top_pathways_for_ic(ic_name, threshold, top_k=3)

        if top_pathways:
            first = top_pathways[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                conclusions.append(f"Top enriched pathway: {first[0]} ({first[1]:+.3f}).")
            elif isinstance(first, dict):
                name = first.get("pathway") or first.get("name")
                score = first.get("score")
                if name is not None and score is not None:
                    conclusions.append(f"Top enriched pathway: {name} ({float(score):+.3f}).")

        if "Subtype" in high.columns:
            vals = high["Subtype"].dropna().astype(str)
            if not vals.empty:
                vc = vals.value_counts(normalize=True)
                conclusions.append(f"Most common subtype among high-scoring samples: {vc.index[0]} ({vc.iloc[0]*100:.1f}%).")

        if "Grade" in high.columns:
            vals = high["Grade"].dropna().astype(str)
            if not vals.empty:
                vc = vals.value_counts(normalize=True)
                conclusions.append(f"Most common grade among high-scoring samples: {vc.index[0]} ({vc.iloc[0]*100:.1f}%).")

        if "Stage" in high.columns:
            vals = high["Stage"].dropna().astype(str)
            if not vals.empty:
                vc = vals.value_counts(normalize=True)
                conclusions.append(f"Most common stage among high-scoring samples: {vc.index[0]} ({vc.iloc[0]*100:.1f}%).")

        if "Age" in high.columns:
            ages = pd.to_numeric(high["Age"], errors="coerce").dropna()
            if not ages.empty:
                conclusions.append(f"Median age of high-scoring samples: {ages.median():.1f} years.")

        if "Type_updated" in high.columns:
            vals = high["Type_updated"].dropna().astype(str)
            if not vals.empty:
                vc = vals.value_counts(normalize=True)
                conclusions.append(f"Most common tumor category among high-scoring samples: {vc.index[0]} ({vc.iloc[0]*100:.1f}%).")

        if "Recurrence.status" in high.columns:
            vals = high["Recurrence.status"].dropna().astype(str)
            if not vals.empty:
                vc = vals.value_counts(normalize=True)
                conclusions.append(f"Most common recurrence status: {vc.index[0]} ({vc.iloc[0]*100:.1f}%).")

        if "Survival.status" in high.columns:
            vals = high["Survival.status"].dropna().astype(str)
            if not vals.empty:
                vc = vals.value_counts(normalize=True)
                conclusions.append(f"Most common survival status: {vc.index[0]} ({vc.iloc[0]*100:.1f}%).")

        if not conclusions:
            conclusions.append("No structured conclusions could be extracted from the available plot data.")

    except Exception as e:
        conclusions.append(f"Plot conclusions unavailable: {e}")

    return conclusions


def _as_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _count_summary_items(text):
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    bullet_like = [
        line for line in lines
        if line.startswith("-") or re.match(r"^(bullet\s*)?\d+[\).:]\s+", line, flags=re.IGNORECASE)
    ]
    return len(bullet_like) if bullet_like else (1 if text and text.strip() else 0)


def _build_structured_judgement(ic, gene, top_pathways, plot_conclusions, annotation_summary, publication_summary):
    clean_gene = (gene or "").strip()
    clean_pub = (publication_summary or "").strip()
    top_pathways = top_pathways or []
    plot_conclusions = plot_conclusions or []
    annotation_summary = annotation_summary or []

    if top_pathways:
        pathway_name, pathway_score = top_pathways[0]
        score = _as_float(pathway_score)
        main_signal = f"{pathway_name} ({score:+.3f})"
        main_direction = "positive" if score > 0 else "negative"
        interpretation = (
            f"The most supported interpretation for {ic} is that it captures biology related to "
            f"{str(pathway_name).replace('HALLMARK_', '').replace('_', ' ').lower()}, because this is the strongest pathway signal."
        )
    else:
        main_signal = "no pathway above the selected threshold"
        main_direction = "not available"
        interpretation = (
            f"The biological interpretation for {ic} is weak because no pathway enrichment passed the selected threshold."
        )

    if top_pathways:
        pathway_bits = []
        for name, score in top_pathways[:3]:
            score = _as_float(score)
            direction = "positive" if score > 0 else "negative"
            pathway_bits.append(f"{name} ({score:+.3f}, {direction})")
        pathway_text = "The strongest pathway evidence is " + "; ".join(pathway_bits) + "."
    else:
        pathway_text = "There are no strong pathway signals available at this threshold."

    sample_patterns = []
    for line in plot_conclusions:
        if not str(line).lower().startswith("top enriched pathway"):
            sample_patterns.append(str(line))
    if not sample_patterns:
        sample_patterns = [str(x) for x in annotation_summary[:3]]
    if sample_patterns:
        sample_text = " ".join(sample_patterns[:4])
    else:
        sample_text = "No sample-level annotation pattern was available for this IC."

    if clean_gene and clean_pub and clean_pub.lower() != "none":
        pub_snippet = _shorten_text(clean_pub, 900)
        gene_text = (
            f"The related gene context is {clean_gene}; publication context was found: {pub_snippet}"
        )
    elif clean_gene:
        gene_text = (
            f"The related gene context is {clean_gene}, but publication-based literature support was not available in this run."
        )
    else:
        gene_text = "No related gene was provided for this IC, and publication-based literature support was not available in this run."

    evidence_level = "moderate" if top_pathways and (plot_conclusions or annotation_summary) else "weak"
    caution_text = (
        f"Overall evidence strength is {evidence_level}: the pathway signal is {main_signal} and its direction is {main_direction}, "
        "while the clinical/sample associations are descriptive and should be treated as hypothesis-generating."
    )

    return "\n".join([
        f"- {interpretation}",
        f"- {pathway_text}",
        f"- Sample-level evidence: {sample_text}",
        f"- {gene_text}",
        f"- {caution_text}",
    ])


def _is_greeting(message):
    normalized = re.sub(r"[^a-z\s]", "", (message or "").lower()).strip()
    return normalized in {
        "hi",
        "hello",
        "hey",
        "hey there",
        "hi there",
        "hello there",
        "good morning",
        "good afternoon",
        "good evening",
    }


def _normalize_chat_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())).strip()


def _is_vague_followup(message):
    normalized = _normalize_chat_text(message)
    if not normalized:
        return False
    vague_phrases = {
        "more",
        "a bit more",
        "tell me more",
        "can you tell me more",
        "something else",
        "anything else",
        "what else",
        "go on",
        "continue",
        "expand",
        "elaborate",
    }
    if normalized in vague_phrases:
        return True
    words = normalized.split()
    return len(words) <= 5 and any(word in normalized for word in ("more", "else", "expand", "elaborate"))


def _shorten_text(text, max_chars=360):
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _context_lines(items):
    return [str(item).strip() for item in (items or []) if str(item).strip()]


def _find_context_line(lines, *needles):
    lowered_needles = [needle.lower() for needle in needles]
    for line in lines:
        lowered = line.lower()
        if all(needle in lowered for needle in lowered_needles):
            return line
    return ""


def _is_publication_question(question):
    return any(
        term in question
        for term in (
            "publication",
            "publications",
            "public",
            "pubmed",
            "literature",
            "literatur",
            "paper",
            "papers",
            "article",
            "articles",
            "study",
            "studies",
        )
    )


def _format_publication_context(publication_hits, max_papers=5):
    lines = []
    for idx, pub in enumerate((publication_hits or [])[:max_papers], start=1):
        title = _shorten_text(pub.get("title") or "(no title)", 160)
        year = str(pub.get("year") or "").strip()
        authors = _shorten_text(pub.get("authors") or "", 120)
        abstract = _shorten_text(pub.get("abstract") or "", 700)
        pieces = [f"{idx}. {title}"]
        if year:
            pieces.append(f"({year})")
        if authors:
            pieces.append(f"Authors: {authors}.")
        if abstract:
            pieces.append(f"Abstract: {abstract}")
        lines.append(" ".join(pieces))
    return "\n".join(lines) if lines else "none"


def _answer_direct_chat_question(message, top_pathways, plot_conclusions, annotation_summary, publication_summary, publication_hits):
    question = _normalize_chat_text(message)
    evidence_lines = _context_lines(plot_conclusions) + _context_lines(annotation_summary)

    if "median age" in question or ("age" in question and "median" in question):
        line = _find_context_line(evidence_lines, "median age")
        if line:
            return line if line.endswith(".") else f"{line}."
        return "I do not see a median age value in the available sample annotation context."

    if _is_publication_question(question):
        if ("first" in question or "1st" in question) and publication_hits:
            first = publication_hits[0]
            title = first.get("title") or "(no title)"
            year = first.get("year") or "no year listed"
            source = first.get("source") or ""
            abstract = _shorten_text(first.get("abstract") or "", 1000)
            if abstract:
                source_text = f" in {source}" if source else ""
                return f'The first publication listed is "{title}" ({year}){source_text}. In short: {abstract}'
            return f'The first publication listed is "{title}" ({year}), but I do not have an abstract for it.'
        if "how many" in question or "count" in question:
            count = len(publication_hits or [])
            return f"There are {count} publication(s) currently loaded for this IC."
        if publication_summary:
            return _shorten_text(publication_summary, 1200)
        if publication_hits:
            return _format_publication_context(publication_hits, max_papers=5)
        return "I do not see any loaded publication context for this IC yet."

    if "strongest" in question and "pathway" in question and top_pathways:
        name, score = top_pathways[0]
        score = _as_float(score)
        direction = "positive" if score > 0 else "negative"
        return f"The strongest pathway evidence is {name} ({score:+.3f}, {direction})."

    if any(word in question for word in ("sample", "samples", "annotation", "subtype", "grade", "stage", "recurrence", "survival")):
        if evidence_lines:
            return " ".join(line if line.endswith(".") else f"{line}." for line in evidence_lines[:3])
        return "I do not see sample annotation evidence for this IC in the current page context."

    return ""


def _last_assistant_message(history):
    for item in reversed(history or []):
        if item.get("role") == "assistant":
            return str(item.get("content") or "").strip()
    return ""


def _normalize_answer_text(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _build_chat_continuation(ic, top_pathways, plot_conclusions, annotation_summary, publication_summary):
    bits = []
    if len(top_pathways or []) > 1:
        pathway_bits = []
        for name, score in top_pathways[1:3]:
            score = _as_float(score)
            direction = "positive" if score > 0 else "negative"
            pathway_bits.append(f"{name} ({score:+.3f}, {direction})")
        bits.append("Another angle is the secondary pathway signal: " + "; ".join(pathway_bits) + ".")

    sample_lines = [
        line for line in _context_lines(plot_conclusions)
        if not line.lower().startswith("top enriched pathway")
    ]
    if not sample_lines:
        sample_lines = _context_lines(annotation_summary)
    if sample_lines:
        bits.append("On the sample side, " + _shorten_text(sample_lines[0], 220))

    if publication_summary:
        bits.append("The loaded publication context adds: " + _shorten_text(publication_summary, 420))

    if bits:
        return " ".join(bits[:2])
    return f"I do not have much more context for {ic} beyond the pathway list shown on this page."


@app.route("/")
def index():
    return render_template("index.html", default_threshold=1.5, genesets=GENESET_DATABASES)


@app.route("/search", methods=["GET"])
def search():
    geneset = request.args.get("geneset", type=str, default="gsea_default")
    gene_query = request.args.get("gene", type=str, default="").strip()
    entrez = request.args.get("entrez", type=str)
    symbol = request.args.get("symbol", type=str)
    genetitle = request.args.get("genetitle", type=str)
    threshold = request.args.get("threshold", default=1.5)
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 3

    row, display = _find_gene(gene_query, entrez, symbol, genetitle)
    if row is None:
        return redirect("/")

    result = find_ic(row["SYMBOL"], threshold=threshold)
    error: Optional[str] = None
    records: List[Dict[str, Any]] = []

    if isinstance(result, str):
        error = result
    else:
        records = sorted(result, key=lambda r: abs(float(r.get("Loading", 0) or 0)), reverse=True)

    return render_template(
        "results.html",
        gene=row["SYMBOL"],
        gene_display=display,
        entrez=row["ENTREZID"],
        genetitle=row["GENETITLE"],
        geneset=geneset,
        genesets=GENESET_DATABASES,
        threshold=threshold,
        error=error,
        records=records,
    )


@app.route("/api/find_ic")
def api_find_ic():
    gene_query = request.args.get("gene", type=str, default="").strip()
    entrez = request.args.get("entrez", type=str)
    symbol = request.args.get("symbol", type=str)
    genetitle = request.args.get("genetitle", type=str)
    threshold = request.args.get("threshold", default=3)
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 3

    row, display = _find_gene(gene_query, entrez, symbol, genetitle)
    if row is None:
        return jsonify({"error": "Gene not found"}), 404

    result = find_ic(row["SYMBOL"], threshold=threshold)
    if isinstance(result, str):
        return jsonify({"error": result}), 404
    return jsonify({"gene": row, "display": display, "threshold": threshold, "data": result})


@app.route("/api/gene_suggest")
def api_gene_suggest():
    q = request.args.get("q", type=str, default="").strip()
    if not q:
        return jsonify({"items": []})

    df = genes
    mask = (
        df["SYMBOL"].astype(str).str.contains(q, case=False, na=False)
        | df["GENETITLE"].astype(str).str.contains(q, case=False, na=False)
        | df["ENTREZID"].astype(str).str.startswith(q)
    )
    sub = df.loc[mask, ["ENTREZID", "SYMBOL", "GENETITLE"]].head(20)

    items = []
    for _, r in sub.iterrows():
        item = {
            "entrez": str(int(r["ENTREZID"])) if pd.notna(r["ENTREZID"]) else "",
            "symbol": str(r["SYMBOL"]) if pd.notna(r["SYMBOL"]) else "",
            "genetitle": str(r["GENETITLE"]) if pd.notna(r["GENETITLE"]) else "",
        }
        item["label"] = f"{item['symbol']} ({item['entrez']}) — {item['genetitle']}"
        items.append(item)

    return jsonify({"items": items})


@app.route("/pathway/<path:pathway_name>")
def pathway_ics(pathway_name):
    geneset = request.args.get("geneset", type=str, default="gsea_default")
    threshold = request.args.get("threshold", default=3)
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 3

    result = find_pathway_ics(pathway_name, threshold=threshold)
    error: Optional[str] = None
    records: List[Dict[str, Any]] = []

    if isinstance(result, str):
        error = result
    else:
        records = result

    return render_template(
        "pathway_results.html",
        pathway=pathway_name,
        geneset=geneset,
        threshold=threshold,
        error=error,
        records=records,
    )


@app.route("/ic/<ic_name>")
def ic_detail(ic_name):
    threshold = request.args.get("threshold", default=3, type=float)
    gene = request.args.get("gene", type=str)

    enrichment_plot = generate_ic_enrichment_plot(ic_name, threshold)
    annotation_plots = generate_ic_sample_annotation_plots(ic_name, threshold)
    top_pathways = get_top_pathways_for_ic(ic_name, threshold, top_k=10)
    annotation_summary = summarize_annotation_patterns(ic_name, threshold)
    plot_conclusions = get_plot_conclusions(ic_name, threshold)

    return render_template(
        "ic_detail.html",
        ic_name=ic_name,
        threshold=threshold,
        gene=gene,
        enrichment_plot=enrichment_plot,
        annotation_plots=annotation_plots,
        top_pathways=top_pathways,
        annotation_summary=annotation_summary,
        plot_conclusions=plot_conclusions,
    )


@app.route("/api/ic_publications")
def api_ic_publications():
    ic_name = request.args.get("ic", type=str, default="").strip()
    gene = request.args.get("gene", type=str, default="").strip()
    threshold = request.args.get("threshold", type=float, default=3.0)
    max_results = request.args.get("max", type=int, default=10)

    if not ic_name:
        return jsonify({"error": "Missing ic parameter"}), 400

    try:
        top_pathways = get_top_pathways_for_ic(ic_name, threshold, top_k=5)
    except Exception as e:
        return jsonify({"error": "Failed to get IC pathways", "detail": str(e)}), 500

    pathway_terms = []
    for item in top_pathways[:3]:
        pathway_name = ""
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            pathway_name = str(item[0])
        elif isinstance(item, dict):
            pathway_name = item.get("pathway") or item.get("name") or ""

        pathway_name = pathway_name.replace("HALLMARK_", "").replace("_", " ").strip()
        if pathway_name:
            pathway_terms.append(f'"{pathway_name}"[Title/Abstract]')

    query_parts = []
    if gene:
        query_parts.append(f'"{gene}"[Title/Abstract]')
    if pathway_terms:
        query_parts.append("(" + " OR ".join(pathway_terms) + ")")

    if not query_parts:
        return jsonify({"error": "No usable IC context for literature search"}), 400

    query = " AND ".join(query_parts) + " AND hasabstract[text]"
    attempted_queries = [query]

    try:
        pubs = fetch_publications(query, max_results=max_results)
        if not pubs and gene and pathway_terms:
            query = "(" + " OR ".join(pathway_terms) + ") AND hasabstract[text]"
            attempted_queries.append(query)
            pubs = fetch_publications(query, max_results=max_results)
        if not pubs and gene:
            query = f'"{gene}"[Title/Abstract] AND hasabstract[text]'
            attempted_queries.append(query)
            pubs = fetch_publications(query, max_results=max_results)
    except Exception as e:
        return jsonify({"error": "Failed to fetch publications", "detail": str(e)}), 500

    try:
        llm_summary = summarize_publications(pubs, max_papers_for_llm=5).strip()
    except Exception:
        llm_summary = ""

    if pubs and (
        not llm_summary
        or llm_summary.lower() == "no abstracts available to summarize."
    ):
        llm_summary = summarize_publication_hits(pubs, max_papers=5)

    return jsonify(
        {
            "ic": ic_name,
            "gene": gene,
            "query": query,
            "attempted_queries": attempted_queries,
            "top_pathways": top_pathways,
            "publications": pubs,
            "llm_summary": llm_summary,
        }
    )


@app.route("/summary", methods=["POST"])
def summary():
    data = request.get_json() or {}

    ic = data.get("ic", "")
    threshold = data.get("threshold", "")
    gene = data.get("gene", "")
    has_enrichment = data.get("hasEnrichment", False)
    annotation_names = data.get("annotationNames", [])
    top_pathways = data.get("topPathways") or []
    annotation_summary = data.get("annotationSummary") or []
    plot_conclusions = data.get("plotConclusions") or []
    publication_summary = (data.get("publicationSummary") or "").strip()

    pathway_lines = "\n".join([f"- {name}: {score:+.3f}" for name, score in top_pathways]) or "none"
    annotation_summary_text = "\n".join(f"- {x}" for x in annotation_summary) if annotation_summary else "none"
    plot_conclusions_text = "\n".join(f"- {x}" for x in plot_conclusions) if plot_conclusions else "none"
    publication_summary_text = publication_summary if publication_summary else "none"

    prompt = f"""
You are a biomedical research assistant interpreting Independent Component Analysis (ICA) results from gene expression data.

Component: {ic}
Threshold: {threshold}
Related gene: {gene}

Enrichment plot available: {"yes" if has_enrichment else "no"}
Sample annotation plots: {", ".join(annotation_names) if annotation_names else "none"}

Top pathway enrichments:
{pathway_lines}

Structured plot conclusions:
{plot_conclusions_text}

Sample annotation evidence:
{annotation_summary_text}

Publication-based literature summary:
{publication_summary_text}

Write a detailed but careful overall judgement for this IC.

Required format:
- Write exactly 5 bullet points.
- Each bullet must be 1-2 complete sentences.
- Do not write a single short paragraph.
- Do not stop after only describing the top pathway.

Cover these five ideas, one per bullet:
- Most likely biological interpretation of this IC.
- Strongest pathway signals and their direction.
- Sample-level annotation patterns, including subtype, grade, stage, tumor type, age, recurrence, or survival when available.
- Related gene and publication summary when evidence is available.
- Cautious conclusion, including whether the evidence is strong, moderate, weak, or mostly descriptive.

Rules:
- Only make claims supported by the provided pathway, annotation, plot, or publication evidence.
- If publication evidence is unavailable, say that literature support was not available in this run.
- If clinical subgroup evidence is descriptive rather than causal, say so clearly.
- Be specific and avoid generic filler.
- Use '-' bullets only.

ASSISTANT ANSWER:
-
""".strip()

    try:
        payload = {
            "prompt": prompt,
            "raw_prompt": True,
            "max_new_tokens": 600,
            "temperature": 0.3,
            "top_p": 0.9,
        }
        resp = requests.post(LLM_URL, json=payload, timeout=120)
        resp.raise_for_status()
        response_data = resp.json()
        reply = (
            response_data.get("output")
            or response_data.get("reply")
            or response_data.get("text")
            or response_data.get("generated_text")
            or ""
        ).strip()

        marker = "ASSISTANT ANSWER:"
        if marker in reply:
            after_marker = reply.rsplit(marker, 1)[-1].strip()
            if after_marker:
                reply = after_marker
        if _count_summary_items(reply) < 3:
            reply = _build_structured_judgement(
                ic,
                gene,
                top_pathways,
                plot_conclusions,
                annotation_summary,
                publication_summary,
            )

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_msg = (data.get("message") or "").strip()
    ctx = data.get("context") or {}

    if not user_msg:
        return jsonify({"reply": "Please ask something."}), 400

    if _is_greeting(user_msg):
        return jsonify({
            "reply": "Hi! Ask me anything about this IC, its pathways, sample annotations, or publications."
        })

    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex

    sid = session["sid"]
    ic = (ctx.get("ic") or "").strip()
    gene = (ctx.get("gene") or "").strip()
    threshold = ctx.get("threshold")
    judgement = (ctx.get("judgement") or "").strip()
    top_pathways = ctx.get("topPathways") or []
    annotation_summary = ctx.get("annotationSummary") or []
    plot_conclusions = ctx.get("plotConclusions") or []
    publication_summary = (ctx.get("publicationSummary") or "").strip()
    publication_hits = ctx.get("publicationHits") or []

    histories = session.get("chat_histories", {})
    key = f"{sid}::{ic}::{gene}::{threshold}"
    history = histories.get(key, [])
    pathway_lines = "\n".join([f"- {name}: {score:+.3f}" for name, score in top_pathways]) or "none"
    annotation_summary_text = "\n".join(f"- {x}" for x in annotation_summary) if annotation_summary else "none"
    plot_conclusions_text = "\n".join(f"- {x}" for x in plot_conclusions) if plot_conclusions else "none"
    publication_summary_text = publication_summary if publication_summary else "none"
    publication_lines = _format_publication_context(publication_hits)
    previous_assistant = _last_assistant_message(history)
    is_vague_followup = _is_vague_followup(user_msg)

    direct_reply = _answer_direct_chat_question(
        user_msg,
        top_pathways,
        plot_conclusions,
        annotation_summary,
        publication_summary,
        publication_hits,
    )
    if direct_reply:
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": direct_reply})
        histories[key] = history[-CHAT_HISTORY_MAX_TURNS:]
        session["chat_histories"] = histories
        return jsonify({"reply": direct_reply})

    history.append({"role": "user", "content": user_msg})

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in history[-CHAT_HISTORY_MAX_TURNS:]
    )

    prompt = f"""SYSTEM:
You are a biomedical research assistant interpreting ICA results from gene expression.
Follow the rules exactly.

CONTEXT:
IC: {ic}
Threshold: {threshold}
Related gene: {gene}

Top pathway enrichments (score shown; positive/negative indicate direction):
{pathway_lines}

Structured plot conclusions:
{plot_conclusions_text}

Sample annotation evidence:
{annotation_summary_text}

Publication summary:
{publication_summary_text}

Loaded publications:
{publication_lines}

Background (do NOT quote or repeat this text; use only as context):
\"\"\"
{judgement}
\"\"\"

Previous assistant answer:
\"\"\"
{previous_assistant or "none"}
\"\"\"

Latest user message is a vague follow-up: {"yes" if is_vague_followup else "no"}

CHAT HISTORY:
{history_text}

RULES:
- Keep the answer concise and natural: usually 2-5 sentences, or 2-4 short bullets only when bullets help.
- Answer the user's exact question first.
- For publication questions, use the loaded publications or publication summary.
- For age, subtype, grade, stage, recurrence, survival, or sample questions, use the sample annotation evidence and plot conclusions.
- Do NOT copy the background judgement text verbatim. Summarize it only if relevant.
- Do NOT repeat a previous assistant answer or sentence.
- If the latest user message is a vague follow-up, add a new detail from pathways, sample annotations, or publications instead of restating the strongest pathway.
- Write ONLY the assistant answer. Do NOT repeat the system prompt.
- Answer the LAST USER message.
- If the user asks to name pathways, ONLY use the pathway list shown above.
- If the pathway list is 'none', say you cannot name them from the provided context.
- When naming pathways, you must copy the pathway label exactly as written in the list.
- If you can't copy at least one exact label, answer: I can't list pathways because none were provided.

ASSISTANT ANSWER:
""".strip()

    def _call_llm(prompt_text: str) -> str:
        payload = {
            "prompt": prompt_text,
            "raw_prompt": True,
            "max_new_tokens": 300,
            "temperature": 0.4,
            "top_p": 0.9,
        }
        resp = requests.post(LLM_URL, json=payload, timeout=120)
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError:
            return ""
        return (
            data.get("output")
            or data.get("reply")
            or data.get("text")
            or data.get("generated_text")
            or ""
        ).strip()

    try:
        reply = _call_llm(prompt)
        marker = "ASSISTANT ANSWER:"
        if marker in reply:
            reply = reply.split(marker, 1)[-1].strip()

        reply = re.sub(
            r"^(SYSTEM:|CONTEXT:|CHAT HISTORY:|RULES:)[^\n]*\n?",
            "",
            reply,
            flags=re.MULTILINE,
        ).strip()

        if previous_assistant and _normalize_answer_text(reply) == _normalize_answer_text(previous_assistant):
            reply = _build_chat_continuation(
                ic,
                top_pathways,
                plot_conclusions,
                annotation_summary,
                publication_summary,
            )

        if not reply:
            retry_prompt = f"""SYSTEM: You are a biomedical research assistant.

CONTEXT:
IC: {ic}
Threshold: {threshold}
Related gene: {gene}
Top pathway enrichments:
{pathway_lines}
Structured plot conclusions:
{plot_conclusions_text}
Sample annotation evidence:
{annotation_summary_text}
Publication summary:
{publication_summary_text}
Loaded publications:
{publication_lines}

USER: {user_msg}
ASSISTANT ANSWER:""".strip()

            reply = _call_llm(retry_prompt)
            if marker in reply:
                reply = reply.split(marker, 1)[-1].strip()
            if not reply:
                reply = _build_chat_continuation(
                    ic,
                    top_pathways,
                    plot_conclusions,
                    annotation_summary,
                    publication_summary,
                )

        history.append({"role": "assistant", "content": reply})
        history = history[-CHAT_HISTORY_MAX_TURNS:]

        histories[key] = history
        session["chat_histories"] = histories

        return jsonify({"reply": reply})
    except Exception:
        fallback = _build_chat_continuation(
            ic,
            top_pathways,
            plot_conclusions,
            annotation_summary,
            publication_summary,
        )
        return jsonify({"reply": fallback})


@app.route("/chat/reset", methods=["POST"])
def chat_reset():
    data = request.get_json() or {}
    ctx = data.get("context") or {}

    if "sid" not in session:
        return jsonify({"ok": True})

    sid = session["sid"]
    ic = (ctx.get("ic") or "").strip()
    gene = (ctx.get("gene") or "").strip()
    threshold = ctx.get("threshold")

    histories = session.get("chat_histories", {})
    key = f"{sid}::{ic}::{gene}::{threshold}"
    histories.pop(key, None)
    session["chat_histories"] = histories

    return jsonify({"ok": True})


@app.route("/api/gene_publications")
def api_gene_publications():
    gene = request.args.get("gene", type=str, default="").strip()
    max_results = request.args.get("max", type=int, default=10)
    if not gene:
        return jsonify({"error": "Missing gene parameter"}), 400

    raw = gene
    m = re.match(r"^\s*([^()\s]+)", raw)
    symbol = m.group(1) if m else raw
    query = f'{symbol}[Title/Abstract] AND hasabstract[text]'

    try:
        pubs = fetch_publications(query, max_results=max_results)
    except Exception as e:
        return jsonify({"error": "Failed to fetch publications", "detail": str(e)}), 500

    try:
        llm_summary = summarize_publications(pubs, max_papers_for_llm=5)
    except Exception as e:
        llm_summary = f"LLM summary failed: {e}"

    return jsonify({"gene": gene, "publications": pubs, "llm_summary": llm_summary})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
