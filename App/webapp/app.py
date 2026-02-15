from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
import re

import uuid
import pandas as pd

from flask import Flask, jsonify, redirect, render_template, request, url_for, session

# Import the pipeline module
import sys

import requests
LLM_URL = "http://localhost:8000/generate"

THIS_DIR = Path(__file__).resolve().parent
APP_DIR = THIS_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from pipeline import find_ic, find_pathway_ics, genes, generate_ic_enrichment_plot, generate_ic_sample_annotation_plots
from pipeline import get_top_pathways_for_ic

# Flask app setup
app = Flask(__name__, template_folder=str(THIS_DIR / "templates"), static_folder=str(THIS_DIR / "static"))

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
CHAT_HISTORY_MAX_TURNS = int(os.environ.get("CHAT_HISTORY_MAX_TURNS", "20"))

app.config['LLM_URL'] = LLM_URL
app.jinja_env.globals['LLM_URL'] = LLM_URL

# List of gene set databases
GENESET_DATABASES = [
    {"id": "gsea_default", "name": "GSEA"},
]


def _find_gene(gene_query: Optional[str] = None,
                  entrez: Optional[str] = None,
                  symbol: Optional[str] = None,
                  genetitle: Optional[str] = None):
    """Gene selection from mapping. Parse inputs with multiple variations.
    Supports combined labels like "SYMBOL (1234) — TITLE".
    Returns tuple (row_dict or None, display_str)
    """
    df = genes.copy()
    def _to_str(x):
        return str(x) if x is not None else None

    entrez = _to_str(entrez)
    symbol = _to_str(symbol)
    genetitle = _to_str(genetitle)
    gene_query = _to_str(gene_query)

    row = None
    # Try to parse a combined label like "SYMBOL (1234) — TITLE"
    if gene_query and not entrez and not symbol and not genetitle:
        match = re.match(r"^\s*([^()\s]+)\s*\((\d+)\)\s*[\u2014\-]\s*(.+)$", gene_query)
        if match:
            symbol = match.group(1)
            entrez = match.group(2)
            genetitle = match.group(3)
        else:
            # Just "SYMBOL (1234)"
            match2 = re.match(r"^\s*([^()\s]+)\s*\((\d+)\)\s*$", gene_query)
            if match2:
                symbol = match2.group(1)
                entrez = match2.group(2)

    # 1) Prefer explicit ENTREZID
    if entrez:
        try:
            eid = int(float(entrez))
            m = df[df["ENTREZID"] == eid]
            if not m.empty:
                row = m.iloc[0]
        except Exception:
            pass
    # 2) Symbol exact (case-insensitive)
    if row is None and symbol:
        m = df[df["SYMBOL"].astype(str).str.upper() == symbol.upper()]
        if not m.empty:
            row = m.iloc[0]
    # 3) Gene title exact (case-insensitive)
    if row is None and genetitle:
        m = df[df["GENETITLE"].astype(str).str.upper() == genetitle.upper()]
        if not m.empty:
            row = m.iloc[0]
    # 4) Free-text query: try numeric as ENTREZ, else symbol exact, else title contains
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
    url = "https://pubmed.ncbi.nlm.nih.gov/"
    params = {
        "query": query,
        "format": "json",
        "pageSize": max_results,
        "sort": "date"
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    results = []
    for rec in data.get("resultList", {}).get("result", []):
        # abstracts can be missing; prefer abstractText or abstract
        abstract = rec.get("abstractText") or rec.get("abstract")
        results.append({
            "id": rec.get("id"),
            "pmid": rec.get("pmid"),
            "title": rec.get("title"),
            "abstract": (abstract or "").strip(),
            "authors": rec.get("authorString"),
            "year": rec.get("pubYear"),
            "source": rec.get("journalTitle"),
            "doi": rec.get("doi"),
            "url": rec.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url")
        })
    return results


def summarize_publications(pub_list, max_papers_for_llm=5):
    pubs_for_prompt = [p for p in pub_list if p.get("abstract")]
    pubs_for_prompt = pubs_for_prompt[:max_papers_for_llm]

    if not pubs_for_prompt:
        return "No abstracts available to summarize."

    prompt_parts = ["You are an expert biomedical summarizer. Given the following publications, produce:"]
    prompt_parts.append("2) A short bullet list (one sentence each) summarizing each paper's main finding (include year and title).")
    prompt_parts.append("Publications:")
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


# Render home page
@app.route("/")
def index():
    return render_template("index.html", default_threshold=3, genesets=GENESET_DATABASES)


# Render search results page
@app.route("/search", methods=["GET"])  # GET so it's linkable
def search():
    # geneset database (future use)
    geneset = request.args.get("geneset", type=str, default="gsea_default")
    # gene selection inputs
    gene_query = request.args.get("gene", type=str, default="").strip()
    entrez = request.args.get("entrez", type=str)
    symbol = request.args.get("symbol", type=str)
    genetitle = request.args.get("genetitle", type=str)
    threshold = request.args.get("threshold", default=3)
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 3

    # Resolve gene
    row, display = _find_gene(gene_query, entrez, symbol, genetitle)
    if row is None:
        # If empty, redirect back to home
        return redirect(url_for("index"))

    # Use symbol for the pipeline function (keeps current behavior)
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


# API endpoint for dynamic IC fetching
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


# Autocomplete gene suggestions
@app.route("/api/gene_suggest")
def api_gene_suggest():
    q = request.args.get("q", type=str, default="").strip()
    if not q:
        return jsonify({"items": []})

    df = genes
    # match if q in symbol or title or equals ENTREZID prefix
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


# Pathway-related ICs page
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


# IC detail page with plots
@app.route("/ic/<ic_name>")
def ic_detail(ic_name):
    threshold = request.args.get("threshold", default=3, type=float)
    gene = request.args.get("gene", type=str)
    
    # Generate plots
    enrichment_plot = generate_ic_enrichment_plot(ic_name, threshold)
    annotation_plots = generate_ic_sample_annotation_plots(ic_name, threshold)
    
    print(f"\n=== FLASK DEBUG ===")
    print(f"IC: {ic_name}, Threshold: {threshold}")
    print(f"Enrichment plot exists: {enrichment_plot is not None}")
    print(f"Annotation plots type: {type(annotation_plots)}")
    print(f"Annotation plots: {annotation_plots.keys() if annotation_plots else 'empty/None'}")
    print(f"=== END FLASK DEBUG ===\n")
    top_pathways = get_top_pathways_for_ic(ic_name, threshold, top_k=10)
    return render_template(
        "ic_detail.html",
        ic_name=ic_name,
        threshold=threshold,
        gene=gene,
        enrichment_plot=enrichment_plot,
        annotation_plots=annotation_plots,
        top_pathways=top_pathways
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
    pathway_lines = "\n".join([f"- {name}: {score:+.3f}" for name, score in top_pathways]) or "none"


    prompt = f"""
        You are a biomedical research assistant interpreting Independent Component Analysis (ICA) results.

        Component: {ic}
        Threshold: {threshold}
        Related gene: {gene}

        Enrichment plot available: {"yes" if has_enrichment else "no"}
        Sample annotation plots: {", ".join(annotation_names) if annotation_names else "none"}
        Top pathway enrichments:
        {pathway_lines}

        Provide a concise biological interpretation in 4-6 bullet points.
        - Use '-' bullets only
        - Do NOT number bullets
        - Be specific and mechanistic where possible

        ASSISTANT ANSWER:
        -
        """.strip()

    try:
        payload = {
            "prompt": prompt,
            "raw_prompt": True,
            "max_new_tokens": 300,
            "temperature": 0.3,
            "top_p": 0.9,
        }
        resp = requests.post(LLM_URL, json=payload, timeout=120)
        resp.raise_for_status()
        reply = resp.json().get("output", "").strip()
        marker = "ASSISTANT ANSWER:"
        if marker in reply:
            reply = reply.split(marker, 1)[-1].strip()
        if not reply:
            reply = "Sorry — I didn't generate a valid answer. Please re-ask your question."

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

    # Identify this browser session
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex

    sid = session["sid"]

    ic = (ctx.get("ic") or "").strip()
    gene = (ctx.get("gene") or "").strip()
    threshold = ctx.get("threshold")
    judgement = (ctx.get("judgement") or "").strip()

    # Per-session, per-IC chat history
    # Stored in signed cookie session, so keep it reasonably sized
    histories = session.get("chat_histories", {})
    key = f"{sid}::{ic}::{gene}::{threshold}"
    history = histories.get(key, [])

    top_pathways = ctx.get("topPathways") or []
    pathway_lines = "\n".join([f"- {name}: {score:+.3f}" for name, score in top_pathways]) or "none"

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

    Background (do NOT quote or repeat this text; use only as context):
    """
    {judgement}
    """

    CHAT HISTORY:
    {history_text}

    RULES:
    -Do NOT copy the background judgement text verbatim. Summarize it only if relevant.
    - Write ONLY the assistant answer. Do NOT repeat the system prompt.
    - Answer the LAST USER message.
    - If the user asks to name pathways, ONLY use the pathway list shown above.
    - If the pathway list is 'none', say you cannot name them from the provided context.
    - When naming pathways, you must copy the pathway label exactly as written in the list.
    - If you can't copy at least one exact label, answer: I can't list pathways because none were provided.

    ASSISTANT ANSWER:
    -
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
        return (data.get("output") or "").strip()

    try:
        reply = _call_llm(prompt)
        marker = "ASSISTANT ANSWER:"
        if marker in reply:
            reply = reply.split(marker, 1)[-1].strip()

        # If it still starts with a prompt-like line, hard-trim common echoes

        reply = re.sub(
            r"^(SYSTEM:|CONTEXT:|CHAT HISTORY:|RULES:)[^\n]*\n?",
            "",
            reply,
            flags=re.MULTILINE
        ).strip()

        # Fallback if reply got nuked
        if not reply:
            # One retry with a shorter prompt to reduce formatting echoes.
            retry_prompt = f"""SYSTEM: You are a biomedical research assistant.\n\nCONTEXT:\nIC: {ic}\nThreshold: {threshold}\nRelated gene: {gene}\nTop pathway enrichments:\n{pathway_lines}\n\nUSER: {user_msg}\nASSISTANT ANSWER:""".strip()
            reply = _call_llm(retry_prompt)
            if marker in reply:
                reply = reply.split(marker, 1)[-1].strip()
            if not reply:
                reply = "Sorry — I didn't generate a valid answer. Please re-ask your question."

        # Append assistant reply and trim
        history.append({"role": "assistant", "content": reply})
        history = history[-CHAT_HISTORY_MAX_TURNS:]

        histories[key] = history
        session["chat_histories"] = histories

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"}), 500


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

    query = f'{gene}'

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
 
