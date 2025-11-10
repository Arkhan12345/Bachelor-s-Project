from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
import re

import pandas as pd

from flask import Flask, jsonify, redirect, render_template, request, url_for

# Import the pipeline module
import sys
THIS_DIR = Path(__file__).resolve().parent
APP_DIR = THIS_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from pipeline import find_ic, find_pathway_ics, genes, generate_ic_enrichment_plot, generate_ic_sample_annotation_plots

# Flask app setup
app = Flask(__name__, template_folder=str(THIS_DIR / "templates"), static_folder=str(THIS_DIR / "static"))

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
        records = result

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
    
    return render_template(
        "ic_detail.html",
        ic_name=ic_name,
        threshold=threshold,
        gene=gene,
        enrichment_plot=enrichment_plot,
        annotation_plots=annotation_plots,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
