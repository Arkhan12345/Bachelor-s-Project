import pandas as pd
from pathlib import Path
import io
import base64


_BASE = Path(__file__).resolve().parent.parent
_ARCHIVE = _BASE / "Archive"

ic = pd.read_csv(_ARCHIVE / "independent_components.txt", sep="\t", index_col=0)
gsea = pd.read_csv(_ARCHIVE / "gsea_matrix.txt", sep=",", index_col=0)
mixing = pd.read_csv(_ARCHIVE / "mixing_matrix.txt", sep="\t", index_col=0)
genes = pd.read_csv(_ARCHIVE / "genomic_mapping.txt", sep="\t")
meta = pd.read_csv(_ARCHIVE / "sample_annotations.txt", sep="\t", index_col=0)

meta.index = meta.index.astype(str).str.strip()
mixing.index = mixing.index.astype(str).str.strip()
mixing.columns = mixing.columns.astype(str).str.strip()


try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    np = None


def filter_ic(threshold: float = 3):
    """Return a copy of the IC matrix where only strong loadings (|value| > threshold)
    are kept. Cells that don't meet the threshold are set to NaN and any rows/columns
    that become entirely NaN are dropped.

    Args:
        threshold: numeric cutoff for absolute value to consider a loading "strong".

    Returns:
        pandas.DataFrame: filtered IC matrix
    """
    ic_filtered = ic.copy()
    mask = (ic_filtered > threshold) | (ic_filtered < -threshold)
    ic_filtered = ic_filtered.where(mask)
    ic_filtered = ic_filtered.dropna(axis=1, how='all').dropna(axis=0, how='all')
    return ic_filtered


def filter_gene_enrichment(threshold: float = 3):
    """Filter the GSEA matrix to keep only pathway scores with absolute value > threshold.
    Non-significant entries are set to NaN. Drops empty rows/columns.
    """
    gsea_filtered = gsea.copy()
    mask = (gsea_filtered > threshold) | (gsea_filtered < -threshold)
    gsea_filtered = gsea_filtered.where(mask)
    gsea_filtered = gsea_filtered.dropna(axis=1, how='all').dropna(axis=0, how='all')
    return gsea_filtered


def filter_mixing_m(threshold: float = 0.1):
    """Filter the mixing matrix to keep only strong sample activations per IC
    (|value| > threshold). Non-strong entries are set to NaN and empty rows/cols removed.
    
    Note: Mixing matrix values are typically much smaller than IC matrix values.
    A threshold of 0.1 is reasonable for the mixing matrix.
    """
    mixing_filtered = mixing.copy()
    mask = (mixing_filtered > threshold) | (mixing_filtered < -threshold)
    mixing_filtered = mixing_filtered.where(mask)
    mixing_filtered = mixing_filtered.dropna(axis=1, how='all').dropna(axis=0, how='all')
    return mixing_filtered


def get_top_sample_annotations(ic_name: str, top_k: int = 10):
    """Return metadata for the samples with the largest absolute score for an IC."""
    if ic_name not in mixing.index:
        return []

    scores = pd.to_numeric(mixing.loc[ic_name], errors="coerce").dropna()
    if scores.empty:
        return []

    top_scores = scores.reindex(scores.abs().sort_values(ascending=False).index).head(top_k)
    sample_meta = meta.reindex(top_scores.index).dropna(how="all").copy()
    if sample_meta.empty:
        return []

    sample_meta.insert(0, "sample_id", sample_meta.index)
    sample_meta.insert(1, "ic_score", top_scores.reindex(sample_meta.index).values)
    return sample_meta.to_dict(orient="records")


def find_ic(gene_symbol, threshold: float = 3):
    # Find the gene ID
    gene_row = genes[genes["SYMBOL"] == gene_symbol]
    if gene_row.empty:
        return f"No gene found with symbol {gene_symbol}"

    entrez = str(gene_row["ENTREZID"].values[0])

    # Filter IC matrix for strong loadings
    ic_filtered = filter_ic(threshold)
    entrez_int = int(entrez)
    if entrez_int not in ic_filtered.index:
        return f"No strong IC associations found for {gene_symbol}"

    # Select the filtered row and drop NaNs to get only strong ICs for this gene
    strong_ics = ic_filtered.loc[entrez_int].dropna()
    if strong_ics.empty:
        return f"No strong IC associations found for {gene_symbol}"

    results = []

    gsea_filtered = filter_gene_enrichment(threshold)
    for ic_name in strong_ics.index:
        # Top pathways for this IC (use the prefiltered GSEA matrix)
        if ic_name in gsea_filtered.columns:
            gsea_hits = gsea_filtered[ic_name].dropna().sort_values(ascending=False).head(5)
        else:
            gsea_hits = pd.Series(dtype=float)

        results.append({
            "IC": ic_name,
            "Loading": strong_ics[ic_name],
            "Top_Pathways": gsea_hits.to_dict(),
            "Top_Samples": get_top_sample_annotations(ic_name)
        })

    return results


def find_pathway_ics(pathway_name, threshold: float = 3):
    """Find ICs related to a specific pathway.
    
    Args:
        pathway_name: the pathway name
        threshold: value to consider a score "strong"
    
    Returns:
        List of dicts with IC info, or error string
    """
    # Check if pathway exists in GSEA matrix
    if pathway_name not in gsea.index:
        return f"Pathway '{pathway_name}' not found in GSEA matrix"
    
    # Get the pathway's row and filter for strong IC associations
    pathway_row = gsea.loc[pathway_name]
    strong_ics = pathway_row[(pathway_row > threshold) | (pathway_row < -threshold)]
    
    if strong_ics.empty:
        return f"No strong IC associations found for pathway {pathway_name}"
    
    # Sort by absolute value descending
    strong_ics = strong_ics.reindex(strong_ics.abs().sort_values(ascending=False).index)
    
    results = []
    
    # Get filtered matrices for top genes and samples
    ic_filtered = filter_ic(threshold)
    for ic_name in strong_ics.index:
        # Top genes for this IC (genes with strong loadings)
        if ic_name in ic_filtered.columns:
            top_gene_loadings = ic_filtered[ic_name].dropna().sort_values(ascending=False, key=abs).head(10)
            # Map entrez IDs to symbols
            top_genes = []
            for entrez_id in top_gene_loadings.index:
                gene_info = genes[genes["ENTREZID"] == entrez_id]
                if not gene_info.empty:
                    top_genes.append({
                        "ENTREZID": str(int(entrez_id)),
                        "SYMBOL": str(gene_info.iloc[0]["SYMBOL"]),
                        "GENETITLE": str(gene_info.iloc[0]["GENETITLE"]),
                        "Loading": top_gene_loadings[entrez_id]
                    })
        else:
            top_genes = []
        
        results.append({
            "IC": ic_name,
            "Score": strong_ics[ic_name],
            "Top_Genes": top_genes,
            "Top_Samples": get_top_sample_annotations(ic_name)
        })
    
    return results

def get_top_pathways_for_ic(ic_name: str, threshold: float, top_k: int = 10):
    """
    Return list of (pathway_name, score) sorted by absolute score desc,
    filtered by abs(score) > threshold.

    Works whether gsea has:
      - ICs as rows (index) and pathways as columns, OR
      - pathways as rows (index) and ICs as columns.
    Never raises KeyError; returns [] if not found.
    """
    # Normalize input
    ic = str(ic_name).strip()

    if "gsea" not in globals() or gsea is None:
        return []

    # Try: IC is a ROW
    if ic in gsea.index:
        scores = gsea.loc[ic]
    # Try: IC is a COLUMN
    elif ic in gsea.columns:
        scores = gsea[ic]
    else:
        # Optional: try case-insensitive match
        idx_match = [x for x in gsea.index.astype(str) if x.lower() == ic.lower()]
        col_match = [x for x in gsea.columns.astype(str) if x.lower() == ic.lower()]
        if idx_match:
            scores = gsea.loc[idx_match[0]]
        elif col_match:
            scores = gsea[col_match[0]]
        else:
            return []

    # Make sure numeric
    scores = pd.to_numeric(scores, errors="coerce").dropna()

    # Filter and sort
    filt = scores[scores.abs() > float(threshold)]
    if filt.empty:
        return []

    filt = filt.sort_values(key=lambda s: s.abs(), ascending=False)
    return [(str(idx), float(val)) for idx, val in filt.head(top_k).items()]


def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def generate_ic_enrichment_plot(ic_name, threshold: float = 3): #param in webiste
    """Generate gene enrichment (GSEA) bar plot for a specific IC.
    
    Returns base64-encoded image string or None if matplotlib not available.
    """
    if not MATPLOTLIB_AVAILABLE or ic_name not in gsea.columns:
        return None
    
    # Get GSEA scores for this IC
    gsea_col = gsea[ic_name]
    strong = gsea_col[(gsea_col > threshold) | (gsea_col < -threshold)]
    
    if strong.empty:
        return None
    
    # Sort by absolute value, take top 15
    top_pathways = strong.reindex(strong.abs().sort_values(ascending=False).index).head(15)
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(top_pathways) * 0.4)))
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in top_pathways.values]
    y_pos = np.arange(len(top_pathways))
    
    ax.barh(y_pos, top_pathways.values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p.replace('HALLMARK_', '').replace('_', ' ') for p in top_pathways.index], fontsize=9)
    ax.set_xlabel('Enrichment Score', fontsize=11)
    ax.set_title(f'Top Pathway Enrichments for {ic_name}', fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    
    return plot_to_base64(fig)


def generate_ic_sample_annotation_plots(ic_name, threshold: float = 3, mixing_threshold: float = 0.1):
    """Generate sample annotation plots for a specific IC.

    Args:
        ic_name: The IC to plot
        threshold: Threshold for gene/pathway filtering (not used here, kept for API consistency)
        mixing_threshold: Deprecated. Kept for compatibility; plots now use top/bottom
            IC-score quantiles because fixed absolute cutoffs remove most ICs.

    Returns dict of plot names to base64-encoded image strings.
    """
    if not MATPLOTLIB_AVAILABLE:
        return {}

    if ic_name not in mixing.index:
        return {}

    scores = pd.to_numeric(mixing.loc[ic_name], errors="coerce").dropna()
    if scores.empty:
        return {}

    sample_meta = meta.reindex(scores.index).dropna(how="all").copy()
    if sample_meta.empty:
        return {}
    sample_meta["ic_score"] = scores.reindex(sample_meta.index)
    sample_meta = sample_meta.dropna(subset=["ic_score"])
    if sample_meta.empty:
        return {}

    low_cut = sample_meta["ic_score"].quantile(0.20)
    high_cut = sample_meta["ic_score"].quantile(0.80)
    low = sample_meta[sample_meta["ic_score"] <= low_cut].copy()
    high = sample_meta[sample_meta["ic_score"] >= high_cut].copy()

    if low.empty or high.empty:
        return {}

    low["IC score group"] = "Low 20%"
    high["IC score group"] = "High 20%"
    grouped = pd.concat([low, high], axis=0)

    plots = {}

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sample_meta["ic_score"], bins=30, color="#4c78a8", alpha=0.75, edgecolor="white")
    ax.axvline(low_cut, color="#d62728", linestyle="--", linewidth=1.5, label="20th percentile")
    ax.axvline(high_cut, color="#2ca02c", linestyle="--", linewidth=1.5, label="80th percentile")
    ax.set_title(f"{ic_name}: IC Score Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("IC score", fontsize=10)
    ax.set_ylabel("Sample count", fontsize=10)
    ax.legend()
    fig.tight_layout()
    plots["ic_score_distribution"] = plot_to_base64(fig)

    def add_grouped_count_plot(column, title=None):
        if column not in grouped.columns:
            return

        counts = pd.crosstab(grouped[column].fillna("Unknown").astype(str), grouped["IC score group"])
        counts = counts.reindex(columns=["Low 20%", "High 20%"], fill_value=0)
        if counts.empty:
            return

        counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index].head(12)
        fig, ax = plt.subplots(figsize=(max(7, min(12, len(counts) * 0.85)), 5))
        counts.plot(kind="bar", ax=ax, color=["#d62728", "#2ca02c"], alpha=0.75)
        ax.set_title(title or f"{ic_name}: {column} by IC Score Group", fontsize=12, fontweight="bold")
        ax.set_xlabel(column, fontsize=10)
        ax.set_ylabel("Sample count", fontsize=10)
        ax.tick_params(axis="x", rotation=35)
        ax.legend(title="")
        fig.tight_layout()
        plots[f"{column.lower().replace('.', '_')}_by_score_group"] = plot_to_base64(fig)

    for column in [
        "Type_updated",
        "Subtype",
        "Stage",
        "Grade",
        "Recurrence.status",
        "Survival.status",
        "Debulking",
    ]:
        add_grouped_count_plot(column)

    if "Age" in grouped.columns:
        age_df = grouped.copy()
        age_df["Age"] = pd.to_numeric(age_df["Age"], errors="coerce")
        age_df = age_df.dropna(subset=["Age"])
        if not age_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            age_groups = [
                age_df.loc[age_df["IC score group"] == "Low 20%", "Age"].values,
                age_df.loc[age_df["IC score group"] == "High 20%", "Age"].values,
            ]
            ax.boxplot(age_groups, labels=["Low 20%", "High 20%"], patch_artist=True)
            ax.set_title(f"{ic_name}: Age by IC Score Group", fontsize=12, fontweight="bold")
            ax.set_xlabel("IC score group", fontsize=10)
            ax.set_ylabel("Age", fontsize=10)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            plots["age_by_score_group"] = plot_to_base64(fig)

    for outcome in ["OS", "PFS"]:
        if outcome in grouped.columns:
            outcome_df = grouped.copy()
            outcome_df[outcome] = pd.to_numeric(outcome_df[outcome], errors="coerce")
            outcome_df = outcome_df.dropna(subset=[outcome])
            if outcome_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            outcome_groups = [
                outcome_df.loc[outcome_df["IC score group"] == "Low 20%", outcome].values,
                outcome_df.loc[outcome_df["IC score group"] == "High 20%", outcome].values,
            ]
            ax.boxplot(outcome_groups, labels=["Low 20%", "High 20%"], patch_artist=True)
            ax.set_title(f"{ic_name}: {outcome} by IC Score Group", fontsize=12, fontweight="bold")
            ax.set_xlabel("IC score group", fontsize=10)
            ax.set_ylabel(outcome, fontsize=10)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            plots[f"{outcome.lower()}_by_score_group"] = plot_to_base64(fig)

    if "Type" in sample_meta.columns:
        type_counts = sample_meta["Type"].fillna("Unknown").astype(str).value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        type_counts.plot(kind="bar", ax=ax, color="#4c78a8", alpha=0.75)
        ax.set_title(f"{ic_name}: All Sample Type Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Type", fontsize=10)
        ax.set_ylabel("Sample count", fontsize=10)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        plots["all_sample_type_distribution"] = plot_to_base64(fig)

    if "Recurrence.status" in sample_meta.columns:
        rec_counts = sample_meta["Recurrence.status"].fillna("Unknown").astype(str).value_counts()
        if not rec_counts.empty:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(rec_counts.values, labels=rec_counts.index, autopct="%1.1f%%", startangle=90)
            ax.set_title(f"{ic_name}: All Sample Recurrence Status", fontsize=12, fontweight="bold")
            fig.tight_layout()
            plots["all_sample_recurrence_status"] = plot_to_base64(fig)

    if "Age" in sample_meta.columns:
        age_data = pd.to_numeric(sample_meta["Age"], errors="coerce").dropna()
        if not age_data.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(age_data, bins=20, color="#9467bd", alpha=0.75, edgecolor="white")
            ax.set_title(f"{ic_name}: All Sample Age Distribution", fontsize=12, fontweight="bold")
            ax.set_xlabel("Age", fontsize=10)
            ax.set_ylabel("Sample count", fontsize=10)
            ax.axvline(age_data.median(), color="red", linestyle="--", linewidth=1.5, label=f"Median: {age_data.median():.1f}")
            ax.legend()
            fig.tight_layout()
            plots["all_sample_age_distribution"] = plot_to_base64(fig)

    return plots

if __name__ == "__main__":
    gene_of_interest = "TP53"
    related_ics = find_ic(gene_of_interest)
    for ic_info in related_ics:
        print(f"IC: {ic_info['IC']}, Loading: {ic_info['Loading']}")
        print("Top Pathways:")
        for pathway, score in ic_info["Top_Pathways"].items():
            print(f"  {pathway}: {score}")
        print("Top Samples:")
        for sample in ic_info["Top_Samples"]:
            print(f"  {sample}")
        print("\n")
