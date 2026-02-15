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
meta = pd.read_csv(_ARCHIVE / "sample_annotations.txt", sep="\t")


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
    mixing_filtered = filter_mixing_m(0.1)

    for ic_name in strong_ics.index:
        # Top pathways for this IC (use the prefiltered GSEA matrix)
        if ic_name in gsea_filtered.columns:
            gsea_hits = gsea_filtered[ic_name].dropna().sort_values(ascending=False).head(5)
        else:
            gsea_hits = pd.Series(dtype=float)

        # Top samples (most active) for this IC (use prefiltered mixing matrix)
        if ic_name in mixing_filtered.index:
            active_samples = mixing_filtered.loc[ic_name].dropna().sort_values(ascending=False).head(10)
        else:
            active_samples = pd.Series(dtype=float)

        # Get metadata for these samples (empty if active_samples is empty)
        sample_meta = meta.loc[meta.index.isin(active_samples.index)]

        results.append({
            "IC": ic_name,
            "Loading": strong_ics[ic_name],
            "Top_Pathways": gsea_hits.to_dict(),
            "Top_Samples": sample_meta.to_dict(orient="records")
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
    mixing_filtered = filter_mixing_m(threshold)
    
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
        
        # Top samples for this IC
        if ic_name in mixing_filtered.index:
            active_samples = mixing_filtered.loc[ic_name].dropna().sort_values(ascending=False).head(10)
        else:
            active_samples = pd.Series(dtype=float)
        
        sample_meta = meta.loc[meta.index.isin(active_samples.index)]
        
        results.append({
            "IC": ic_name,
            "Score": strong_ics[ic_name],
            "Top_Genes": top_genes,
            "Top_Samples": sample_meta.to_dict(orient="records")
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
        mixing_threshold: Threshold for sample activation filtering (default 0.1)
    
    Returns dict of plot names to base64-encoded image strings.
    """
    print(f"\n=== DEBUG generate_ic_sample_annotation_plots ===")
    print(f"IC: {ic_name}, Mixing Threshold: {mixing_threshold}")
    
    if not MATPLOTLIB_AVAILABLE:
        print("ERROR: Matplotlib not available")
        return {}
    
    # Filter mixing matrix to get only strong sample associations
    mixing_filtered = filter_mixing_m(mixing_threshold)
    print(f"Filtered mixing matrix shape: {mixing_filtered.shape}")
    
    if ic_name not in mixing_filtered.index:
        print(f"ERROR: {ic_name} not in filtered index")
        print(f"Available ICs (first 10): {mixing_filtered.index[:10].tolist()}")
        return {}
    
    # Get samples strongly associated with this IC (already filtered)
    strong_samples = mixing_filtered.loc[ic_name].dropna()
    print(f"Strong samples found: {len(strong_samples)}")
    
    if strong_samples.empty:
        print("ERROR: No strong samples found")
        return {}
    
    # Get metadata for these samples
    sample_indices = strong_samples.index
    sample_meta = meta[meta.index.isin(sample_indices)].copy()
    print(f"Sample metadata shape: {sample_meta.shape}")
    print(f"Sample metadata columns: {sample_meta.columns.tolist()}")
    
    if sample_meta.empty:
        print("ERROR: No sample metadata found")
        return {}
    
    plots = {}
    
    # Plot 1: Type distribution
    if 'Type' in sample_meta.columns:
        type_counts = sample_meta['Type'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        type_counts.plot(kind='bar', ax=ax, color='#1f77b4', alpha=0.7)
        ax.set_title(f'{ic_name}: Sample Type Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Type', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        plots['type_distribution'] = plot_to_base64(fig)
    
    # Plot 2: Grade distribution
    if 'Grade' in sample_meta.columns:
        grade_counts = sample_meta['Grade'].astype(str).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7, 5))
        grade_counts.plot(kind='bar', ax=ax, color='#ff7f0e', alpha=0.7)
        ax.set_title(f'{ic_name}: Sample Grade Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Grade', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        fig.tight_layout()
        plots['grade_distribution'] = plot_to_base64(fig)
    
    # Plot 3: Stage distribution
    if 'Stage' in sample_meta.columns:
        stage_counts = sample_meta['Stage'].astype(str).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7, 5))
        stage_counts.plot(kind='bar', ax=ax, color='#2ca02c', alpha=0.7)
        ax.set_title(f'{ic_name}: Sample Stage Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Stage', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        fig.tight_layout()
        plots['stage_distribution'] = plot_to_base64(fig)
    
    # Plot 4: Age histogram (if numeric)
    if 'Age' in sample_meta.columns:
        age_data = pd.to_numeric(sample_meta['Age'], errors='coerce').dropna()
        if not age_data.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(age_data, bins=15, color='#9467bd', alpha=0.7, edgecolor='white')
            ax.set_title(f'{ic_name}: Age Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Age', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.axvline(age_data.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {age_data.median():.1f}')
            ax.legend()
            fig.tight_layout()
            plots['age_distribution'] = plot_to_base64(fig)
    
    # Plot 5: Recurrence status (if available)
    if 'Recurrence.status' in sample_meta.columns:
        rec_counts = sample_meta['Recurrence.status'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(rec_counts.values, labels=rec_counts.index, autopct='%1.1f%%', startangle=90, colors=['#8c564b', '#e377c2'])
        ax.set_title(f'{ic_name}: Recurrence Status', fontsize=12, fontweight='bold')
        fig.tight_layout()
        plots['recurrence_status'] = plot_to_base64(fig)
    
    print(f"Generated {len(plots)} plots: {list(plots.keys())}")
    print("=== END DEBUG ===\n")
    return plots


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
