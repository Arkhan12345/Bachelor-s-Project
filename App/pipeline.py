import pandas as pd
from pathlib import Path


_BASE = Path(__file__).resolve().parent.parent
_ARCHIVE = _BASE / "Archive"

ic = pd.read_csv(_ARCHIVE / "independent_components.txt", sep="\t", index_col=0)
gsea = pd.read_csv(_ARCHIVE / "gsea_matrix.txt", sep=",", index_col=0)
mixing = pd.read_csv(_ARCHIVE / "mixing_matrix.txt", sep="\t", index_col=0)
genes = pd.read_csv(_ARCHIVE / "genomic_mapping.txt", sep="\t")
meta = pd.read_csv(_ARCHIVE / "sample_annotations.txt", sep="\t")


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

def filter_mixing_m(threshold: float = 3):
    """Filter the mixing matrix to keep only strong sample activations per IC
    (|value| > threshold). Non-strong entries are set to NaN and empty rows/cols removed.
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
    mixing_filtered = filter_mixing_m(threshold)

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
