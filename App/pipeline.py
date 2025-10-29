import pandas as pd
from pathlib import Path

# Resolve data paths relative to this file to be robust across working directories
_BASE = Path(__file__).resolve().parent.parent
_ARCHIVE = _BASE / "Archive"

ic = pd.read_csv(_ARCHIVE / "independent_components.txt", sep="\t", index_col=0)
gsea = pd.read_csv(_ARCHIVE / "gsea_matrix.txt", sep=",", index_col=0)
mixing = pd.read_csv(_ARCHIVE / "mixing_matrix.txt", sep="\t", index_col=0)
genes = pd.read_csv(_ARCHIVE / "genomic_mapping.txt", sep="\t")
meta = pd.read_csv(_ARCHIVE / "sample_annotations.txt", sep="\t")

# default threshold for considering an IC "strong"
DEFAULT_THRESHOLD = 3

def filter_ic(threshold: float = DEFAULT_THRESHOLD):
    """Return a copy of the IC matrix where only strong loadings (|value| > threshold)
    are kept. Cells that don't meet the threshold are set to NaN and any rows/columns
    that become entirely NaN are dropped.

    Args:
        threshold: numeric cutoff for absolute value to consider a loading "strong".

    Returns:
        pandas.DataFrame: filtered IC matrix
    """
    ic_filtered = ic.copy()
    # keep values > threshold OR < -threshold (strong positive or strong negative)
    mask = (ic_filtered > threshold) | (ic_filtered < -threshold)
    ic_filtered = ic_filtered.where(mask)
    # drop ICs (columns) and genes (rows) that have no strong values
    ic_filtered = ic_filtered.dropna(axis=1, how='all').dropna(axis=0, how='all')
    return ic_filtered

def filter_gene_enrichment(threshold: float = DEFAULT_THRESHOLD):
    """Filter the GSEA matrix to keep only pathway scores with absolute value > threshold.
    Non-significant entries are set to NaN. Drops empty rows/columns.
    """
    gsea_filtered = gsea.copy()
    # keep values > threshold OR < -threshold
    mask = (gsea_filtered > threshold) | (gsea_filtered < -threshold)
    gsea_filtered = gsea_filtered.where(mask)
    gsea_filtered = gsea_filtered.dropna(axis=1, how='all').dropna(axis=0, how='all')
    return gsea_filtered

def filter_mixing_m(threshold: float = DEFAULT_THRESHOLD):
    """Filter the mixing matrix to keep only strong sample activations per IC
    (|value| > threshold). Non-strong entries are set to NaN and empty rows/cols removed.
    """
    mixing_filtered = mixing.copy()
    # keep values > threshold OR < -threshold
    mask = (mixing_filtered > threshold) | (mixing_filtered < -threshold)
    mixing_filtered = mixing_filtered.where(mask)
    mixing_filtered = mixing_filtered.dropna(axis=1, how='all').dropna(axis=0, how='all')
    return mixing_filtered

def find_related_ic(gene_symbol, threshold: float = DEFAULT_THRESHOLD):
    # Find the gene ID
    gene_row = genes[genes["SYMBOL"] == gene_symbol]
    if gene_row.empty:
        return f"No gene found with symbol {gene_symbol}"

    entrez = str(gene_row["ENTREZID"].values[0])

    # Get IC loadings for that gene
    ic_row = ic.loc[int(entrez)]
    
    # Filter for ICs where |loading| > threshold
    strong_ics = ic_row[(ic_row > threshold) | (ic_row < -threshold)]
    if strong_ics.empty:
        return f"No strong IC associations found for {gene_symbol}"

    results = []

    for ic_name in strong_ics.index:
        # Get top pathways from GSEA matrix
        # get GSEA hits for this IC using the same threshold
        if ic_name in gsea.columns:
            gsea_filtered = gsea[ic_name][(gsea[ic_name] > threshold) | (gsea[ic_name] < -threshold)].sort_values(ascending=False)
        else:
            gsea_filtered = pd.Series(dtype=float)
        gsea_hits = gsea_filtered.head(5)

        # Get top samples (most active) for this IC
        active_samples = mixing.loc[ic_name].sort_values(ascending=False).head(10)

        # Get metadata for these samples
        sample_meta = meta.loc[meta.index.isin(active_samples.index)]

        results.append({
            "IC": ic_name,
            "Loading": strong_ics[ic_name],
            "Top_Pathways": gsea_hits.to_dict(),
            "Top_Samples": sample_meta.to_dict(orient="records")
        })

    return results

gene_of_interest = "TP53"
related_ics = find_related_ic(gene_of_interest)
for ic_info in related_ics:
    print(f"IC: {ic_info['IC']}, Loading: {ic_info['Loading']}")
    print("Top Pathways:")
    for pathway, score in ic_info["Top_Pathways"].items():
        print(f"  {pathway}: {score}")
    print("Top Samples:")
    for sample in ic_info["Top_Samples"]:
        print(f"  {sample}")
    print("\n")
