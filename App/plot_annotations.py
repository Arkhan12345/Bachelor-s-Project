import os
from pathlib import Path
import sys
from typing import List, Optional

import pandas as pd

# Try to import matplotlib lazily and fail gracefully with guidance
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as e:  # pragma: no cover
    plt = None
    _matplotlib_import_error = e
else:
    _matplotlib_import_error = None

ARCHIVE_DIR = Path(__file__).resolve().parent.parent / "Archive"
INPUT_FILE = ARCHIVE_DIR / "sample_annotations.txt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"


def safe_filename(name: str) -> str:
    """Make a safe filename from a column name or category label."""
    keep = [c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name]
    # collapse consecutive underscores
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_.") or "unnamed"


def load_annotations(path: Path = INPUT_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find annotations file at: {path}")
    # Tab-separated; do not force index so it works regardless of header shape
    return pd.read_csv(path, sep="\t")


def ensure_output_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_categorical_counts(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    """For each categorical column in df, save a bar plot of value counts.

    Returns list of saved file paths.
    """
    if plt is None:
        raise RuntimeError(
            "matplotlib is required to create plots. Please install it with: pip install matplotlib"
        )

    saved = []
    # Consider object, category, and boolean as categorical
    cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)

    for col in cat_cols:
        counts = df[col].fillna("NA").astype(str).value_counts(dropna=False)
        # Avoid overly long plots by limiting to top 50 categories
        top_counts = counts.head(50)

        fig, ax = plt.subplots(figsize=(max(6, min(20, int(len(top_counts) * 0.6))), 6))
        top_counts.plot(kind="bar", ax=ax, color="#4c78a8")
        ax.set_title(f"Counts per category: {col}")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelrotation=45)
        # Align tick labels to the right for readability
        import matplotlib.pyplot as _plt  # local alias to use setp safely
        _plt.setp(ax.get_xticklabels(), ha="right")
        fig.tight_layout()

        out_path = out_dir / f"{safe_filename(col)}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)

    return saved


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Safely coerce a column to numeric, dropping NaNs."""
    s = pd.to_numeric(df[col], errors="coerce")
    return s.dropna()


def plot_histogram(df: pd.DataFrame, col: str, out_dir: Path, bins: int = 30) -> Path:
    s = _numeric_series(df, col)
    if s.empty:
        return out_dir / f"{safe_filename(col)}_hist_empty.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(s.values, bins=bins, color="#4c78a8", edgecolor="white")
    ax.set_title(f"Histogram: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    fig.tight_layout()
    out_path = out_dir / f"{safe_filename(col)}_hist.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_box_by_category(df: pd.DataFrame, num_col: str, cat_col: str, out_dir: Path) -> Path:
    s = _numeric_series(df, num_col)
    # Align with categorical labels for same index
    sub = df.loc[s.index]
    cats = sub[cat_col].astype(str).fillna("NA")
    groups = [s[cats == c].values for c in sorted(cats.unique())]
    labels = sorted(cats.unique())
    if len(groups) == 0:
        return out_dir / f"{safe_filename(num_col)}_by_{safe_filename(cat_col)}_box_empty.png"
    fig, ax = plt.subplots(figsize=(max(6, min(20, int(len(labels) * 0.6))), 6))
    ax.boxplot(groups, labels=labels, vert=True, patch_artist=True)
    ax.set_title(f"{num_col} by {cat_col} (box plot)")
    ax.set_xlabel(cat_col)
    ax.set_ylabel(num_col)
    ax.tick_params(axis="x", labelrotation=45)
    import matplotlib.pyplot as _plt
    _plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    out_path = out_dir / f"{safe_filename(num_col)}_by_{safe_filename(cat_col)}_box.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_violin_by_category(df: pd.DataFrame, num_col: str, cat_col: str, out_dir: Path) -> Path:
    s = _numeric_series(df, num_col)
    sub = df.loc[s.index]
    cats = sub[cat_col].astype(str).fillna("NA")
    labels = sorted(cats.unique())
    groups = [s[cats == c].values for c in labels]
    if len(groups) == 0:
        return out_dir / f"{safe_filename(num_col)}_by_{safe_filename(cat_col)}_violin_empty.png"
    fig, ax = plt.subplots(figsize=(max(6, min(20, int(len(labels) * 0.6))), 6))
    parts = ax.violinplot(groups, showmeans=True, showextrema=True, showmedians=True)
    # color violins
    for pc in parts['bodies']:
        pc.set_facecolor('#4c78a8')
        pc.set_alpha(0.7)
    ax.set_title(f"{num_col} by {cat_col} (violin)")
    ax.set_xlabel(cat_col)
    ax.set_ylabel(num_col)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    fig.tight_layout()
    out_path = out_dir / f"{safe_filename(num_col)}_by_{safe_filename(cat_col)}_violin.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_binary_counts(df: pd.DataFrame, col: str, out_dir: Path) -> List[Path]:
    s = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
    counts = s.value_counts().sort_index()
    files: List[Path] = []
    # Bar
    fig, ax = plt.subplots(figsize=(6, 5))
    counts.plot(kind="bar", ax=ax, color="#4c78a8")
    ax.set_title(f"Counts: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    fig.tight_layout()
    out_bar = out_dir / f"{safe_filename(col)}_counts.png"
    fig.savefig(out_bar, dpi=150)
    plt.close(fig)
    files.append(out_bar)
    # Pie
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values, labels=[str(k) for k in counts.index], autopct='%1.1f%%', startangle=90)
    ax.set_title(f"Distribution: {col}")
    fig.tight_layout()
    out_pie = out_dir / f"{safe_filename(col)}_pie.png"
    fig.savefig(out_pie, dpi=150)
    plt.close(fig)
    files.append(out_pie)
    return files


def plot_grouped_binary_counts(df: pd.DataFrame, bin_col: str, group_col: str, out_dir: Path) -> Path:
    s = pd.to_numeric(df[bin_col], errors="coerce").dropna().astype(int)
    sub = df.loc[s.index]
    group = sub[group_col].astype(str).fillna("NA")
    # Build counts matrix: rows=group categories, cols=0/1
    groups = sorted(group.unique())
    levels = [0, 1]
    import numpy as np
    matrix = np.zeros((len(groups), len(levels)), dtype=int)
    for i, g in enumerate(groups):
        vals = s[group == g]
        vc = vals.value_counts()
        for j, lvl in enumerate(levels):
            matrix[i, j] = int(vc.get(lvl, 0))
    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(max(7, min(20, int(len(groups) * 0.7))), 6))
    x = np.arange(len(groups))
    width = 0.35
    ax.bar(x - width/2, matrix[:, 0], width, label='0')
    ax.bar(x + width/2, matrix[:, 1], width, label='1')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.set_title(f"{bin_col} grouped by {group_col}")
    ax.set_xlabel(group_col)
    ax.set_ylabel("Count")
    ax.legend(title=bin_col)
    fig.tight_layout()
    out_path = out_dir / f"{safe_filename(bin_col)}_by_{safe_filename(group_col)}_grouped_counts.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    # Correlation on numeric columns only
    num_df = df.select_dtypes(include=["number"])  # includes int/float
    if num_df.empty:
        return out_dir / "correlation_heatmap_empty.png"
    corr = num_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title('Numeric feature correlation')
    fig.colorbar(cax, ax=ax, shrink=0.8, label='r')
    fig.tight_layout()
    out_path = out_dir / "correlation_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_numeric_and_ordinal(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    saved: List[Path] = []
    cols = set(df.columns)
    # Age: Histogram, box by Type or Grade
    if 'Age' in cols:
        saved.append(plot_histogram(df, 'Age', out_dir))
        if 'Type' in cols:
            saved.append(plot_box_by_category(df, 'Age', 'Type', out_dir))
        if 'Grade' in cols:
            saved.append(plot_box_by_category(df, 'Age', 'Grade', out_dir))
    # Stage: Bar, box vs OS/PFS
    if 'Stage' in cols:
        # Bar counts (treat as categorical labels)
        counts = df['Stage'].astype(str).fillna('NA').value_counts()
        fig, ax = plt.subplots(figsize=(6, 5))
        counts.sort_index().plot(kind='bar', ax=ax, color='#4c78a8')
        ax.set_title('Counts: Stage')
        ax.set_xlabel('Stage')
        ax.set_ylabel('Count')
        fig.tight_layout()
        out_path = out_dir / 'Stage_counts.png'
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)
        # Box vs OS, PFS
        if 'OS' in cols:
            saved.append(plot_box_by_category(df, 'OS', 'Stage', out_dir))
        if 'PFS' in cols:
            saved.append(plot_box_by_category(df, 'PFS', 'Stage', out_dir))
    # Grade: Bar, violin vs OS/PFS
    if 'Grade' in cols:
        counts = df['Grade'].astype(str).fillna('NA').value_counts()
        fig, ax = plt.subplots(figsize=(6, 5))
        counts.sort_index().plot(kind='bar', ax=ax, color='#4c78a8')
        ax.set_title('Counts: Grade')
        ax.set_xlabel('Grade')
        ax.set_ylabel('Count')
        fig.tight_layout()
        out_path = out_dir / 'Grade_counts.png'
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)
        if 'OS' in cols:
            saved.append(plot_violin_by_category(df, 'OS', 'Grade', out_dir))
        if 'PFS' in cols:
            saved.append(plot_violin_by_category(df, 'PFS', 'Grade', out_dir))
    # PFS, OS: Histogram, box by Type or Debulking
    for num in ('PFS', 'OS'):
        if num in cols:
            saved.append(plot_histogram(df, num, out_dir))
            if 'Type' in cols:
                saved.append(plot_box_by_category(df, num, 'Type', out_dir))
            if 'Debulking' in cols:
                saved.append(plot_box_by_category(df, num, 'Debulking', out_dir))
    # PFS.binary, OS.binary: count + pie
    for b in ('PFS.binary', 'OS.binary'):
        if b in cols:
            saved.extend(plot_binary_counts(df, b, out_dir))
    # Platinum, Taxol, Neo.adjuvant: grouped by Recurrence.status
    for b in ('Platinum', 'Taxol', 'Neo.adjuvant'):
        if b in cols:
            saved.extend(plot_binary_counts(df, b, out_dir))
            if 'Recurrence.status' in cols:
                saved.append(plot_grouped_binary_counts(df, b, 'Recurrence.status', out_dir))
    # Optional: correlation heatmap
    saved.append(plot_correlation_heatmap(df, out_dir))
    return saved


def main(output_dir: Optional[str] = None) -> int:
    try:
        df = load_annotations()
    except Exception as e:
        print(f"Failed to load annotations: {e}")
        return 1

    out_dir = ensure_output_dir(Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR)

    if plt is None:
        # Provide clear guidance without crashing
        print(
            "matplotlib is not available. Install it to generate plots:\n"
            "  pip install matplotlib\n\n"
            f"Once installed, re-run: python {Path(__file__).name}"
        )
        return 0

    try:
        saved = plot_categorical_counts(df, out_dir)
        # Also create numeric/ordinal plots per the requested table
        saved += plot_numeric_and_ordinal(df, out_dir)
    except Exception as e:
        print(f"Failed to generate plots: {e}")
        return 1

    print(f"Generated {len(saved)} plot(s) in: {out_dir}")
    for p in saved:
        print(f" - {p}")
    return 0


if __name__ == "__main__":
    # Optional first arg: output directory
    out_arg = sys.argv[1] if len(sys.argv) > 1 else None
    raise SystemExit(main(out_arg))
