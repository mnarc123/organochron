"""Phase 1 — Aging Signatures: organ-specific aging gene identification.

Implements:
1.1  GTEx preprocessing (filtering, log-transform, covariate correction)
1.2  Age-associated gene identification (Spearman + robust regression)
1.3  WGCNA-like co-expression modules and GO enrichment
1.4  Tissue Aging Score (TAS) and Aging Acceleration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import HuberRegressor

from src.utils import (
    Timer,
    checkpoint_exists,
    ensure_dir,
    fdr_correction,
    load_checkpoint,
    load_parquet,
    save_checkpoint,
    save_json,
    save_parquet,
    set_seed,
)

PROCESSED = Path("data/processed")
RESULTS = Path("results")


# =========================================================================
# 1.1  Preprocessing
# =========================================================================


def _tissue_safe_name(tissue: str) -> str:
    """Convert tissue name to filesystem-safe string.

    Parameters
    ----------
    tissue : str

    Returns
    -------
    str
    """
    return tissue.replace(" ", "_").replace("-", "").replace("(", "").replace(")", "")


def preprocess_tissue(
    tissue: str,
    meta: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame | None:
    """Preprocess expression data for a single tissue.

    Steps:
    - Filter genes by minimum expression
    - Regress out sex and ischemia time covariates
    - Return residualised expression matrix

    Parameters
    ----------
    tissue : str
        GTEx tissue name.
    meta : pd.DataFrame
        GTEx metadata (must include SAMPID, SMTSD, AGE_MID, SEX, SMTSISCH).
    config : dict
        Global configuration.

    Returns
    -------
    pd.DataFrame or None
        Residualised log2(TPM+1) expression (genes × samples), or *None* on
        failure.
    """
    ts = _tissue_safe_name(tissue)
    expr_path = PROCESSED / f"gtex_expr_{ts}.parquet"
    if not expr_path.exists():
        logger.warning(f"Expression file not found for {tissue}")
        return None

    expr = load_parquet(expr_path)

    # Align metadata
    tissue_meta = meta[meta["SMTSD"] == tissue].copy()
    common_samples = sorted(set(expr.columns) & set(tissue_meta["SAMPID"]))
    if len(common_samples) < config["gtex"]["min_samples_per_tissue"]:
        logger.warning(f"Too few samples for {tissue}: {len(common_samples)}")
        return None

    expr = expr[common_samples]
    tissue_meta = tissue_meta.set_index("SAMPID").loc[common_samples]

    # Gene filtering: mean TPM > threshold in ≥ fraction of samples
    min_tpm = config["gtex"]["min_tpm_threshold"]
    min_frac = config["gtex"]["min_sample_fraction"]
    gene_mask = (expr > np.log2(min_tpm + 1)).mean(axis=1) >= min_frac
    expr = expr.loc[gene_mask]
    logger.info(f"  {tissue}: {gene_mask.sum()} genes pass filter ({len(common_samples)} samples)")

    # Covariate correction: regress out sex and ischemia time
    sex = pd.to_numeric(tissue_meta["SEX"], errors="coerce").fillna(1).values
    ischemia = pd.to_numeric(tissue_meta.get("SMTSISCH", pd.Series(dtype=float)),
                             errors="coerce").fillna(0).values
    covariates = np.column_stack([sex, ischemia, np.ones(len(sex))])

    # Residualise via OLS: expr_corrected = expr - covariates @ beta
    expr_arr = expr.values.astype(np.float64)
    beta, _, _, _ = np.linalg.lstsq(covariates, expr_arr.T, rcond=None)
    residuals = expr_arr - (covariates @ beta).T

    expr_corrected = pd.DataFrame(residuals, index=expr.index, columns=expr.columns)
    return expr_corrected


# =========================================================================
# 1.2  Age-associated gene identification
# =========================================================================


def _spearman_age_correlation(
    gene_expr: np.ndarray,
    ages: np.ndarray,
) -> tuple[float, float]:
    """Compute Spearman correlation between gene expression and age.

    Parameters
    ----------
    gene_expr : np.ndarray
    ages : np.ndarray

    Returns
    -------
    rho : float
    pvalue : float
    """
    rho, pval = stats.spearmanr(gene_expr, ages, nan_policy="omit")
    return float(rho), float(pval)


def _huber_regression_age(
    gene_expr: np.ndarray,
    ages: np.ndarray,
    sex: np.ndarray,
) -> tuple[float, float]:
    """Robust (Huber) regression of expression on age, adjusted for sex.

    Parameters
    ----------
    gene_expr : np.ndarray
    ages : np.ndarray
    sex : np.ndarray

    Returns
    -------
    coef_age : float
        Regression coefficient for age.
    pvalue : float
        Approximate p-value from t-test on the coefficient.
    """
    X = np.column_stack([ages, sex, np.ones_like(ages)])
    y = gene_expr
    mask = np.isfinite(y) & np.isfinite(ages)
    X, y = X[mask], y[mask]
    if len(y) < 10:
        return 0.0, 1.0
    try:
        model = HuberRegressor(max_iter=200)
        model.fit(X, y)
        coef = model.coef_[0]  # age coefficient

        # Approximate p-value via residual-based t-statistic
        residuals = y - model.predict(X)
        n, p = X.shape
        se = np.sqrt(np.sum(residuals**2) / (n - p))
        XtX_inv = np.linalg.pinv(X.T @ X)
        se_coef = se * np.sqrt(XtX_inv[0, 0])
        if se_coef > 0:
            t_stat = coef / se_coef
            pval = 2 * stats.t.sf(np.abs(t_stat), df=n - p)
        else:
            pval = 1.0
        return float(coef), float(pval)
    except Exception:
        return 0.0, 1.0


def _vectorized_spearman(expr_arr: np.ndarray, ages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Spearman correlation for all genes at once.

    Parameters
    ----------
    expr_arr : np.ndarray
        Expression matrix (genes × samples).
    ages : np.ndarray
        Numeric ages (samples,).

    Returns
    -------
    rhos : np.ndarray
    pvals : np.ndarray
    """
    from scipy.stats import rankdata

    n = expr_arr.shape[1]
    rank_ages = rankdata(ages)
    rank_expr = np.apply_along_axis(rankdata, 1, expr_arr)

    # Pearson on ranks = Spearman
    age_centered = rank_ages - rank_ages.mean()
    expr_centered = rank_expr - rank_expr.mean(axis=1, keepdims=True)

    cov = (expr_centered @ age_centered) / n
    std_age = age_centered.std()
    std_expr = expr_centered.std(axis=1)
    denom = std_age * std_expr
    denom[denom == 0] = 1e-30

    rhos = cov / denom

    # Approximate p-value via t-distribution
    t_stat = rhos * np.sqrt((n - 2) / (1 - rhos**2 + 1e-30))
    pvals = 2 * stats.t.sf(np.abs(t_stat), df=n - 2)

    return rhos, pvals


def _vectorized_ols_age(
    expr_arr: np.ndarray,
    ages: np.ndarray,
    sex: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized OLS regression of expression ~ age + sex for all genes.

    Parameters
    ----------
    expr_arr : np.ndarray
        (genes × samples).
    ages : np.ndarray
    sex : np.ndarray

    Returns
    -------
    coefs : np.ndarray
        Age coefficient per gene.
    pvals : np.ndarray
        P-value per gene.
    """
    n = expr_arr.shape[1]
    X = np.column_stack([ages, sex, np.ones(n)])  # (n, 3)
    # Beta = (X'X)^-1 X' Y'  →  (3, genes)
    XtX_inv = np.linalg.pinv(X.T @ X)
    betas = XtX_inv @ X.T @ expr_arr.T  # (3, genes)

    # Residuals
    residuals = expr_arr.T - X @ betas  # (n, genes)
    df = n - X.shape[1]
    rss = np.sum(residuals**2, axis=0)  # (genes,)
    mse = rss / df

    # Standard error of age coefficient (index 0)
    se_age = np.sqrt(mse * XtX_inv[0, 0])
    se_age[se_age == 0] = 1e-30

    coefs = betas[0, :]
    t_stat = coefs / se_age
    pvals = 2 * stats.t.sf(np.abs(t_stat), df=df)

    return coefs, pvals


def identify_aging_genes(
    expr: pd.DataFrame,
    ages: np.ndarray,
    sex: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Identify age-associated genes for one tissue.

    A gene is aging-associated if:
    - |Spearman rho| > threshold AND Spearman BH-adjusted p < 0.05, OR
    - OLS regression (age + sex) BH-adjusted p < 0.01

    Uses fully vectorized computation for speed (~100× faster than
    per-gene HuberRegressor).

    Parameters
    ----------
    expr : pd.DataFrame
        Residualised expression (genes × samples).
    ages : np.ndarray
        Numeric age midpoints per sample.
    sex : np.ndarray
        Sex coding per sample.
    config : dict

    Returns
    -------
    pd.DataFrame
        One row per gene with columns:
        ``gene, spearman_rho, spearman_pval, spearman_fdr, huber_coef,
        huber_pval, huber_fdr, direction, is_aging``.
    """
    rho_thresh = config["aging"]["spearman_rho_threshold"]
    sp_fdr = config["aging"]["spearman_fdr_threshold"]
    reg_fdr = config["aging"]["regression_fdr_threshold"]

    genes = expr.index.tolist()
    expr_arr = expr.values.astype(np.float64)
    ages_f = ages.astype(np.float64)
    sex_f = sex.astype(np.float64)

    # Vectorized Spearman
    sp_rhos, sp_pvals = _vectorized_spearman(expr_arr, ages_f)

    # Vectorized OLS (fast substitute for per-gene Huber)
    hub_coefs, hub_pvals = _vectorized_ols_age(expr_arr, ages_f, sex_f)

    _, sp_fdr_vals = fdr_correction(sp_pvals, alpha=sp_fdr)
    _, hub_fdr_vals = fdr_correction(hub_pvals, alpha=reg_fdr)

    # Criteria
    spearman_hit = (np.abs(sp_rhos) > rho_thresh) & (sp_fdr_vals < sp_fdr)
    huber_hit = hub_fdr_vals < reg_fdr
    is_aging = spearman_hit | huber_hit

    direction = np.where(sp_rhos > 0, "up", "down")

    result = pd.DataFrame({
        "gene": genes,
        "spearman_rho": sp_rhos,
        "spearman_pval": sp_pvals,
        "spearman_fdr": sp_fdr_vals,
        "huber_coef": hub_coefs,
        "huber_pval": hub_pvals,
        "huber_fdr": hub_fdr_vals,
        "direction": direction,
        "is_aging": is_aging,
    })
    return result


# =========================================================================
# 1.3  Simplified WGCNA (Python-native)
# =========================================================================


def wgcna_modules(
    expr: pd.DataFrame,
    ages: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute WGCNA-like co-expression modules and identify aging modules.

    Implementation:
    1. Select top genes by MAD
    2. Compute correlation matrix → adjacency via soft-thresholding
    3. Build TOM-like distance (approximated by 1 - |cor|^power)
    4. Hierarchical clustering → modules
    5. Module eigengenes → correlate with age

    Parameters
    ----------
    expr : pd.DataFrame
        Residualised expression (genes × samples).
    ages : np.ndarray
        Numeric age midpoints.
    config : dict

    Returns
    -------
    dict
        Keys: ``module_assignments`` (gene→module), ``module_eigengenes``
        (DataFrame), ``aging_modules`` (list of module IDs), ``module_age_cors``
        (dict module→(rho, pval)).
    """
    wgcna_cfg = config["wgcna"]
    n_top = min(wgcna_cfg["top_genes_mad"], expr.shape[0])

    # Select top genes by MAD
    mad = expr.subtract(expr.median(axis=1), axis=0).abs().median(axis=1)
    top_genes = mad.nlargest(n_top).index.tolist()
    sub_expr = expr.loc[top_genes]

    # Correlation matrix
    cor_mat = np.corrcoef(sub_expr.values)  # genes × genes
    cor_mat = np.nan_to_num(cor_mat, nan=0.0)

    # Determine soft-threshold power (scale-free fit)
    best_power = _pick_soft_threshold(cor_mat, r2_target=wgcna_cfg["scale_free_r2"])
    logger.info(f"  WGCNA soft-threshold power: {best_power}")

    # Adjacency (soft-thresholded)
    adj = np.abs(cor_mat) ** best_power

    # TOM (Topological Overlap Matrix) approximation
    np.fill_diagonal(adj, 0)
    # TOM_ij = (sum_u(a_iu * a_uj) + a_ij) / (min(k_i, k_j) + 1 - a_ij)
    numerator = adj @ adj + adj
    k = adj.sum(axis=1)
    k_min = np.minimum.outer(k, k)
    denominator = k_min + 1.0 - adj
    denominator[denominator < 1e-12] = 1e-12
    tom = numerator / denominator
    np.fill_diagonal(tom, 1.0)
    tom = np.clip(tom, 0, 1)

    # Distance = 1 - TOM
    tom_dist = 1.0 - tom

    # Hierarchical clustering using condensed distance form
    n_genes = tom_dist.shape[0]
    condensed = tom_dist[np.triu_indices(n_genes, k=1)]
    Z = linkage(condensed, method="average")

    # Dynamic tree cutting: try the configured cut height first,
    # then progressively lower it to get more modules
    merge_height = wgcna_cfg["merge_cut_height"]
    min_size = wgcna_cfg["min_module_size"]

    # Try configured height first
    labels = fcluster(Z, t=merge_height, criterion="distance")
    n_clusters = len(set(labels))

    # If distance-based cut produces too many clusters (near-singletons)
    # or too few, use adaptive approach
    target_modules = max(10, n_top // 200)  # ~25 for 5000 genes
    if n_clusters > n_top * 0.5 or n_clusters < 5:
        # Try percentile-based cuts on the linkage distances
        link_dists = Z[:, 2]
        for pct in [0.95, 0.90, 0.85, 0.80, 0.70, 0.60]:
            cut_h = np.percentile(link_dists, pct * 100)
            labels = fcluster(Z, t=cut_h, criterion="distance")
            n_clusters = len(set(labels))
            if 5 <= n_clusters <= n_top * 0.5:
                break

    if n_clusters > n_top * 0.5 or n_clusters < 5:
        # Force a target number of clusters
        labels = fcluster(Z, t=target_modules, criterion="maxclust")

    logger.info(f"  WGCNA clustering: {len(set(labels))} raw clusters")

    # Filter small modules
    module_counts = pd.Series(labels).value_counts()
    valid_modules = module_counts[module_counts >= min_size].index.tolist()

    module_assignments: dict[str, int] = {}
    for i, gene in enumerate(top_genes):
        if labels[i] in valid_modules:
            module_assignments[gene] = int(labels[i])
        else:
            module_assignments[gene] = 0  # unassigned

    # Module eigengenes (PC1 of each module)
    module_eigengenes: dict[int, np.ndarray] = {}
    for mod_id in valid_modules:
        mod_genes = [g for g, m in module_assignments.items() if m == mod_id]
        if len(mod_genes) < 3:
            continue
        mod_expr = sub_expr.loc[mod_genes].values.T  # samples × genes
        pca = PCA(n_components=1, random_state=42)
        me = pca.fit_transform(mod_expr).ravel()
        # Orient so that ME correlates positively with age
        if stats.spearmanr(me, ages)[0] < 0:
            me = -me
        module_eigengenes[mod_id] = me

    # Correlate MEs with age
    module_age_cors: dict[int, tuple[float, float]] = {}
    aging_modules: list[int] = []
    pval_thresh = wgcna_cfg["module_age_correlation_pvalue"]
    for mod_id, me in module_eigengenes.items():
        rho, pval = stats.spearmanr(me, ages)
        module_age_cors[mod_id] = (float(rho), float(pval))
        if pval < pval_thresh:
            aging_modules.append(mod_id)

    # Build ME DataFrame
    me_df = pd.DataFrame(module_eigengenes)

    logger.info(
        f"  WGCNA: {len(valid_modules)} modules, {len(aging_modules)} aging-associated"
    )

    return {
        "module_assignments": module_assignments,
        "module_eigengenes": me_df,
        "aging_modules": aging_modules,
        "module_age_cors": module_age_cors,
        "top_genes": top_genes,
    }


def _pick_soft_threshold(
    cor_mat: np.ndarray,
    r2_target: float = 0.8,
    powers: list[int] | None = None,
) -> int:
    """Select soft-threshold power for scale-free topology fit.

    Parameters
    ----------
    cor_mat : np.ndarray
        Absolute correlation matrix.
    r2_target : float
        Target R² for scale-free fit.
    powers : list[int], optional
        Candidate powers to test.

    Returns
    -------
    int
        Selected power.
    """
    if powers is None:
        powers = list(range(4, 21))  # min power=4 per Langfelder & Horvath
    n = cor_mat.shape[0]
    best_power = 6  # default fallback

    for power in powers:
        adj = np.abs(cor_mat) ** power
        np.fill_diagonal(adj, 0)
        k = adj.sum(axis=1)
        k = k[k > 0]
        if len(k) < 10:
            continue
        # Fit log(p(k)) ~ log(k) for scale-free check
        hist, bin_edges = np.histogram(np.log10(k + 1), bins=30)
        hist = hist.astype(float)
        hist[hist == 0] = np.nan
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mask = np.isfinite(np.log10(hist + 1))
        if mask.sum() < 3:
            continue
        try:
            slope, intercept, r, _, _ = stats.linregress(
                bin_centers[mask], np.log10(hist[mask] + 1)
            )
            if r**2 >= r2_target:
                best_power = power
                break
        except Exception:
            continue

    return best_power


# =========================================================================
# 1.4  Tissue Aging Score (TAS)
# =========================================================================


def compute_tissue_aging_scores(
    expr_dict: dict[str, pd.DataFrame],
    aging_genes_dict: dict[str, pd.DataFrame],
    meta: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compute Tissue Aging Score for each individual × tissue.

    TAS = PC1 of aging-associated genes, oriented to correlate positively
    with chronological age.  Aging Acceleration = residual of TAS ~ age.

    Parameters
    ----------
    expr_dict : dict
        tissue → expression DataFrame.
    aging_genes_dict : dict
        tissue → aging gene table.
    meta : pd.DataFrame
        GTEx metadata.
    config : dict

    Returns
    -------
    pd.DataFrame
        Columns: ``SUBJID, tissue, age, TAS, aging_acceleration``.
    """
    all_rows: list[dict] = []

    for tissue, expr in expr_dict.items():
        ag_df = aging_genes_dict.get(tissue)
        if ag_df is None or ag_df["is_aging"].sum() < 5:
            continue

        aging_genes = ag_df.loc[ag_df["is_aging"], "gene"].tolist()
        common_genes = [g for g in aging_genes if g in expr.index]
        if len(common_genes) < 5:
            continue

        sub = expr.loc[common_genes].T  # samples × genes
        pca = PCA(n_components=1, random_state=42)
        pc1 = pca.fit_transform(sub.values).ravel()

        # Get ages for these samples
        sample_meta = meta.set_index("SAMPID").loc[sub.index]
        ages = sample_meta["AGE_MID"].astype(float).values

        # Orient PC1 to correlate positively with age
        if stats.spearmanr(pc1, ages)[0] < 0:
            pc1 = -pc1

        # Z-score normalise
        pc1 = (pc1 - pc1.mean()) / (pc1.std() + 1e-8)

        # Aging acceleration: residual of TAS ~ age
        slope, intercept, _, _, _ = stats.linregress(ages, pc1)
        expected = slope * ages + intercept
        accel = pc1 - expected

        for i, sample_id in enumerate(sub.index):
            subjid = sample_meta.loc[sample_id, "SUBJID"]
            if isinstance(subjid, pd.Series):
                subjid = subjid.iloc[0]
            all_rows.append({
                "SUBJID": subjid,
                "tissue": tissue,
                "age": ages[i],
                "TAS": pc1[i],
                "aging_acceleration": accel[i],
            })

    tas_df = pd.DataFrame(all_rows)
    logger.info(f"TAS computed: {len(tas_df)} entries across {tas_df['tissue'].nunique()} tissues")
    return tas_df


# =========================================================================
# Master entry point
# =========================================================================


def compute_all(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full Phase 1 pipeline.

    Parameters
    ----------
    config : dict

    Returns
    -------
    dict
        Keys: ``meta, tissues, expr_dict, aging_genes_dict, wgcna_dict,
        tas_df, aging_acceleration_matrix``.
    """
    set_seed(42)

    # Check for checkpoint
    if checkpoint_exists("phase1_results"):
        logger.info("Loading Phase 1 from checkpoint")
        return load_checkpoint("phase1_results")

    # Load metadata
    meta = load_parquet(PROCESSED / "gtex_meta.parquet")
    tissues = sorted(meta["SMTSD"].unique())
    tissues = [t for t in config["gtex"]["tissues"] if t in tissues]

    expr_dict: dict[str, pd.DataFrame] = {}
    aging_genes_dict: dict[str, pd.DataFrame] = {}
    wgcna_dict: dict[str, dict] = {}

    for tissue in tissues:
        logger.info(f"Processing tissue: {tissue}")
        ts = _tissue_safe_name(tissue)

        # 1.1 Preprocess
        with Timer(f"  Preprocess {tissue}"):
            expr = preprocess_tissue(tissue, meta, config)
        if expr is None:
            continue

        # Get aligned metadata
        tissue_meta = meta[meta["SMTSD"] == tissue].set_index("SAMPID")
        common = sorted(set(expr.columns) & set(tissue_meta.index))
        expr = expr[common]
        tissue_meta = tissue_meta.loc[common]
        ages = tissue_meta["AGE_MID"].astype(float).values
        sex = pd.to_numeric(tissue_meta["SEX"], errors="coerce").fillna(1).values

        # 1.2 Aging genes
        with Timer(f"  Aging genes {tissue}"):
            ag_df = identify_aging_genes(expr, ages, sex, config)
        n_aging = ag_df["is_aging"].sum()
        n_up = ((ag_df["is_aging"]) & (ag_df["direction"] == "up")).sum()
        n_down = ((ag_df["is_aging"]) & (ag_df["direction"] == "down")).sum()
        logger.info(f"  {tissue}: {n_aging} aging genes ({n_up} up, {n_down} down)")

        # 1.3 WGCNA
        with Timer(f"  WGCNA {tissue}"):
            wgcna_res = wgcna_modules(expr, ages, config)

        expr_dict[tissue] = expr
        aging_genes_dict[tissue] = ag_df
        wgcna_dict[tissue] = wgcna_res

        # Save per-tissue results
        save_parquet(ag_df, RESULTS / "tables" / f"aging_genes_{ts}.parquet")

    # 1.4 Tissue Aging Scores
    with Timer("Tissue Aging Scores"):
        tas_df = compute_tissue_aging_scores(expr_dict, aging_genes_dict, meta, config)

    # Build aging acceleration matrix (subjects × tissues)
    aa_pivot = tas_df.pivot_table(
        index="SUBJID", columns="tissue", values="aging_acceleration", aggfunc="first"
    )
    logger.info(
        f"Aging Acceleration matrix: {aa_pivot.shape[0]} subjects × {aa_pivot.shape[1]} tissues"
    )

    # Save outputs
    save_parquet(tas_df, RESULTS / "tables" / "tissue_aging_scores.parquet")
    save_parquet(aa_pivot, RESULTS / "tables" / "aging_acceleration_matrix.parquet")

    results = {
        "meta": meta,
        "tissues": tissues,
        "expr_dict": expr_dict,
        "aging_genes_dict": aging_genes_dict,
        "wgcna_dict": wgcna_dict,
        "tas_df": tas_df,
        "aging_acceleration_matrix": aa_pivot,
    }

    # Summary statistics
    summary = {
        "n_tissues_processed": len(expr_dict),
        "tissues": list(expr_dict.keys()),
        "aging_genes_per_tissue": {
            t: int(df["is_aging"].sum()) for t, df in aging_genes_dict.items()
        },
        "wgcna_modules_per_tissue": {
            t: len(w.get("aging_modules", [])) for t, w in wgcna_dict.items()
        },
        "tas_subjects": int(aa_pivot.shape[0]),
        "tas_tissues": int(aa_pivot.shape[1]),
    }
    ensure_dir(RESULTS / "stats")
    save_json(summary, RESULTS / "stats" / "phase1_summary.json")

    save_checkpoint(results, "phase1_results")
    logger.info(f"Phase 1 complete: {len(expr_dict)} tissues processed")
    return results
