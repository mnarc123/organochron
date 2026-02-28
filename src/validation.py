"""Phase 5 — Computational Validation.

Implements:
5.1  Internal validation (cross-validation, held-out prediction)
5.2  External validation (GenAge overlap, DrugAge overlap)
5.3  Confounding analysis (sex-stratified, cause-of-death filtering)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.model_selection import KFold

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
# 5.1  Internal validation
# =========================================================================


def cross_validate_graph(
    aging_data: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Cross-validate the causal graph structure.

    Splits subjects into k folds, recomputes the inter-tissue correlation
    structure for each fold, and measures Jaccard similarity of significant
    edges across folds.

    Parameters
    ----------
    aging_data : dict
    config : dict

    Returns
    -------
    dict
        ``jaccard_similarities`` (list of pairwise Jaccard values),
        ``mean_jaccard``, ``std_jaccard``.
    """
    set_seed(42)
    n_folds = config["validation"]["cv_folds"]
    aa_matrix = aging_data["aging_acceleration_matrix"]

    # Drop subjects with too many missing values
    min_tissues = config["causal"]["min_tissues_per_individual"]
    valid = aa_matrix.dropna(thresh=min_tissues)
    subjects = valid.index.tolist()
    n_subj = len(subjects)

    if n_subj < n_folds * 10:
        logger.warning(f"Only {n_subj} valid subjects — reducing folds")
        n_folds = min(n_folds, max(2, n_subj // 10))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_edges: list[set[tuple[str, str]]] = []

    alpha = config["causal"]["pc_alpha"]

    for fold_idx, (train_idx, _) in enumerate(kf.split(subjects)):
        fold_subjects = [subjects[i] for i in train_idx]
        fold_aa = valid.loc[fold_subjects]

        # Compute pairwise Spearman correlations between tissues
        tissues = fold_aa.columns.tolist()
        edges: set[tuple[str, str]] = set()
        for i in range(len(tissues)):
            for j in range(i + 1, len(tissues)):
                col_i = fold_aa[tissues[i]].dropna()
                col_j = fold_aa[tissues[j]].dropna()
                common = col_i.index.intersection(col_j.index)
                if len(common) < config["causal"]["min_shared_individuals"]:
                    continue
                rho, pval = stats.spearmanr(col_i[common], col_j[common])
                if pval < alpha and abs(rho) > 0.15:
                    edges.add((tissues[i], tissues[j]))

        fold_edges.append(edges)
        logger.debug(f"  Fold {fold_idx}: {len(edges)} significant edges")

    # Pairwise Jaccard
    jaccard_vals: list[float] = []
    for i in range(len(fold_edges)):
        for j in range(i + 1, len(fold_edges)):
            inter = len(fold_edges[i] & fold_edges[j])
            union = len(fold_edges[i] | fold_edges[j])
            jaccard_vals.append(inter / union if union > 0 else 0.0)

    result = {
        "n_folds": n_folds,
        "jaccard_similarities": jaccard_vals,
        "mean_jaccard": float(np.mean(jaccard_vals)) if jaccard_vals else 0.0,
        "std_jaccard": float(np.std(jaccard_vals)) if jaccard_vals else 0.0,
        "edges_per_fold": [len(e) for e in fold_edges],
    }
    logger.info(f"CV Jaccard stability: {result['mean_jaccard']:.3f} ± {result['std_jaccard']:.3f}")
    return result


def held_out_prediction(
    aging_data: dict[str, Any],
    integrated_graph: nx.DiGraph,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Predict Aging Acceleration in held-out subjects via graph propagation.

    For each fold, use the graph to predict a tissue's Aging Acceleration
    from the other tissues of the same individual, then correlate with
    the observed value.

    Parameters
    ----------
    aging_data : dict
    integrated_graph : nx.DiGraph
    config : dict

    Returns
    -------
    dict
        ``predictions`` (DataFrame), ``correlations`` (per-tissue r²).
    """
    set_seed(42)
    n_folds = config["validation"]["cv_folds"]
    aa_matrix = aging_data["aging_acceleration_matrix"]
    min_tissues = config["causal"]["min_tissues_per_individual"]
    valid = aa_matrix.dropna(thresh=min_tissues)

    # Build weight matrix from graph
    tissues = sorted(set(valid.columns) & set(integrated_graph.nodes()))
    n_tissues = len(tissues)
    tissue_idx = {t: i for i, t in enumerate(tissues)}
    W = np.zeros((n_tissues, n_tissues))
    for u, v, d in integrated_graph.edges(data=True):
        if u in tissue_idx and v in tissue_idx:
            W[tissue_idx[v], tissue_idx[u]] = d.get("weight", 0.0)

    # Normalise rows
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_norm = W / row_sums

    subjects = valid.index.tolist()
    kf = KFold(n_splits=min(n_folds, len(subjects) // 2), shuffle=True, random_state=42)

    all_predicted: list[dict] = []
    all_observed: list[dict] = []

    for _, test_idx in kf.split(subjects):
        test_subjects = [subjects[i] for i in test_idx]
        for subj in test_subjects:
            row = valid.loc[subj, tissues].values.astype(float)
            observed = row.copy()
            # Replace NaN with 0 for propagation
            row_clean = np.nan_to_num(row, nan=0.0)
            # For each tissue, predict from others using graph
            for t_idx in range(n_tissues):
                if np.isnan(observed[t_idx]):
                    continue
                # Mask this tissue and predict from neighbours
                masked = row_clean.copy()
                masked[t_idx] = 0.0
                pred = float(W_norm[t_idx, :] @ masked)
                all_predicted.append({
                    "subject": subj,
                    "tissue": tissues[t_idx],
                    "predicted": pred,
                })
                all_observed.append({
                    "subject": subj,
                    "tissue": tissues[t_idx],
                    "observed": float(observed[t_idx]),
                })

    pred_df = pd.DataFrame(all_predicted)
    obs_df = pd.DataFrame(all_observed)
    merged = pred_df.merge(obs_df, on=["subject", "tissue"])

    # Per-tissue correlation
    tissue_cors: dict[str, float] = {}
    for tissue in tissues:
        sub = merged[merged["tissue"] == tissue]
        if len(sub) > 10:
            pred_vals = sub["predicted"].values
            obs_vals = sub["observed"].values
            if np.std(pred_vals) > 1e-12 and np.std(obs_vals) > 1e-12:
                r, _ = stats.pearsonr(pred_vals, obs_vals)
                tissue_cors[tissue] = float(r)

    # Overall
    if len(merged) > 10:
        pred_all = merged["predicted"].values
        obs_all = merged["observed"].values
        if np.std(pred_all) > 1e-12 and np.std(obs_all) > 1e-12:
            overall_r, overall_p = stats.pearsonr(pred_all, obs_all)
        else:
            overall_r, overall_p = 0.0, 1.0
            logger.warning("Held-out: predicted values have zero variance")
    else:
        overall_r, overall_p = 0.0, 1.0

    result = {
        "predictions": merged,
        "tissue_correlations": tissue_cors,
        "overall_r": float(overall_r),
        "overall_p": float(overall_p),
        "n_predictions": len(merged),
    }
    logger.info(f"Held-out prediction: r = {overall_r:.3f} (p = {overall_p:.2e})")
    return result


# =========================================================================
# 5.2  External validation
# =========================================================================


def genage_enrichment(
    aging_data: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Test enrichment of aging-associated genes in GenAge.

    Uses Fisher's exact test per tissue.

    Parameters
    ----------
    aging_data : dict
    config : dict

    Returns
    -------
    dict
        Per-tissue Fisher test results and overall summary.
    """
    genage_path = PROCESSED / "genage_genes.csv"
    if genage_path.exists():
        genage_genes = set(pd.read_csv(genage_path)["gene"].dropna().unique())
    else:
        genage_genes = set()
        logger.warning("GenAge gene list not found")

    if not genage_genes:
        return {"error": "GenAge genes not available"}

    results: dict[str, dict] = {}
    for tissue, ag_df in aging_data["aging_genes_dict"].items():
        aging_genes = set(ag_df.loc[ag_df["is_aging"], "gene"])
        all_genes = set(ag_df["gene"])
        background = len(all_genes)

        # 2×2 table: aging & genage, aging & ~genage, ~aging & genage, ~aging & ~genage
        a = len(aging_genes & genage_genes)
        b = len(aging_genes - genage_genes)
        c = len(genage_genes - aging_genes)
        # Approximate d (assuming genage genes in background)
        genage_in_bg = len(genage_genes & all_genes)
        d = background - len(aging_genes) - genage_in_bg + a

        if a + b > 0 and a + c > 0:
            odds_ratio, pval = stats.fisher_exact([[a, b], [c, max(0, d)]], alternative="greater")
        else:
            odds_ratio, pval = 1.0, 1.0

        results[tissue] = {
            "n_aging_genes": len(aging_genes),
            "n_genage_overlap": a,
            "odds_ratio": float(odds_ratio),
            "pvalue": float(pval),
            "background_size": background,
        }

    # FDR correction across tissues
    tissues_list = list(results.keys())
    pvals = np.array([results[t]["pvalue"] for t in tissues_list])
    _, fdr_vals = fdr_correction(pvals)
    for i, t in enumerate(tissues_list):
        results[t]["fdr"] = float(fdr_vals[i])

    n_sig = sum(1 for t in tissues_list if results[t]["fdr"] < 0.05)
    logger.info(f"GenAge enrichment: {n_sig}/{len(tissues_list)} tissues significant (FDR < 0.05)")

    return {"per_tissue": results, "n_significant": n_sig}


def drugage_overlap(
    drug_results: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Test whether top drug candidates overlap with DrugAge more than expected.

    Uses a hypergeometric test.

    Parameters
    ----------
    drug_results : dict
    config : dict

    Returns
    -------
    dict
        Overlap statistics.
    """
    top_drugs = drug_results.get("top_drugs", pd.DataFrame())
    if top_drugs.empty:
        return {"error": "No drug results available"}

    da_path = PROCESSED / "drugage.parquet"
    if da_path.exists():
        drugage = load_parquet(da_path)
        drugage_names = set()
        for col in drugage.columns:
            if "compound" in col.lower() or "name" in col.lower() or "drug" in col.lower():
                drugage_names = set(drugage[col].dropna().str.lower().unique())
                break
    else:
        drugage_names = set()

    our_drugs = set(top_drugs["drug"].str.lower())
    overlap = our_drugs & drugage_names

    # Hypergeometric test
    # Population: union of all drugs in our scored pool + DrugAge
    # (any drug that could appear in either set)
    crs_df = drug_results.get("crs_df", pd.DataFrame())
    all_scored = set()
    if not crs_df.empty and "drug" in crs_df.columns:
        all_scored = set(crs_df["drug"].str.lower())
    if not all_scored:
        all_scored = our_drugs
    universe = all_scored | drugage_names
    M = len(universe)  # population size
    K = len(drugage_names & universe)  # successes in population
    n = len(our_drugs)  # draws
    k = len(overlap)  # observed successes

    # Guard: M must be >= max(K, n) for valid hypergeom
    if M >= K and M >= n and k > 0:
        pval = float(stats.hypergeom.sf(k - 1, M, K, n))
    else:
        pval = 1.0

    result = {
        "our_top_drugs": sorted(our_drugs),
        "drugage_drugs": sorted(drugage_names),
        "overlap_drugs": sorted(overlap),
        "n_overlap": len(overlap),
        "n_our": len(our_drugs),
        "n_drugage": len(drugage_names),
        "hypergeom_pvalue": pval,
        "population_size": M,
    }
    logger.info(
        f"DrugAge overlap: {len(overlap)}/{len(our_drugs)} "
        f"(hypergeometric p = {pval:.2e})"
    )
    return result


# =========================================================================
# 5.3  Confounding analysis
# =========================================================================


def sex_stratified_analysis(
    aging_data: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compare causal graph structure between males and females.

    Parameters
    ----------
    aging_data : dict
    config : dict

    Returns
    -------
    dict
        Jaccard similarity of edges, hub rank Spearman correlation.
    """
    meta = aging_data["meta"]
    aa_matrix = aging_data["aging_acceleration_matrix"]
    alpha = config["causal"]["pc_alpha"]
    min_obs = config["causal"]["min_shared_individuals"] // 2  # relax for subgroups

    results_by_sex: dict[str, set[tuple[str, str]]] = {}
    hub_ranks_by_sex: dict[str, dict[str, int]] = {}

    for sex_label, sex_code in [("male", "1"), ("female", "2")]:
        # Get subjects of this sex
        sex_subjects = meta[meta["SEX"] == sex_code]["SUBJID"].unique()
        sex_aa = aa_matrix.loc[aa_matrix.index.isin(sex_subjects)]

        # Drop columns/rows with insufficient data
        min_tissues = config["causal"]["min_tissues_per_individual"]
        sex_aa = sex_aa.dropna(thresh=min(min_tissues, sex_aa.shape[1] // 2))
        tissues = sex_aa.columns.tolist()

        edges: set[tuple[str, str]] = set()
        out_degree: dict[str, float] = {t: 0.0 for t in tissues}

        for i in range(len(tissues)):
            for j in range(i + 1, len(tissues)):
                col_i = sex_aa[tissues[i]].dropna()
                col_j = sex_aa[tissues[j]].dropna()
                common = col_i.index.intersection(col_j.index)
                if len(common) < min_obs:
                    continue
                rho, pval = stats.spearmanr(col_i[common], col_j[common])
                if pval < alpha and abs(rho) > 0.15:
                    edges.add((tissues[i], tissues[j]))
                    out_degree[tissues[i]] += abs(rho)
                    out_degree[tissues[j]] += abs(rho)

        results_by_sex[sex_label] = edges

        # Rank tissues by out-degree
        ranked = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)
        hub_ranks_by_sex[sex_label] = {t: rank for rank, (t, _) in enumerate(ranked)}

    # Jaccard similarity of edge sets
    male_edges = results_by_sex.get("male", set())
    female_edges = results_by_sex.get("female", set())
    inter = len(male_edges & female_edges)
    union = len(male_edges | female_edges)
    jaccard = inter / union if union > 0 else 0.0

    # Spearman correlation of hub ranks
    common_tissues = sorted(
        set(hub_ranks_by_sex.get("male", {}).keys())
        & set(hub_ranks_by_sex.get("female", {}).keys())
    )
    if len(common_tissues) >= 5:
        male_ranks = [hub_ranks_by_sex["male"][t] for t in common_tissues]
        female_ranks = [hub_ranks_by_sex["female"][t] for t in common_tissues]
        rank_rho, rank_p = stats.spearmanr(male_ranks, female_ranks)
    else:
        rank_rho, rank_p = 0.0, 1.0

    result = {
        "male_edges": len(male_edges),
        "female_edges": len(female_edges),
        "edge_jaccard": float(jaccard),
        "hub_rank_spearman_rho": float(rank_rho),
        "hub_rank_spearman_p": float(rank_p),
        "n_common_tissues": len(common_tissues),
    }
    logger.info(
        f"Sex-stratified analysis: edge Jaccard = {jaccard:.3f}, "
        f"hub rank rho = {rank_rho:.3f}"
    )
    return result


def permutation_test(
    aging_data: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Permutation test: shuffle age labels and check graph structure.

    Verifies that the real causal graph is significantly different from
    random (permuted) graphs.

    Parameters
    ----------
    aging_data : dict
    config : dict

    Returns
    -------
    dict
        Permutation p-value and summary statistics.
    """
    set_seed(42)
    n_perm = min(config["validation"]["permutation_n"], 200)  # Cap for speed
    aa_matrix = aging_data["aging_acceleration_matrix"]
    alpha = config["causal"]["pc_alpha"]
    min_obs = config["causal"]["min_shared_individuals"]

    # Compute real number of significant edges
    min_tissues = config["causal"]["min_tissues_per_individual"]
    valid = aa_matrix.dropna(thresh=min_tissues)
    tissues = valid.columns.tolist()

    def _count_edges(data: pd.DataFrame) -> int:
        count = 0
        for i in range(len(tissues)):
            for j in range(i + 1, len(tissues)):
                col_i = data[tissues[i]].dropna()
                col_j = data[tissues[j]].dropna()
                common = col_i.index.intersection(col_j.index)
                if len(common) < min_obs:
                    continue
                rho, pval = stats.spearmanr(col_i[common], col_j[common])
                if pval < alpha and abs(rho) > 0.15:
                    count += 1
        return count

    real_edges = _count_edges(valid)

    perm_edges: list[int] = []
    for p in range(n_perm):
        # Shuffle each tissue column independently
        perm_data = valid.copy()
        for col in tissues:
            non_null = perm_data[col].dropna()
            perm_data.loc[non_null.index, col] = np.random.permutation(non_null.values)
        perm_edges.append(_count_edges(perm_data))

    perm_edges_arr = np.array(perm_edges)
    p_value = float(np.mean(perm_edges_arr >= real_edges))

    result = {
        "real_n_edges": real_edges,
        "permutation_mean_edges": float(np.mean(perm_edges_arr)),
        "permutation_std_edges": float(np.std(perm_edges_arr)),
        "permutation_p_value": p_value,
        "n_permutations": n_perm,
    }
    logger.info(
        f"Permutation test: real={real_edges}, "
        f"perm={np.mean(perm_edges_arr):.1f}±{np.std(perm_edges_arr):.1f}, "
        f"p={p_value:.4f}"
    )
    return result


# =========================================================================
# Master entry point
# =========================================================================


def run_all(
    aging_data: dict[str, Any],
    integrated_graph: nx.DiGraph,
    hub_results: dict[str, Any],
    drug_results: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run all validation analyses.

    Parameters
    ----------
    aging_data : dict
    integrated_graph : nx.DiGraph
    hub_results : dict
    drug_results : dict
    config : dict

    Returns
    -------
    dict
        Comprehensive validation results.
    """
    set_seed(42)

    if checkpoint_exists("phase5_results"):
        logger.info("Loading Phase 5 from checkpoint")
        return load_checkpoint("phase5_results")

    # 5.1 Internal validation
    with Timer("Cross-validation"):
        cv_results = cross_validate_graph(aging_data, config)

    with Timer("Held-out prediction"):
        holdout_results = held_out_prediction(aging_data, integrated_graph, config)

    # 5.2 External validation
    with Timer("GenAge enrichment"):
        genage_results = genage_enrichment(aging_data, config)

    with Timer("DrugAge overlap"):
        drugage_results = drugage_overlap(drug_results, config)

    # 5.3 Confounding analysis
    with Timer("Sex-stratified analysis"):
        sex_results = sex_stratified_analysis(aging_data, config)

    with Timer("Permutation test"):
        perm_results = permutation_test(aging_data, config)

    # Save all results
    ensure_dir(RESULTS / "stats")
    validation_summary = {
        "cross_validation": {
            "mean_jaccard": cv_results["mean_jaccard"],
            "std_jaccard": cv_results["std_jaccard"],
            "n_folds": cv_results["n_folds"],
        },
        "held_out_prediction": {
            "overall_r": holdout_results["overall_r"],
            "overall_p": holdout_results["overall_p"],
            "n_predictions": holdout_results["n_predictions"],
        },
        "genage_enrichment": {
            "n_significant_tissues": genage_results.get("n_significant", 0),
        },
        "drugage_overlap": {
            "n_overlap": drugage_results.get("n_overlap", 0),
            "hypergeom_p": drugage_results.get("hypergeom_pvalue", 1.0),
        },
        "sex_stratified": {
            "edge_jaccard": sex_results["edge_jaccard"],
            "hub_rank_rho": sex_results["hub_rank_spearman_rho"],
        },
        "permutation_test": {
            "p_value": perm_results["permutation_p_value"],
            "real_edges": perm_results["real_n_edges"],
            "perm_mean_edges": perm_results["permutation_mean_edges"],
        },
    }
    save_json(validation_summary, RESULTS / "stats" / "phase5_summary.json")

    results = {
        "cv_results": cv_results,
        "holdout_results": holdout_results,
        "genage_results": genage_results,
        "drugage_results": drugage_results,
        "sex_results": sex_results,
        "permutation_results": perm_results,
    }

    save_checkpoint(results, "phase5_results")
    return results
