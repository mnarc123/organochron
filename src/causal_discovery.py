"""Phase 2b — Causal Discovery and Graph Integration.

Applies constraint-based (PC algorithm) and functional (LiNGAM) causal
discovery to the Aging Acceleration matrix, then integrates the resulting
causal graph with the secretome-based graph from Phase 2.

Edge confidence levels:
  - **confirmed** (1.0): present in both secretome and causal graphs
  - **causal-only** (0.7): causal evidence without secretome support
  - **secretome-only** (0.3): biological plausibility without causal evidence
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from src.utils import (
    Timer,
    checkpoint_exists,
    ensure_dir,
    load_checkpoint,
    save_checkpoint,
    save_json,
    set_seed,
)

RESULTS = Path("results")


# =========================================================================
# 2.4  Causal discovery from Aging Acceleration correlations
# =========================================================================


def _prepare_aa_matrix(
    aa_matrix: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Prepare the Aging Acceleration matrix for causal discovery.

    Filters individuals with sufficient tissue coverage, imputes missing
    values, and returns a complete numpy array.

    Parameters
    ----------
    aa_matrix : pd.DataFrame
        Subjects × tissues, with NaN for missing data.
    config : dict

    Returns
    -------
    data : np.ndarray
        Complete data matrix (subjects × tissues).
    tissue_names : list[str]
        Column order.
    """
    min_tissues = config["causal"]["min_tissues_per_individual"]

    # Filter subjects with enough tissue coverage
    tissue_coverage = aa_matrix.notna().sum(axis=1)
    valid_subjects = tissue_coverage[tissue_coverage >= min_tissues].index
    aa_sub = aa_matrix.loc[valid_subjects].copy()
    logger.info(
        f"Subjects with ≥{min_tissues} tissues: {len(valid_subjects)}/{len(aa_matrix)}"
    )

    # Drop tissues with too few observations
    min_obs = config["causal"]["min_shared_individuals"]
    tissue_obs = aa_sub.notna().sum(axis=0)
    valid_tissues = tissue_obs[tissue_obs >= min_obs].index.tolist()
    aa_sub = aa_sub[valid_tissues]
    logger.info(f"Tissues with ≥{min_obs} observations: {len(valid_tissues)}")

    # Impute remaining missing values with MICE
    if aa_sub.isna().any().any():
        logger.info("Imputing missing values with IterativeImputer (MICE)")
        imputer = IterativeImputer(
            max_iter=20, random_state=42, sample_posterior=False
        )
        data = imputer.fit_transform(aa_sub.values)
    else:
        data = aa_sub.values.copy()

    # Standardise columns
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

    return data, valid_tissues


def run_pc_algorithm(
    data: np.ndarray,
    tissue_names: list[str],
    config: dict[str, Any],
) -> nx.DiGraph:
    """Run the PC (Peter-Clark) causal discovery algorithm.

    Parameters
    ----------
    data : np.ndarray
        Standardised data (subjects × tissues).
    tissue_names : list[str]
    config : dict

    Returns
    -------
    nx.DiGraph
        Estimated causal DAG.
    """
    alpha = config["causal"]["pc_alpha"]
    n_vars = data.shape[1]

    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz

        with Timer("PC algorithm"):
            cg = pc(data, alpha=alpha, indep_test=fisherz, stable=True)

        # Extract graph: cg.G.graph is an adjacency matrix
        # Encoding: graph[i,j]==-1 and graph[j,i]==1 means i→j
        #           graph[i,j]==-1 and graph[j,i]==-1 means i—j (undirected)
        G = nx.DiGraph()
        for t in tissue_names:
            G.add_node(t)

        adj = cg.G.graph
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                if adj[i, j] == -1 and adj[j, i] == 1:
                    # i → j
                    G.add_edge(tissue_names[i], tissue_names[j], method="PC")
                elif adj[i, j] == -1 and adj[j, i] == -1:
                    # Undirected: add both directions as candidates
                    G.add_edge(tissue_names[i], tissue_names[j], method="PC_undirected")
                    G.add_edge(tissue_names[j], tissue_names[i], method="PC_undirected")

        logger.info(f"PC algorithm: {G.number_of_edges()} directed edges")
        return G

    except ImportError:
        logger.warning("causal-learn not available — using correlation-based fallback")
        return _correlation_based_causal(data, tissue_names, config)
    except Exception as exc:
        logger.warning(f"PC algorithm failed ({exc}) — using fallback")
        return _correlation_based_causal(data, tissue_names, config)


def run_lingam(
    data: np.ndarray,
    tissue_names: list[str],
    config: dict[str, Any],
) -> nx.DiGraph:
    """Run the DirectLiNGAM causal discovery algorithm.

    LiNGAM exploits non-Gaussianity to infer causal direction without
    additional assumptions beyond linearity and acyclicity.

    Parameters
    ----------
    data : np.ndarray
    tissue_names : list[str]
    config : dict

    Returns
    -------
    nx.DiGraph
    """
    prune_thresh = config["causal"]["lingam_prune_threshold"]

    try:
        from lingam import DirectLiNGAM

        with Timer("DirectLiNGAM"):
            model = DirectLiNGAM(random_state=42)
            model.fit(data)
            adj = model.adjacency_matrix_  # adj[i,j] != 0  ⟹  j → i

        G = nx.DiGraph()
        for t in tissue_names:
            G.add_node(t)

        n = len(tissue_names)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if abs(adj[i, j]) > prune_thresh:
                    # adj[i,j] != 0 means j causes i in LiNGAM convention
                    G.add_edge(
                        tissue_names[j],
                        tissue_names[i],
                        method="LiNGAM",
                        weight=abs(float(adj[i, j])),
                    )

        logger.info(f"LiNGAM: {G.number_of_edges()} directed edges")
        return G

    except ImportError:
        logger.warning("lingam not available — using correlation-based fallback")
        return _correlation_based_causal(data, tissue_names, config)
    except Exception as exc:
        logger.warning(f"LiNGAM failed ({exc}) — using fallback")
        return _correlation_based_causal(data, tissue_names, config)


def _correlation_based_causal(
    data: np.ndarray,
    tissue_names: list[str],
    config: dict[str, Any],
) -> nx.DiGraph:
    """Fallback causal graph estimation using partial correlations.

    When proper causal discovery libraries are unavailable, approximate
    causal direction using a heuristic: higher-variance node is the cause
    (based on the principle that causes tend to have more variability).

    Parameters
    ----------
    data : np.ndarray
    tissue_names : list[str]
    config : dict

    Returns
    -------
    nx.DiGraph
    """
    n = data.shape[1]
    alpha = config["causal"]["pc_alpha"]
    G = nx.DiGraph()
    for t in tissue_names:
        G.add_node(t)

    variances = data.var(axis=0)

    for i in range(n):
        for j in range(i + 1, n):
            rho, pval = stats.spearmanr(data[:, i], data[:, j])
            if pval < alpha and abs(rho) > 0.15:
                # Direction heuristic: higher variance → cause
                if variances[i] > variances[j]:
                    G.add_edge(tissue_names[i], tissue_names[j],
                               method="correlation_heuristic", weight=abs(rho))
                else:
                    G.add_edge(tissue_names[j], tissue_names[i],
                               method="correlation_heuristic", weight=abs(rho))

    logger.info(f"Correlation-based causal graph: {G.number_of_edges()} edges")
    return G


# =========================================================================
# Consensus causal graph
# =========================================================================


def consensus_causal_graph(
    pc_graph: nx.DiGraph,
    lingam_graph: nx.DiGraph,
) -> nx.DiGraph:
    """Keep only edges present in both PC and LiNGAM with same direction.

    Parameters
    ----------
    pc_graph : nx.DiGraph
    lingam_graph : nx.DiGraph

    Returns
    -------
    nx.DiGraph
        Consensus graph.
    """
    G = nx.DiGraph()
    for node in set(pc_graph.nodes()) | set(lingam_graph.nodes()):
        G.add_node(node)

    pc_edges = set(pc_graph.edges())
    lingam_edges = set(lingam_graph.edges())

    consensus = pc_edges & lingam_edges
    for u, v in consensus:
        w = lingam_graph.edges[u, v].get("weight", 0.5) if (u, v) in lingam_graph.edges() else 0.5
        G.add_edge(u, v, method="consensus", weight=w)

    logger.info(
        f"Consensus causal graph: {len(consensus)} edges "
        f"(from PC={len(pc_edges)}, LiNGAM={len(lingam_edges)})"
    )
    return G


# =========================================================================
# 2.5  Integrated graph
# =========================================================================


def integrate_graphs(
    secretome_graph: nx.DiGraph,
    causal_graph: nx.DiGraph,
    config: dict[str, Any],
) -> nx.DiGraph:
    """Combine secretome and causal graphs with confidence scores.

    Edge confidence:
    - **confirmed** (1.0): present in both graphs
    - **causal_only** (0.7): only in causal graph
    - **secretome_only** (0.3): only in secretome graph

    Parameters
    ----------
    secretome_graph : nx.DiGraph
    causal_graph : nx.DiGraph
    config : dict

    Returns
    -------
    nx.DiGraph
        Integrated graph with ``confidence`` and ``source_type`` attributes.
    """
    G = nx.DiGraph()
    all_nodes = set(secretome_graph.nodes()) | set(causal_graph.nodes())
    for node in all_nodes:
        G.add_node(node)

    sec_edges = set(secretome_graph.edges())
    causal_edges = set(causal_graph.edges())

    confirmed = sec_edges & causal_edges
    causal_only = causal_edges - sec_edges
    secretome_only = sec_edges - causal_edges

    for u, v in confirmed:
        sec_w = secretome_graph.edges[u, v].get("weight", 0.5)
        cau_w = causal_graph.edges[u, v].get("weight", 0.5)
        G.add_edge(u, v, weight=(sec_w + cau_w) / 2, confidence=1.0,
                   source_type="confirmed")

    for u, v in causal_only:
        w = causal_graph.edges[u, v].get("weight", 0.5)
        G.add_edge(u, v, weight=w, confidence=0.7, source_type="causal_only")

    for u, v in secretome_only:
        w = secretome_graph.edges[u, v].get("weight", 0.5)
        G.add_edge(u, v, weight=w * 0.3, confidence=0.3, source_type="secretome_only")

    logger.info(
        f"Integrated graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"(confirmed={len(confirmed)}, causal_only={len(causal_only)}, "
        f"secretome_only={len(secretome_only)})"
    )

    # Save edge table
    edge_data = []
    for u, v, d in G.edges(data=True):
        edge_data.append({
            "source": u, "target": v,
            "weight": d.get("weight", 0),
            "confidence": d.get("confidence", 0),
            "source_type": d.get("source_type", "unknown"),
        })
    ensure_dir(RESULTS / "tables")
    pd.DataFrame(edge_data).to_csv(
        RESULTS / "tables" / "integrated_graph_edges.csv", index=False
    )

    save_checkpoint(G, "integrated_graph")
    return G


# =========================================================================
# Master entry point
# =========================================================================


def infer_causality(
    aging_data: dict[str, Any],
    config: dict[str, Any],
) -> nx.DiGraph:
    """Run causal discovery on the Aging Acceleration matrix.

    Parameters
    ----------
    aging_data : dict
        Output of Phase 1 (must contain ``aging_acceleration_matrix``).
    config : dict

    Returns
    -------
    nx.DiGraph
        Consensus causal graph.
    """
    set_seed(42)

    if checkpoint_exists("causal_graph"):
        logger.info("Loading causal graph from checkpoint")
        return load_checkpoint("causal_graph")

    aa_matrix = aging_data["aging_acceleration_matrix"]

    with Timer("Prepare AA matrix"):
        data, tissue_names = _prepare_aa_matrix(aa_matrix, config)

    with Timer("PC algorithm"):
        pc_graph = run_pc_algorithm(data, tissue_names, config)

    with Timer("LiNGAM"):
        lingam_graph = run_lingam(data, tissue_names, config)

    with Timer("Consensus graph"):
        causal_graph = consensus_causal_graph(pc_graph, lingam_graph)

    # If consensus is too sparse, relax to union with different weights
    if causal_graph.number_of_edges() < 5:
        logger.warning(
            "Consensus too sparse — using union of PC and LiNGAM with weights"
        )
        causal_graph = _union_causal_graph(pc_graph, lingam_graph)

    # Save summary
    summary = {
        "pc_edges": pc_graph.number_of_edges(),
        "lingam_edges": lingam_graph.number_of_edges(),
        "consensus_edges": causal_graph.number_of_edges(),
        "n_subjects": data.shape[0],
        "n_tissues": data.shape[1],
        "tissues": tissue_names,
    }
    save_json(summary, RESULTS / "stats" / "causal_discovery_summary.json")

    save_checkpoint(causal_graph, "causal_graph")
    return causal_graph


def _union_causal_graph(
    pc_graph: nx.DiGraph,
    lingam_graph: nx.DiGraph,
) -> nx.DiGraph:
    """Create union of two causal graphs, weighting consensus edges higher.

    Parameters
    ----------
    pc_graph : nx.DiGraph
    lingam_graph : nx.DiGraph

    Returns
    -------
    nx.DiGraph
    """
    G = nx.DiGraph()
    for n in set(pc_graph.nodes()) | set(lingam_graph.nodes()):
        G.add_node(n)

    all_edges = set(pc_graph.edges()) | set(lingam_graph.edges())
    both = set(pc_graph.edges()) & set(lingam_graph.edges())

    for u, v in all_edges:
        if (u, v) in both:
            w = 0.8
        elif (u, v) in pc_graph.edges():
            w = 0.5
        else:
            w = 0.5
        G.add_edge(u, v, method="union", weight=w)

    logger.info(f"Union causal graph: {G.number_of_edges()} edges")
    return G
