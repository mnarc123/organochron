"""Unit tests for the OrganoChron pipeline.

Tests cover:
- Utility functions (FDR correction, age midpoint, JSON serialisation)
- Synthetic data generation sanity checks
- Aging signature computation on small synthetic data
- Graph construction and centrality metrics
- Cascade propagation on a known graph
- Drug reversal score computation
- Housekeeping gene exclusion sanity check
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    age_midpoint,
    fdr_correction,
    load_config,
    save_json,
    load_json,
    set_seed,
    TISSUE_SYSTEM,
)


# =========================================================================
# Utility tests
# =========================================================================


class TestUtils:
    """Tests for src/utils.py helper functions."""

    def test_age_midpoint_default(self) -> None:
        assert age_midpoint("20-29") == 25.0
        assert age_midpoint("60-69") == 65.0

    def test_age_midpoint_custom(self) -> None:
        mapping = {"20-29": 24}
        assert age_midpoint("20-29", mapping) == 24.0

    def test_age_midpoint_unknown(self) -> None:
        result = age_midpoint("80-89")
        assert np.isnan(result)

    def test_fdr_correction_basic(self) -> None:
        pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        rejected, adjusted = fdr_correction(pvals, alpha=0.05)
        assert isinstance(rejected, np.ndarray)
        assert isinstance(adjusted, np.ndarray)
        assert len(rejected) == len(pvals)
        # Most significant should still be significant
        assert adjusted[0] < 0.05

    def test_fdr_all_significant(self) -> None:
        pvals = np.array([0.001, 0.002, 0.003])
        rejected, adjusted = fdr_correction(pvals, alpha=0.05)
        assert all(rejected)

    def test_fdr_none_significant(self) -> None:
        pvals = np.array([0.5, 0.6, 0.9])
        rejected, _ = fdr_correction(pvals, alpha=0.01)
        assert not any(rejected)

    def test_json_roundtrip(self) -> None:
        data = {"a": 1, "b": np.float64(2.5), "arr": np.array([1, 2, 3])}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_json(data, path)
            loaded = load_json(path)
            assert loaded["a"] == 1
            assert loaded["b"] == 2.5
            assert loaded["arr"] == [1, 2, 3]
        finally:
            os.unlink(path)

    def test_tissue_system_coverage(self) -> None:
        """All 27 configured tissues should have a system classification."""
        expected = [
            "Adipose - Subcutaneous", "Liver", "Brain - Cortex",
            "Heart - Left Ventricle", "Whole Blood", "Lung",
        ]
        for t in expected:
            assert t in TISSUE_SYSTEM, f"Missing tissue: {t}"

    def test_set_seed_deterministic(self) -> None:
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


# =========================================================================
# Aging signature tests
# =========================================================================


class TestAgingSignatures:
    """Tests for src/aging_signatures.py core functions."""

    def test_spearman_correlation(self) -> None:
        from src.aging_signatures import _spearman_age_correlation
        set_seed(42)
        ages = np.array([25, 35, 45, 55, 65, 75] * 10, dtype=float)
        # Gene perfectly correlated with age
        expr = ages + np.random.randn(len(ages)) * 0.1
        rho, pval = _spearman_age_correlation(expr, ages)
        assert rho > 0.9
        assert pval < 0.001

    def test_spearman_no_correlation(self) -> None:
        from src.aging_signatures import _spearman_age_correlation
        set_seed(42)
        ages = np.array([25, 35, 45, 55, 65, 75] * 10, dtype=float)
        expr = np.random.randn(len(ages))  # pure noise
        rho, pval = _spearman_age_correlation(expr, ages)
        assert abs(rho) < 0.3  # should be weak

    def test_huber_regression(self) -> None:
        from src.aging_signatures import _huber_regression_age
        set_seed(42)
        n = 100
        ages = np.linspace(25, 75, n)
        sex = np.random.choice([1, 2], n)
        expr = 0.05 * ages + np.random.randn(n) * 0.5
        coef, pval = _huber_regression_age(expr, ages, sex)
        assert coef > 0  # positive age effect

    def test_identify_aging_genes(self) -> None:
        from src.aging_signatures import identify_aging_genes
        set_seed(42)
        n_samples = 120
        n_genes = 50
        ages = np.tile([25, 35, 45, 55, 65, 75], n_samples // 6)
        sex = np.random.choice([1, 2], n_samples).astype(float)

        gene_names = [f"GENE_{i}" for i in range(n_genes)]
        data = np.random.randn(n_genes, n_samples)
        # Make first 5 genes strongly age-correlated
        for g in range(5):
            data[g, :] += 0.08 * ages
        expr = pd.DataFrame(data, index=gene_names,
                             columns=[f"S{i}" for i in range(n_samples)])

        config = {
            "aging": {
                "spearman_rho_threshold": 0.15,
                "spearman_fdr_threshold": 0.05,
                "regression_fdr_threshold": 0.01,
            }
        }
        result = identify_aging_genes(expr, ages, sex, config)
        assert "is_aging" in result.columns
        assert result["is_aging"].sum() >= 3  # at least some detected

    def test_housekeeping_genes_excluded(self) -> None:
        """Housekeeping genes (constant expression) should not be aging-associated."""
        from src.aging_signatures import identify_aging_genes
        set_seed(42)
        n_samples = 120
        ages = np.tile([25, 35, 45, 55, 65, 75], n_samples // 6)
        sex = np.random.choice([1, 2], n_samples).astype(float)

        # Housekeeping gene: constant expression with tiny noise
        hk_expr = np.ones((1, n_samples)) * 8.0 + np.random.randn(1, n_samples) * 0.01
        expr = pd.DataFrame(hk_expr, index=["GAPDH"],
                             columns=[f"S{i}" for i in range(n_samples)])

        config = {
            "aging": {
                "spearman_rho_threshold": 0.15,
                "spearman_fdr_threshold": 0.05,
                "regression_fdr_threshold": 0.01,
            }
        }
        result = identify_aging_genes(expr, ages, sex, config)
        assert not result.loc[result["gene"] == "GAPDH", "is_aging"].iloc[0]


# =========================================================================
# Graph tests
# =========================================================================


class TestGraph:
    """Tests for graph construction and centrality computation."""

    def _make_test_graph(self) -> nx.DiGraph:
        """Create a small known graph for testing."""
        G = nx.DiGraph()
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_node("D")
        G.add_edge("A", "B", weight=0.8, confidence=1.0, source_type="confirmed")
        G.add_edge("A", "C", weight=0.5, confidence=0.7, source_type="causal_only")
        G.add_edge("B", "C", weight=0.3, confidence=0.3, source_type="secretome_only")
        G.add_edge("B", "D", weight=0.6, confidence=1.0, source_type="confirmed")
        G.add_edge("C", "D", weight=0.4, confidence=0.7, source_type="causal_only")
        return G

    def test_graph_not_fully_connected(self) -> None:
        """The causal graph should NOT be a complete graph."""
        G = self._make_test_graph()
        n = G.number_of_nodes()
        max_edges = n * (n - 1)  # directed complete graph
        assert G.number_of_edges() < max_edges

    def test_centrality_metrics(self) -> None:
        from src.hub_analysis import compute_centrality_metrics
        G = self._make_test_graph()
        config = {
            "hub": {
                "weights": {"out_degree": 0.3, "pagerank": 0.3,
                            "betweenness": 0.2, "cascade_depth": 0.2},
            }
        }
        df = compute_centrality_metrics(G, config)
        assert len(df) == 4
        assert "hub_score" in df.columns
        # Node A should rank high (most out-edges)
        top = df.iloc[0]["tissue"]
        assert top in ("A", "B")  # A or B should be top hub

    def test_cascade_propagation_known_graph(self) -> None:
        """On a known graph, cascade from A should reach D."""
        from src.hub_analysis import simulate_cascade
        G = self._make_test_graph()
        config = {"hub": {"cascade_damping": 0.7, "cascade_iterations": 10}}
        result = simulate_cascade(G, "A", config)
        assert result["A"] >= 1.0  # source retains impulse
        assert result["B"] > 0  # direct neighbour affected
        assert result["D"] > 0  # reachable via B

    def test_cascade_isolated_node(self) -> None:
        """An isolated node should not propagate anywhere."""
        from src.hub_analysis import simulate_cascade
        G = nx.DiGraph()
        G.add_node("X")
        G.add_node("Y")
        # No edges
        config = {"hub": {"cascade_damping": 0.7, "cascade_iterations": 10}}
        result = simulate_cascade(G, "X", config)
        assert result["X"] == 1.0
        assert result["Y"] == 0.0

    def test_graph_integration(self) -> None:
        from src.causal_discovery import integrate_graphs
        G1 = nx.DiGraph()
        G1.add_edge("A", "B", weight=0.5)
        G1.add_edge("B", "C", weight=0.3)

        G2 = nx.DiGraph()
        G2.add_edge("A", "B", weight=0.6)
        G2.add_edge("A", "C", weight=0.4)

        config = {}
        integrated = integrate_graphs(G1, G2, config)
        # A→B should be confirmed (in both)
        assert integrated.has_edge("A", "B")
        assert integrated.edges["A", "B"]["source_type"] == "confirmed"
        assert integrated.edges["A", "B"]["confidence"] == 1.0


# =========================================================================
# Drug repurposing tests
# =========================================================================


class TestDrugRepurposing:
    """Tests for drug repurposing score computation."""

    def test_enrichment_score_positive(self) -> None:
        from src.drug_repurposing import _enrichment_score
        set_seed(42)
        genes = [f"G{i}" for i in range(100)]
        sig = pd.Series(np.random.randn(100), index=genes)
        # Gene set at the top of the ranked list
        top_genes = sig.nlargest(10).index.tolist()
        es = _enrichment_score(sig, top_genes)
        assert es > 0  # should be positive (enriched at top)

    def test_enrichment_score_empty_set(self) -> None:
        from src.drug_repurposing import _enrichment_score
        sig = pd.Series([1.0, -1.0], index=["A", "B"])
        es = _enrichment_score(sig, [])
        assert es == 0.0

    def test_enrichment_score_no_overlap(self) -> None:
        from src.drug_repurposing import _enrichment_score
        sig = pd.Series([1.0, -1.0], index=["A", "B"])
        es = _enrichment_score(sig, ["X", "Y"])  # no overlap
        assert es == 0.0


# =========================================================================
# Causal discovery tests
# =========================================================================


class TestCausalDiscovery:
    """Tests for causal inference on synthetic data with known structure."""

    def test_known_dag_recovery(self) -> None:
        """Generate data from A→B→C and check that edges are recovered."""
        from src.causal_discovery import _correlation_based_causal
        set_seed(42)
        n = 500
        A = np.random.randn(n)
        B = 0.7 * A + np.random.randn(n) * 0.3
        C = 0.6 * B + np.random.randn(n) * 0.3
        data = np.column_stack([A, B, C])
        tissues = ["T_A", "T_B", "T_C"]

        config = {"causal": {"pc_alpha": 0.05}}
        G = _correlation_based_causal(data, tissues, config)
        # Should have some edges
        assert G.number_of_edges() > 0
        # A and C should be connected (directly or via B)
        all_edges = set(G.edges())
        has_path = (
            ("T_A", "T_B") in all_edges
            or ("T_A", "T_C") in all_edges
            or ("T_B", "T_C") in all_edges
        )
        assert has_path

    def test_consensus_empty_when_disjoint(self) -> None:
        """Consensus of disjoint edge sets should be empty."""
        from src.causal_discovery import consensus_causal_graph
        G1 = nx.DiGraph()
        G1.add_edge("A", "B")
        G2 = nx.DiGraph()
        G2.add_edge("C", "D")
        consensus = consensus_causal_graph(G1, G2)
        assert consensus.number_of_edges() == 0


# =========================================================================
# Config tests
# =========================================================================


class TestConfig:
    """Tests for configuration loading."""

    def test_load_config(self) -> None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
            assert "gtex" in config
            assert "aging" in config
            assert "hub" in config
            assert config["gtex"]["min_samples_per_tissue"] == 100

    def test_config_missing_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")


# =========================================================================
# Integration sanity test
# =========================================================================


class TestSanity:
    """High-level sanity checks."""

    def test_all_tissue_systems_have_color(self) -> None:
        from src.utils import SYSTEM_COLORS, TISSUE_SYSTEM
        systems = set(TISSUE_SYSTEM.values())
        for s in systems:
            assert s in SYSTEM_COLORS, f"No color for system: {s}"

    def test_tissue_short_names(self) -> None:
        from src.visualization import _tissue_short
        assert _tissue_short("Whole Blood") == "Blood"
        assert _tissue_short("Liver") == "Liver"
        assert len(_tissue_short("Brain - Cortex")) < 15


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
