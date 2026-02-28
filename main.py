#!/usr/bin/env python3
"""OrganoChron — Pipeline Orchestrator.

Runs all phases sequentially:
  Phase 0: Data acquisition
  Phase 1: Aging signatures
  Phase 2: Inter-organ secretome network
  Phase 2b: Causal discovery & graph integration
  Phase 3: Pacemaker organ identification
  Phase 4: Cascade drug repurposing
  Phase 5: Computational validation
  Phase 6: Figure generation

Usage:
    python main.py                     # Run full pipeline
    python main.py --phase 3           # Start from phase 3 (loads checkpoints)
    python main.py --synthetic         # Force synthetic data mode
    python main.py --config path.yaml  # Custom config file
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from src.utils import (
    Timer,
    checkpoint_exists,
    ensure_dir,
    load_checkpoint,
    load_config,
    save_json,
    set_seed,
    setup_logging,
)

app = typer.Typer(add_completion=False)


@app.command()
def run(
    config_path: str = typer.Option("config/config.yaml", "--config", "-c",
                                     help="Path to YAML configuration file."),
    start_phase: int = typer.Option(0, "--phase", "-p",
                                     help="Phase to start from (0–6). Earlier checkpoints are loaded."),
    synthetic: bool = typer.Option(False, "--synthetic", "-s",
                                    help="Force synthetic data mode even if real data exists."),
    skip_figures: bool = typer.Option(False, "--skip-figures",
                                       help="Skip figure generation (Phase 6)."),
) -> None:
    """Run the OrganoChron pipeline.

    Parameters
    ----------
    config_path : str
    start_phase : int
    synthetic : bool
    skip_figures : bool
    """
    t0 = time.perf_counter()

    # --- Setup ---
    setup_logging()
    config = load_config(config_path)
    if synthetic:
        config.setdefault("pipeline", {})["use_synthetic_fallback"] = True
    seed = config.get("pipeline", {}).get("random_seed", 42)
    set_seed(seed)

    # Ensure output directories exist
    for d in ["results/figures", "results/tables", "results/stats",
              "results/checkpoints", "results/logs",
              "data/raw", "data/processed", "data/external"]:
        ensure_dir(d)

    logger.info("=" * 60)
    logger.info("  OrganoChron Pipeline — Starting")
    logger.info(f"  Config: {config_path}")
    logger.info(f"  Start phase: {start_phase}")
    logger.info(f"  Synthetic fallback: {config.get('pipeline', {}).get('use_synthetic_fallback', True)}")
    logger.info("=" * 60)

    # ================================================================
    # Phase 0 — Data Acquisition
    # ================================================================
    if start_phase <= 0:
        logger.info("=" * 60)
        logger.info("  PHASE 0: Data Acquisition")
        logger.info("=" * 60)
        from src import data_acquisition
        with Timer("Phase 0 total"):
            acq_status = data_acquisition.download_all(config)
        logger.info(f"Phase 0 status: {acq_status}")

    # ================================================================
    # Phase 1 — Aging Signatures
    # ================================================================
    if start_phase <= 1:
        logger.info("=" * 60)
        logger.info("  PHASE 1: Aging Signatures")
        logger.info("=" * 60)
        from src import aging_signatures
        with Timer("Phase 1 total"):
            aging_data = aging_signatures.compute_all(config)
        n_tissues = len(aging_data.get("expr_dict", {}))
        n_genes = sum(
            int(df["is_aging"].sum())
            for df in aging_data.get("aging_genes_dict", {}).values()
        )
        logger.info(f"Phase 1 summary: {n_tissues} tissues, {n_genes} total aging genes")
    else:
        logger.info("Loading Phase 1 from checkpoint …")
        aging_data = load_checkpoint("phase1_results")

    # ================================================================
    # Phase 2 — Inter-Organ Secretome Network
    # ================================================================
    if start_phase <= 2:
        logger.info("=" * 60)
        logger.info("  PHASE 2: Inter-Organ Secretome Network")
        logger.info("=" * 60)
        from src import secretome_network
        with Timer("Phase 2 total"):
            secretome_graph = secretome_network.build_graph(aging_data, config)
        logger.info(
            f"Phase 2 summary: {secretome_graph.number_of_nodes()} nodes, "
            f"{secretome_graph.number_of_edges()} edges"
        )
    else:
        logger.info("Loading secretome graph from checkpoint …")
        secretome_graph = load_checkpoint("secretome_graph")

    # ================================================================
    # Phase 2b — Causal Discovery & Integration
    # ================================================================
    if start_phase <= 2:
        logger.info("=" * 60)
        logger.info("  PHASE 2b: Causal Discovery")
        logger.info("=" * 60)
        from src import causal_discovery
        with Timer("Phase 2b — causal inference"):
            causal_graph = causal_discovery.infer_causality(aging_data, config)
        with Timer("Phase 2b — graph integration"):
            integrated_graph = causal_discovery.integrate_graphs(
                secretome_graph, causal_graph, config
            )
        logger.info(
            f"Phase 2b summary: integrated graph has "
            f"{integrated_graph.number_of_edges()} edges"
        )
    else:
        logger.info("Loading integrated graph from checkpoint …")
        integrated_graph = load_checkpoint("integrated_graph")

    # ================================================================
    # Phase 3 — Pacemaker Identification
    # ================================================================
    if start_phase <= 3:
        logger.info("=" * 60)
        logger.info("  PHASE 3: Pacemaker Identification")
        logger.info("=" * 60)
        from src import hub_analysis
        with Timer("Phase 3 total"):
            hub_results = hub_analysis.find_pacemakers(
                integrated_graph, aging_data, config
            )
        pacemakers = hub_results.get("pacemaker_tissues", [])
        logger.info(f"Phase 3 summary: pacemaker organs = {pacemakers}")
    else:
        logger.info("Loading Phase 3 from checkpoint …")
        hub_results = load_checkpoint("phase3_results")

    # ================================================================
    # Phase 4 — Drug Repurposing
    # ================================================================
    if start_phase <= 4:
        logger.info("=" * 60)
        logger.info("  PHASE 4: Drug Repurposing")
        logger.info("=" * 60)
        from src import drug_repurposing
        with Timer("Phase 4 total"):
            drug_results = drug_repurposing.cascade_repurposing(
                integrated_graph, aging_data, hub_results, config
            )
        top = drug_results.get("top_drugs", pd.DataFrame())
        if not top.empty:
            logger.info(f"Phase 4 summary: top drugs = {top.head(5)['drug'].tolist()}")
    else:
        logger.info("Loading Phase 4 from checkpoint …")
        drug_results = load_checkpoint("phase4_results")

    # ================================================================
    # Phase 5 — Validation
    # ================================================================
    if start_phase <= 5:
        logger.info("=" * 60)
        logger.info("  PHASE 5: Validation")
        logger.info("=" * 60)
        from src import validation
        with Timer("Phase 5 total"):
            validation_results = validation.run_all(
                aging_data, integrated_graph, hub_results, drug_results, config
            )
        cv_j = validation_results.get("cv_results", {}).get("mean_jaccard", 0)
        ho_r = validation_results.get("holdout_results", {}).get("overall_r", 0)
        logger.info(f"Phase 5 summary: CV Jaccard = {cv_j:.3f}, held-out r = {ho_r:.3f}")
    else:
        logger.info("Loading Phase 5 from checkpoint …")
        validation_results = load_checkpoint("phase5_results")

    # ================================================================
    # Phase 6 — Figure Generation
    # ================================================================
    if not skip_figures and start_phase <= 6:
        logger.info("=" * 60)
        logger.info("  PHASE 6: Figure Generation")
        logger.info("=" * 60)
        from src import visualization
        with Timer("Phase 6 total"):
            visualization.generate_all_figures(
                aging_data, integrated_graph, hub_results,
                drug_results, validation_results, config,
            )

    # ================================================================
    # Final summary
    # ================================================================
    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info(f"  PIPELINE COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 60)

    # Build global summary JSON
    summary = {
        "pipeline_version": "1.0.0",
        "elapsed_seconds": round(elapsed, 1),
        "n_tissues": len(aging_data.get("expr_dict", {})),
        "n_aging_genes_total": sum(
            int(df["is_aging"].sum())
            for df in aging_data.get("aging_genes_dict", {}).values()
        ),
        "integrated_graph_edges": integrated_graph.number_of_edges(),
        "pacemaker_organs": hub_results.get("pacemaker_tissues", []),
        "top_5_drugs": (
            drug_results.get("top_drugs", pd.DataFrame())
            .head(5)[["drug", "CRS_global", "best_tissue"]]
            .to_dict("records")
            if not drug_results.get("top_drugs", pd.DataFrame()).empty
            else []
        ),
        "validation": {
            "cv_jaccard": validation_results.get("cv_results", {}).get("mean_jaccard", 0),
            "holdout_r": validation_results.get("holdout_results", {}).get("overall_r", 0),
            "permutation_p": validation_results.get("permutation_results", {}).get(
                "permutation_p_value", 1.0
            ),
            "genage_significant_tissues": validation_results.get(
                "genage_results", {}
            ).get("n_significant", 0),
        },
    }
    save_json(summary, "results/stats/summary_statistics.json")
    logger.info(f"Summary saved to results/stats/summary_statistics.json")

    # Print key findings to stdout
    print("\n" + "=" * 60)
    print("  OrganoChron — Key Findings")
    print("=" * 60)
    print(f"  Tissues analysed: {summary['n_tissues']}")
    print(f"  Total aging genes: {summary['n_aging_genes_total']}")
    print(f"  Causal graph edges: {summary['integrated_graph_edges']}")
    print(f"  Pacemaker organs: {', '.join(summary['pacemaker_organs'])}")
    print(f"  Top drug candidate: {summary['top_5_drugs'][0]['drug'] if summary['top_5_drugs'] else 'N/A'}")
    print(f"  CV Jaccard stability: {summary['validation']['cv_jaccard']:.3f}")
    print(f"  Held-out prediction r: {summary['validation']['holdout_r']:.3f}")
    print(f"  Permutation test p: {summary['validation']['permutation_p']:.4f}")
    print(f"  Pipeline time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    app()
