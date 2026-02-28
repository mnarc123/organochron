"""Phase 6 — Figure Generation.

Produces all five main figures for the OrganoChron manuscript at
publication quality (300 DPI, Nature-compatible dimensions).

Figure 1: Framework overview (workflow, gene counts, expression trends, WGCNA)
Figure 2: Causal inter-organ graph (network, hub scores, adjacency matrix)
Figure 3: Pacemaker organs (cascade propagation, TCI, radar chart)
Figure 4: Drug repurposing (volcano, top drugs, mini-network, DrugAge overlap)
Figure 5: Validation (Jaccard CV, held-out scatter, GenAge, sex-stratified)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from src.utils import (
    Timer,
    SYSTEM_COLORS,
    TISSUE_SYSTEM,
    ensure_dir,
    set_seed,
)

RESULTS = Path("results")
FIGDIR = RESULTS / "figures"


# =========================================================================
# Global style setup
# =========================================================================


def _setup_style(config: dict[str, Any]) -> None:
    """Apply publication-quality matplotlib style.

    Parameters
    ----------
    config : dict
    """
    fig_cfg = config["figures"]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [fig_cfg["font"], "DejaVu Sans", "Arial", "Helvetica"],
        "font.size": fig_cfg["font_size_label"],
        "axes.titlesize": fig_cfg["font_size_title"],
        "axes.labelsize": fig_cfg["font_size_axis"],
        "xtick.labelsize": fig_cfg["font_size_label"],
        "ytick.labelsize": fig_cfg["font_size_label"],
        "legend.fontsize": fig_cfg["font_size_label"],
        "figure.dpi": fig_cfg["dpi"],
        "savefig.dpi": fig_cfg["dpi"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    sns.set_palette(fig_cfg["palette"])


def _mm_to_inches(mm: float) -> float:
    """Convert millimetres to inches.

    Parameters
    ----------
    mm : float

    Returns
    -------
    float
    """
    return mm / 25.4


def _add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    """Add a bold panel label (A, B, C …) to an axes.

    Parameters
    ----------
    ax : plt.Axes
    label : str
    x, y : float
        Position in axes coordinates.
    """
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")


def _save_figure(fig: plt.Figure, name: str, config: dict[str, Any]) -> None:
    """Save figure in both PNG and PDF.

    Parameters
    ----------
    fig : plt.Figure
    name : str
        Base filename without extension.
    config : dict
    """
    ensure_dir(FIGDIR)
    fig.savefig(FIGDIR / f"{name}.png", format="png")
    fig.savefig(FIGDIR / f"{name}.pdf", format="pdf")
    plt.close(fig)
    logger.info(f"Saved figure: {name}.png / .pdf")


def _tissue_short(name: str) -> str:
    """Abbreviate a tissue name for plot labels.

    Parameters
    ----------
    name : str

    Returns
    -------
    str
    """
    abbr = {
        "Adipose - Subcutaneous": "Adipose SC",
        "Adipose - Visceral (Omentum)": "Adipose Visc",
        "Adrenal Gland": "Adrenal",
        "Artery - Aorta": "Art. Aorta",
        "Artery - Coronary": "Art. Coronary",
        "Artery - Tibial": "Art. Tibial",
        "Brain - Cortex": "Brain Ctx",
        "Brain - Cerebellum": "Brain Cbl",
        "Colon - Sigmoid": "Colon Sig",
        "Colon - Transverse": "Colon Trans",
        "Esophagus - Mucosa": "Esoph. Muc",
        "Heart - Atrial Appendage": "Heart AA",
        "Heart - Left Ventricle": "Heart LV",
        "Kidney - Cortex": "Kidney Ctx",
        "Liver": "Liver",
        "Lung": "Lung",
        "Muscle - Skeletal": "Muscle Sk",
        "Nerve - Tibial": "Nerve Tib",
        "Pancreas": "Pancreas",
        "Pituitary": "Pituitary",
        "Skin - Not Sun Exposed (Suprapubic)": "Skin NSE",
        "Skin - Sun Exposed (Lower leg)": "Skin SE",
        "Small Intestine - Terminal Ileum": "Sm. Int.",
        "Spleen": "Spleen",
        "Stomach": "Stomach",
        "Thyroid": "Thyroid",
        "Whole Blood": "Blood",
    }
    return abbr.get(name, name[:12])


def _tissue_color(name: str) -> str:
    """Get colour for a tissue based on its organ system.

    Parameters
    ----------
    name : str

    Returns
    -------
    str
        Hex colour.
    """
    system = TISSUE_SYSTEM.get(name, "other")
    return SYSTEM_COLORS.get(system, "#999999")


# =========================================================================
# Figure 1 — Overview
# =========================================================================


def figure1(
    aging_data: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Generate Figure 1: OrganoChron framework overview.

    Panels:
    A — Workflow schematic (text-based)
    B — Aging gene counts per tissue (horizontal bar)
    C — Example gene expression vs age trends
    D — WGCNA aging module heatmap

    Parameters
    ----------
    aging_data : dict
    config : dict
    """
    _setup_style(config)
    w = _mm_to_inches(183)
    fig, axes = plt.subplots(2, 2, figsize=(w, w * 0.75))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # --- Panel A: workflow ---
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.axis("off")
    steps = [
        (1.5, 8.5, "GTEx v8\n27 tissues", "#E8F0FE"),
        (5, 8.5, "Aging\nSignatures", "#FFF3E0"),
        (8.5, 8.5, "WGCNA\nModules", "#E8F5E9"),
        (1.5, 5, "Secretome\nGraph", "#FCE4EC"),
        (5, 5, "Causal\nDiscovery", "#F3E5F5"),
        (8.5, 5, "Integrated\nGraph", "#E0F7FA"),
        (3, 1.5, "Pacemaker\nOrgans", "#FFF9C4"),
        (7, 1.5, "Cascade Drug\nRepurposing", "#FFECB3"),
    ]
    for x, y, txt, col in steps:
        ax_a.add_patch(plt.Rectangle((x - 1.2, y - 0.7), 2.4, 1.4,
                                      facecolor=col, edgecolor="black", lw=0.8,
                                      transform=ax_a.transData, zorder=2))
        ax_a.text(x, y, txt, ha="center", va="center", fontsize=7, zorder=3)
    # Arrows
    arrows = [(2.7, 8.5, 3.8, 8.5), (6.2, 8.5, 7.3, 8.5),
              (1.5, 7.8, 1.5, 5.7), (5, 7.8, 5, 5.7),
              (6.2, 5, 7.3, 5), (3.8, 5, 5, 5),
              (5, 4.3, 3, 2.2), (5, 4.3, 7, 2.2)]
    for x1, y1, x2, y2 in arrows:
        ax_a.annotate("", xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle="->", lw=1.0, color="gray"))
    _add_panel_label(ax_a, "A")

    # --- Panel B: aging gene counts ---
    aging_genes_dict = aging_data.get("aging_genes_dict", {})
    tissues_b = sorted(aging_genes_dict.keys(), key=lambda t: aging_genes_dict[t]["is_aging"].sum())
    n_up = [((aging_genes_dict[t]["is_aging"]) & (aging_genes_dict[t]["direction"] == "up")).sum()
            for t in tissues_b]
    n_down = [((aging_genes_dict[t]["is_aging"]) & (aging_genes_dict[t]["direction"] == "down")).sum()
              for t in tissues_b]
    short_names = [_tissue_short(t) for t in tissues_b]
    y_pos = np.arange(len(tissues_b))
    ax_b.barh(y_pos, n_up, color="#E24A33", label="Up with age", height=0.7)
    ax_b.barh(y_pos, [-x for x in n_down], color="#348ABD", label="Down with age", height=0.7)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(short_names, fontsize=6)
    ax_b.set_xlabel("Number of aging-associated genes")
    ax_b.axvline(0, color="black", lw=0.5)
    ax_b.legend(fontsize=6, loc="lower right")
    _add_panel_label(ax_b, "B")

    # --- Panel C: example gene trends ---
    expr_dict = aging_data.get("expr_dict", {})
    meta = aging_data.get("meta", pd.DataFrame())
    example_genes = ["IL6", "SIRT1", "TP53"]
    example_tissues = list(expr_dict.keys())[:3]
    colors_c = ["#E24A33", "#348ABD", "#7A68A6"]
    for gi, gene in enumerate(example_genes):
        for ti, tissue in enumerate(example_tissues):
            expr = expr_dict.get(tissue)
            if expr is None or gene not in expr.index:
                continue
            tissue_meta = meta[meta["SMTSD"] == tissue].set_index("SAMPID")
            common = sorted(set(expr.columns) & set(tissue_meta.index))
            if len(common) < 10:
                continue
            ages = tissue_meta.loc[common, "AGE_MID"].astype(float).values
            vals = expr.loc[gene, common].values.astype(float)
            ax_c.scatter(ages + np.random.uniform(-1, 1, len(ages)),
                         vals, s=3, alpha=0.3, color=colors_c[gi])
            # Trend line
            z = np.polyfit(ages, vals, 1)
            x_line = np.linspace(ages.min(), ages.max(), 50)
            ax_c.plot(x_line, np.polyval(z, x_line), color=colors_c[gi],
                      lw=1.5, label=f"{gene} ({_tissue_short(tissue)})" if ti == 0 else "")
            break  # one tissue per gene
    ax_c.set_xlabel("Age (years)")
    ax_c.set_ylabel("log₂(TPM + 1)")
    ax_c.legend(fontsize=6, loc="best")
    _add_panel_label(ax_c, "C")

    # --- Panel D: WGCNA heatmap ---
    wgcna_dict = aging_data.get("wgcna_dict", {})
    tissues_d = sorted(wgcna_dict.keys())[:15]  # Cap for readability
    # Collect module-age correlations
    all_modules: set[int] = set()
    for t in tissues_d:
        cors = wgcna_dict[t].get("module_age_cors", {})
        all_modules.update(cors.keys())
    all_modules_sorted = sorted(all_modules)[:20]

    if all_modules_sorted and tissues_d:
        heatmap_data = np.full((len(tissues_d), len(all_modules_sorted)), np.nan)
        for ti, t in enumerate(tissues_d):
            cors = wgcna_dict[t].get("module_age_cors", {})
            for mi, m in enumerate(all_modules_sorted):
                if m in cors:
                    heatmap_data[ti, mi] = cors[m][0]  # rho value
        hm_df = pd.DataFrame(heatmap_data,
                              index=[_tissue_short(t) for t in tissues_d],
                              columns=[f"M{m}" for m in all_modules_sorted])
        sns.heatmap(hm_df, ax=ax_d, cmap="RdBu_r", center=0,
                    vmin=-0.5, vmax=0.5, xticklabels=True, yticklabels=True,
                    cbar_kws={"label": "Spearman ρ (age)", "shrink": 0.7})
        ax_d.set_xlabel("Module")
        ax_d.set_ylabel("Tissue")
        ax_d.tick_params(axis="both", labelsize=5)
    else:
        ax_d.text(0.5, 0.5, "Insufficient WGCNA data", ha="center", va="center",
                  transform=ax_d.transAxes)
    _add_panel_label(ax_d, "D")

    fig.tight_layout()
    _save_figure(fig, "figure1", config)


# =========================================================================
# Figure 2 — Causal inter-organ graph
# =========================================================================


def figure2(
    integrated_graph: nx.DiGraph,
    hub_results: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Generate Figure 2: The causal inter-organ graph.

    Panels:
    A — Force-directed graph layout
    B — Hub Score barplot
    C — Adjacency heatmap

    Parameters
    ----------
    integrated_graph : nx.DiGraph
    hub_results : dict
    config : dict
    """
    _setup_style(config)
    w = _mm_to_inches(183)
    fig = plt.figure(figsize=(w, w * 0.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1.2])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    centrality_df = hub_results.get("centrality_df", pd.DataFrame())
    hub_scores = dict(zip(centrality_df["tissue"], centrality_df["hub_score"])) if not centrality_df.empty else {}

    # --- Panel A: Network ---
    G = integrated_graph
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    node_sizes = []
    node_colors = []
    for n in G.nodes():
        hs = hub_scores.get(n, 0.5)
        node_sizes.append(200 + 800 * hs)
        node_colors.append(_tissue_color(n))

    # Draw edges with confidence-based styling
    for u, v, d in G.edges(data=True):
        conf = d.get("confidence", 0.5)
        st = d.get("source_type", "unknown")
        if st == "confirmed":
            ec, ls = "black", "-"
        elif st == "causal_only":
            ec, ls = "#348ABD", "--"
        else:
            ec, ls = "#CCCCCC", ":"
        w_val = d.get("weight", 0.1)
        ax_a.annotate("",
                       xy=pos[v], xytext=pos[u],
                       arrowprops=dict(arrowstyle="-|>", color=ec,
                                        lw=0.3 + 2.0 * w_val,
                                        linestyle=ls, alpha=0.7,
                                        connectionstyle="arc3,rad=0.1"))

    nx.draw_networkx_nodes(G, pos, ax=ax_a, node_size=node_sizes,
                           node_color=node_colors, edgecolors="black", linewidths=0.5)
    labels = {n: _tissue_short(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax_a, font_size=5)

    # Legend for systems
    system_patches = [mpatches.Patch(color=c, label=s.capitalize())
                      for s, c in SYSTEM_COLORS.items()]
    ax_a.legend(handles=system_patches, fontsize=5, loc="lower left",
                ncol=2, framealpha=0.8)
    ax_a.set_title("Causal Inter-Organ Graph")
    ax_a.axis("off")
    _add_panel_label(ax_a, "A", x=-0.05)

    # --- Panel B: Hub score barplot ---
    if not centrality_df.empty:
        top = centrality_df.head(15)
        colors_b = [_tissue_color(t) for t in top["tissue"]]
        bootstrap_df = hub_results.get("bootstrap_df", pd.DataFrame())
        yerr = None
        if not bootstrap_df.empty:
            boot_map = dict(zip(bootstrap_df["tissue"], bootstrap_df["std_hub_score"]))
            yerr = [boot_map.get(t, 0) for t in top["tissue"]]
        y_pos = np.arange(len(top))
        ax_b.barh(y_pos, top["hub_score"].values, color=colors_b,
                   xerr=yerr, ecolor="gray", capsize=2, height=0.7)
        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels([_tissue_short(t) for t in top["tissue"]], fontsize=6)
        ax_b.set_xlabel("Hub Score")
        ax_b.invert_yaxis()
    _add_panel_label(ax_b, "B", x=-0.2)

    # --- Panel C: Adjacency heatmap ---
    nodes = sorted(G.nodes())
    n = len(nodes)
    adj = np.zeros((n, n))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if G.has_edge(u, v):
                adj[i, j] = G.edges[u, v].get("weight", 0)
    short = [_tissue_short(t) for t in nodes]
    sns.heatmap(adj, ax=ax_c, xticklabels=short, yticklabels=short,
                cmap="YlOrRd", vmin=0, cbar_kws={"label": "Edge weight", "shrink": 0.7})
    ax_c.set_title("Adjacency Matrix")
    ax_c.tick_params(axis="both", labelsize=4, rotation=45)
    _add_panel_label(ax_c, "C", x=-0.15)

    fig.tight_layout()
    _save_figure(fig, "figure2", config)


# =========================================================================
# Figure 3 — Pacemaker organs and cascade
# =========================================================================


def figure3(
    hub_results: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Generate Figure 3: Pacemaker organs and cascade simulation.

    Panels:
    A — Cascade propagation time-series (one sub-panel per top-3 pacemaker)
    B — Total Cascade Impact dot plot
    C — Radar chart of top-5 tissues on 4 centrality metrics

    Parameters
    ----------
    hub_results : dict
    config : dict
    """
    _setup_style(config)
    w = _mm_to_inches(183)
    fig = plt.figure(figsize=(w, w * 0.7))

    # Layout: top row = 3 cascade panels, bottom = TCI + radar
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    cascade_ts = hub_results.get("cascade_timeseries", {})
    pacemakers = hub_results.get("pacemaker_tissues", [])[:3]
    centrality_df = hub_results.get("centrality_df", pd.DataFrame())
    tci_df = hub_results.get("tci_df", pd.DataFrame())

    # --- Panel A: Cascade time-series ---
    for pi, pm in enumerate(pacemakers):
        ax = fig.add_subplot(gs[0, pi])
        ts_df = cascade_ts.get(pm, pd.DataFrame())
        if not ts_df.empty:
            for tissue in ts_df["tissue"].unique():
                sub = ts_df[ts_df["tissue"] == tissue]
                if sub["delta_aging_score"].max() > 0.01:
                    ax.plot(sub["iteration"], sub["delta_aging_score"],
                            color=_tissue_color(tissue), lw=1.0, alpha=0.7,
                            label=_tissue_short(tissue))
            ax.set_xlabel("Iteration")
            ax.set_ylabel("ΔAging Score")
            ax.set_title(f"Source: {_tissue_short(pm)}", fontsize=9)
            if pi == 0:
                ax.legend(fontsize=4, loc="upper right", ncol=2)
        _add_panel_label(ax, chr(65 + pi))  # A1, A2, A3

    # --- Panel B: TCI dot plot ---
    ax_b = fig.add_subplot(gs[1, 0:2])
    if not tci_df.empty:
        top_tci = tci_df.head(15)
        colors = [_tissue_color(t) for t in top_tci["tissue"]]
        bootstrap_df = hub_results.get("bootstrap_df", pd.DataFrame())
        ax_b.scatter(top_tci["TCI"], range(len(top_tci)), c=colors, s=60,
                     edgecolors="black", linewidths=0.5, zorder=3)
        ax_b.set_yticks(range(len(top_tci)))
        ax_b.set_yticklabels([_tissue_short(t) for t in top_tci["tissue"]], fontsize=6)
        ax_b.set_xlabel("Total Cascade Impact")
        ax_b.invert_yaxis()
        ax_b.axvline(0, color="gray", lw=0.5, ls="--")
    _add_panel_label(ax_b, "B" if len(pacemakers) < 3 else "D")

    # --- Panel C: Radar chart ---
    ax_c = fig.add_subplot(gs[1, 2], polar=True)
    if not centrality_df.empty:
        metrics = ["out_degree_norm", "pagerank_norm", "betweenness_norm", "cascade_depth_norm"]
        metric_labels = ["Out-degree", "PageRank\n(inv)", "Betweenness", "Cascade\nDepth"]
        top5 = centrality_df.head(5)
        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        colors_radar = sns.color_palette("colorblind", n_colors=5)
        for i, (_, row) in enumerate(top5.iterrows()):
            values = [row[m] for m in metrics]
            values += values[:1]
            ax_c.plot(angles, values, lw=1.5, color=colors_radar[i],
                      label=_tissue_short(row["tissue"]))
            ax_c.fill(angles, values, alpha=0.1, color=colors_radar[i])

        ax_c.set_xticks(angles[:-1])
        ax_c.set_xticklabels(metric_labels, fontsize=6)
        ax_c.legend(fontsize=5, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _add_panel_label(ax_c, "E" if len(pacemakers) >= 3 else "C", x=0.0, y=1.15)

    fig.tight_layout()
    _save_figure(fig, "figure3", config)


# =========================================================================
# Figure 4 — Drug repurposing
# =========================================================================


def figure4(
    drug_results: dict[str, Any],
    hub_results: dict[str, Any],
    integrated_graph: nx.DiGraph,
    config: dict[str, Any],
) -> None:
    """Generate Figure 4: Drug repurposing results.

    Panels:
    A — Volcano plot of reversal scores
    B — Top-20 drugs by CRS_global (horizontal bar)
    C — Mini-network for top-1 drug
    D — DrugAge overlap (UpSet-like or Venn)

    Parameters
    ----------
    drug_results : dict
    hub_results : dict
    integrated_graph : nx.DiGraph
    config : dict
    """
    _setup_style(config)
    w = _mm_to_inches(183)
    fig, axes = plt.subplots(2, 2, figsize=(w, w * 0.75))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    reversal_df = drug_results.get("reversal_df", pd.DataFrame())
    top_drugs = drug_results.get("top_drugs", pd.DataFrame())
    mechanisms = drug_results.get("mechanisms", {})
    pacemakers = hub_results.get("pacemaker_tissues", [])[:3]

    # --- Panel A: Volcano plot ---
    if not reversal_df.empty and pacemakers:
        pace_rev = reversal_df[reversal_df["tissue"].isin(pacemakers)].copy()
        if not pace_rev.empty:
            # Compute pseudo-significance from reversal score magnitude
            pace_rev["neg_log_p"] = np.abs(pace_rev["reversal_score"]) * 5
            pace_rev["significant"] = pace_rev["reversal_score"].abs() > 0.3
            colors = np.where(pace_rev["significant"], "#E24A33", "#CCCCCC")
            ax_a.scatter(pace_rev["reversal_score"], pace_rev["neg_log_p"],
                         c=colors, s=8, alpha=0.5, edgecolors="none")
            ax_a.axvline(0.3, color="gray", ls="--", lw=0.5)
            ax_a.axvline(-0.3, color="gray", ls="--", lw=0.5)
            ax_a.set_xlabel("Reversal Score")
            ax_a.set_ylabel("−log₁₀(p-value) [proxy]")
            ax_a.set_title("Drug–Aging Signature Reversal")
    _add_panel_label(ax_a, "A")

    # --- Panel B: Top drugs barplot ---
    if not top_drugs.empty:
        top = top_drugs.head(20)
        colors_b = ["#188487" if da else "#E5AE38" for da in top["in_drugage"]]
        y_pos = np.arange(len(top))
        ax_b.barh(y_pos, top["CRS_global"].values, color=colors_b, height=0.7)
        labels_b = [f"{d} {'★' if da else ''}" for d, da in zip(top["drug"], top["in_drugage"])]
        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels(labels_b, fontsize=5)
        ax_b.set_xlabel("Cascade Reversal Score (CRS)")
        ax_b.invert_yaxis()
        # Legend
        ax_b.legend(handles=[
            mpatches.Patch(color="#188487", label="In DrugAge"),
            mpatches.Patch(color="#E5AE38", label="Not in DrugAge"),
        ], fontsize=5, loc="lower right")
    _add_panel_label(ax_b, "B")

    # --- Panel C: Mini-network for top-1 drug ---
    ax_c.axis("off")
    if mechanisms:
        top_drug = list(mechanisms.keys())[0]
        mech = mechanisms[top_drug]
        # Simple diagram
        ax_c.set_xlim(0, 10)
        ax_c.set_ylim(0, 10)
        # Drug node
        ax_c.add_patch(plt.Rectangle((3.5, 8.5), 3, 1, facecolor="#FCE4EC",
                                      edgecolor="black", lw=1))
        ax_c.text(5, 9, top_drug.capitalize(), ha="center", va="center", fontsize=7,
                  fontweight="bold")
        # Targets
        targets = mech.get("targets", [])[:3]
        for i, t in enumerate(targets):
            x = 2 + i * 3
            ax_c.add_patch(plt.Circle((x, 6.5), 0.6, facecolor="#E8F0FE",
                                       edgecolor="black", lw=0.5))
            ax_c.text(x, 6.5, t[:6], ha="center", va="center", fontsize=5)
            ax_c.annotate("", xy=(x, 7.1), xytext=(5, 8.5),
                           arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"))
        # Tissue
        tissue = mech.get("target_tissue", "Unknown")
        ax_c.add_patch(plt.Rectangle((3.5, 4), 3, 1, facecolor="#E8F5E9",
                                      edgecolor="black", lw=1))
        ax_c.text(5, 4.5, _tissue_short(tissue), ha="center", va="center", fontsize=7)
        ax_c.annotate("", xy=(5, 5), xytext=(5, 5.9),
                       arrowprops=dict(arrowstyle="->", lw=1, color="#E24A33"))
        # Downstream
        downstream = mech.get("downstream_tissues", [])[:3]
        for i, ds in enumerate(downstream):
            x = 2 + i * 3
            ax_c.add_patch(plt.Rectangle((x - 1, 1.5), 2, 0.8,
                                          facecolor="#FFF3E0", edgecolor="black", lw=0.5))
            ax_c.text(x, 1.9, _tissue_short(ds["tissue"]), ha="center",
                      va="center", fontsize=5)
            ax_c.annotate("", xy=(x, 2.3), xytext=(5, 4),
                           arrowprops=dict(arrowstyle="->", lw=0.8, color="gray",
                                            connectionstyle="arc3,rad=0.2"))
        ax_c.set_title(f"Mechanism: {top_drug.capitalize()}", fontsize=8)
    _add_panel_label(ax_c, "C")

    # --- Panel D: DrugAge overlap ---
    if not top_drugs.empty:
        in_da = top_drugs["in_drugage"].sum()
        not_da = len(top_drugs) - in_da
        ax_d.bar(["In DrugAge", "Not in DrugAge"], [in_da, not_da],
                 color=["#188487", "#E5AE38"])
        ax_d.set_ylabel("Number of top candidates")
        ax_d.set_title("DrugAge Overlap")
        # Add percentage annotation
        total = len(top_drugs)
        ax_d.text(0, in_da + 0.3, f"{in_da}/{total}\n({100*in_da/total:.0f}%)",
                  ha="center", fontsize=7)
    _add_panel_label(ax_d, "D")

    fig.tight_layout()
    _save_figure(fig, "figure4", config)


# =========================================================================
# Figure 5 — Validation
# =========================================================================


def figure5(
    validation_results: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Generate Figure 5: Validation results.

    Panels:
    A — Jaccard stability across CV folds
    B — Held-out prediction scatter
    C — GenAge enrichment per tissue
    D — Sex-stratified comparison

    Parameters
    ----------
    validation_results : dict
    config : dict
    """
    _setup_style(config)
    w = _mm_to_inches(183)
    fig, axes = plt.subplots(2, 2, figsize=(w, w * 0.7))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    cv = validation_results.get("cv_results", {})
    holdout = validation_results.get("holdout_results", {})
    genage = validation_results.get("genage_results", {})
    sex = validation_results.get("sex_results", {})

    # --- Panel A: Jaccard stability ---
    jaccard_vals = cv.get("jaccard_similarities", [])
    if jaccard_vals:
        ax_a.hist(jaccard_vals, bins=min(20, len(jaccard_vals)),
                  color="#348ABD", edgecolor="white", alpha=0.8)
        ax_a.axvline(cv.get("mean_jaccard", 0), color="#E24A33", lw=2,
                     label=f"Mean = {cv.get('mean_jaccard', 0):.3f}")
        ax_a.set_xlabel("Pairwise Jaccard Similarity")
        ax_a.set_ylabel("Count")
        ax_a.set_title("Cross-Validation Edge Stability")
        ax_a.legend(fontsize=7)
    else:
        ax_a.text(0.5, 0.5, "No CV data", ha="center", va="center",
                  transform=ax_a.transAxes)
    _add_panel_label(ax_a, "A")

    # --- Panel B: Held-out scatter ---
    predictions = holdout.get("predictions", pd.DataFrame())
    if not predictions.empty and "predicted" in predictions.columns:
        ax_b.scatter(predictions["observed"], predictions["predicted"],
                     s=5, alpha=0.3, color="#348ABD", edgecolors="none")
        # Identity line
        lims = [min(predictions["observed"].min(), predictions["predicted"].min()),
                max(predictions["observed"].max(), predictions["predicted"].max())]
        ax_b.plot(lims, lims, "k--", lw=0.5, alpha=0.5)
        r_val = holdout.get("overall_r", 0)
        ax_b.set_xlabel("Observed Aging Acceleration")
        ax_b.set_ylabel("Predicted (from graph)")
        ax_b.set_title(f"Held-out Prediction (r = {r_val:.3f})")
    else:
        ax_b.text(0.5, 0.5, "No held-out data", ha="center", va="center",
                  transform=ax_b.transAxes)
    _add_panel_label(ax_b, "B")

    # --- Panel C: GenAge enrichment ---
    per_tissue = genage.get("per_tissue", {})
    if per_tissue:
        tissues_ga = sorted(per_tissue.keys(),
                            key=lambda t: -np.log10(per_tissue[t]["pvalue"] + 1e-300))[:15]
        neglogp = [-np.log10(per_tissue[t]["pvalue"] + 1e-300) for t in tissues_ga]
        colors_c = ["#E24A33" if per_tissue[t].get("fdr", 1) < 0.05 else "#CCCCCC"
                    for t in tissues_ga]
        y_pos = np.arange(len(tissues_ga))
        ax_c.barh(y_pos, neglogp, color=colors_c, height=0.7)
        ax_c.set_yticks(y_pos)
        ax_c.set_yticklabels([_tissue_short(t) for t in tissues_ga], fontsize=6)
        ax_c.set_xlabel("−log₁₀(p-value)")
        ax_c.axvline(-np.log10(0.05), color="gray", ls="--", lw=0.5, label="p = 0.05")
        ax_c.set_title("GenAge Gene Enrichment")
        ax_c.invert_yaxis()
        ax_c.legend(fontsize=6)
    _add_panel_label(ax_c, "C")

    # --- Panel D: Sex-stratified summary ---
    if sex:
        metrics_d = ["Edge\nJaccard", "Hub Rank\nρ"]
        vals_d = [sex.get("edge_jaccard", 0), sex.get("hub_rank_spearman_rho", 0)]
        colors_d = ["#7A68A6", "#E24A33"]
        ax_d.bar(metrics_d, vals_d, color=colors_d, width=0.5)
        ax_d.set_ylabel("Similarity Score")
        ax_d.set_title("Male vs Female Consistency")
        ax_d.set_ylim(0, 1)
        for i, v in enumerate(vals_d):
            ax_d.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    _add_panel_label(ax_d, "D")

    fig.tight_layout()
    _save_figure(fig, "figure5", config)


# =========================================================================
# Master entry point
# =========================================================================


def generate_all_figures(
    aging_data: dict[str, Any],
    integrated_graph: nx.DiGraph,
    hub_results: dict[str, Any],
    drug_results: dict[str, Any],
    validation_results: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Generate all five manuscript figures.

    Parameters
    ----------
    aging_data : dict
    integrated_graph : nx.DiGraph
    hub_results : dict
    drug_results : dict
    validation_results : dict
    config : dict
    """
    set_seed(42)
    ensure_dir(FIGDIR)

    with Timer("Figure 1 — Overview"):
        figure1(aging_data, config)

    with Timer("Figure 2 — Causal Graph"):
        figure2(integrated_graph, hub_results, config)

    with Timer("Figure 3 — Pacemaker Organs"):
        figure3(hub_results, config)

    with Timer("Figure 4 — Drug Repurposing"):
        figure4(drug_results, hub_results, integrated_graph, config)

    with Timer("Figure 5 — Validation"):
        figure5(validation_results, config)

    logger.info(f"All figures saved to {FIGDIR}")
