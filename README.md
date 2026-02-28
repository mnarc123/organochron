# OrganoChron

**A Causal Inter-Organ Crosstalk Map Reveals Aging Pacemaker Organs and Cascade Drug Repurposing Targets**

[![DOI](https://zenodo.org/badge/DOI/PLACEHOLDER.svg)](https://doi.org/PLACEHOLDER)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

---

## Overview

OrganoChron is a computational framework that constructs a **causal directed graph of inter-organ aging crosstalk** in humans and uses it for **drug repurposing**. It integrates transcriptomic aging signatures from GTEx v8 (26 tissues, 944 subjects) with secretome-mediated communication networks, causal discovery (PC algorithm + LiNGAM), and LINCS L1000 drug perturbation profiles to:

1. **Map tissue-specific aging programs** — identify age-associated genes and co-expression modules (WGCNA) across 26 human tissues
2. **Build a causal inter-organ graph** — infer directional aging cascades using constraint-based and functional causal discovery on cross-tissue Aging Acceleration correlations
3. **Discover a broadcaster–receiver hierarchy** — the causal graph partitions tissues into 17 aging broadcasters (metabolically active, secretory) and 9 pure receivers (post-mitotic organs including brain and heart)
4. **Identify pacemaker organs** — thyroid emerges as the dominant aging pacemaker (Hub Score = 0.654, bootstrap top-3 frequency = 100%)
5. **Score drugs for cascade reversal potential** — the Cascade Reversal Score (CRS) quantifies a drug's ability to reverse aging not only in its target tissue but also in downstream organs through the causal graph

### Key findings

- **Broadcaster–receiver hierarchy**: peripheral metabolic tissues (thyroid, adipose, intestine) causally drive aging in post-mitotic organs (brain, heart) — challenging the prevailing view that the hypothalamus is the master regulator of systemic aging
- **Drug validation**: 7 of the top-20 CRS-ranked drugs are independently present in the DrugAge database of lifespan-extending compounds (hypergeometric *p* = 8.7 × 10⁻⁵), including metformin, aspirin, dasatinib, and captopril
- **Robust graph topology**: cross-validation Jaccard = 0.684; permutation test: 97 real edges vs. 8.78 permuted (*p* < 10⁻⁴)

## Pipeline architecture

```
Phase 0  ──  Data acquisition (GTEx v8, STRING v12, LINCS L1000, DrugBank, GenAge/DrugAge)
Phase 1  ──  Tissue-specific aging signatures + WGCNA modules + Tissue Aging Scores
Phase 2  ──  Secretome-mediated inter-organ communication graph
Phase 2b ──  Causal discovery (PC + LiNGAM consensus)
Phase 3  ──  Pacemaker identification (centrality + cascade simulation + bootstrap)
Phase 4  ──  Cascade drug repurposing (Reversal Score × graph propagation → CRS)
Phase 5  ──  Validation (CV, held-out, GenAge GSEA, DrugAge overlap, sex-stratified, permutation)
Phase 6  ──  Figure generation
```

## Repository structure

```
organochron/
├── config/
│   └── config.yaml              # All configurable parameters
├── src/
│   ├── data_acquisition.py      # Download and preprocess public datasets
│   ├── aging_signatures.py      # Phase 1: aging genes, WGCNA, TAS
│   ├── secretome_network.py     # Phase 2: secretome inter-organ graph
│   ├── causal_discovery.py      # Phase 2b: PC + LiNGAM causal inference
│   ├── hub_analysis.py          # Phase 3: pacemaker identification
│   ├── drug_repurposing.py      # Phase 4: CRS computation
│   ├── validation.py            # Phase 5: all validation analyses
│   ├── visualization.py         # Phase 6: figure generation
│   └── utils.py                 # Shared utilities
├── main.py                      # Pipeline orchestrator
├── tests/
│   └── test_pipeline.py         # Unit tests (28 tests)
├── paper/
│   ├── manuscript.tex           # LaTeX manuscript (Nature Aging format)
│   └── figures/                 # Publication figures (300 DPI)
├── results/
│   ├── figures/                 # Generated figures
│   ├── tables/                  # Supplementary tables (CSV)
│   └── stats/                   # JSON summaries per phase
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

### Hardware
- **Minimum**: 32 GB RAM, 8-core CPU, 200 GB disk
- **Recommended** (as used in the paper): 64 GB DDR5, Intel i7-12700F (12C/20T), 500 GB disk
- **GPU** (optional): AMD RX 7900 XTX or NVIDIA GPU for PyTorch acceleration

### Software
- Python 3.11+
- Debian/Ubuntu Linux (tested on Debian 13 Trixie)

## Installation

```bash
git clone https://github.com/[username]/organochron.git
cd organochron
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Full pipeline
```bash
python main.py
```

The pipeline runs all phases sequentially (0→5 + figure generation). Each phase saves intermediate results to `results/stats/` as JSON checkpoints, allowing restart from any phase.

### Individual phases
```bash
python main.py --phase 1    # Only aging signatures
python main.py --phase 4    # Only drug repurposing (requires phases 1-3)
```

### Configuration

All parameters are in `config/config.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `aging.spearman_rho_threshold` | 0.15 | Minimum \|ρ\| for Spearman age-correlation |
| `aging.spearman_fdr_threshold` | 0.05 | FDR threshold for Spearman test |
| `causal.pc_alpha` | 0.05 | Significance level for PC algorithm |
| `hub.cascade_damping` | 0.7 | Damping factor for belief propagation |
| `hub.bootstrap_n` | 200 | Number of bootstrap resamples |

## Data sources

All data are publicly available. The pipeline downloads them automatically (Phase 0).

| Dataset | Source | Size |
|---------|--------|------|
| GTEx v8 TPM | [gtexportal.org](https://gtexportal.org) | ~3 GB |
| GTEx annotations | [gtexportal.org](https://gtexportal.org) | ~50 MB |
| STRING v12 | [string-db.org](https://string-db.org) | ~400 MB |
| LINCS L1000 Level 5 | [GEO GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742) | ~21 GB |
| Human Protein Atlas (secretome) | [proteinatlas.org](https://www.proteinatlas.org) | ~5 MB |
| DrugBank (open access) | [drugbank.com](https://go.drugbank.com) | ~200 MB |
| GenAge / DrugAge | [genomics.senescence.info](https://genomics.senescence.info) | ~2 MB |

**Note**: the LINCS L1000 GCTX file (GSE92742) is ~21 GB compressed. If automatic download fails, download manually and place in `data/raw/`.

## Outputs

### Main results (`results/stats/`)
- `phase1_summary.json` — Aging gene counts, WGCNA modules per tissue
- `causal_discovery_summary.json` — PC, LiNGAM, and consensus edge counts
- `phase3_summary.json` — Pacemaker organs, hub scores, TCI, bootstrap frequencies
- `phase4_summary.json` — Top drug candidates, CRS scores, DrugAge overlap
- `phase5_summary.json` — All validation metrics
- `centrality_comparison.json` — Causal vs. secretome vs. integrated graph centralities
- `summary_statistics.json` — Pipeline-wide summary

### Figures (`results/figures/`)
- `figure1.png/pdf` — Framework overview and aging signatures
- `figure2.png/pdf` — Causal graph and broadcaster–receiver hierarchy
- `figure3.png/pdf` — Pacemaker cascade simulation
- `figure4_drug_repurposing_real.png/pdf` — Drug repurposing with real LINCS data
- `figure5.png/pdf` — Computational validation

### Supplementary tables (`results/tables/`)
- `supplementary_table_S1.csv` — Tissue summary (samples, aging genes, modules, hub scores)
- `supplementary_table_S2.csv` — WGCNA aging modules with GO/KEGG enrichment
- `supplementary_table_S3.csv` — Causal graph edges with confidence scores
- `supplementary_table_S4.csv` — Top-50 drug candidates with CRS and annotations
- `supplementary_table_S5.csv` — GenAge enrichment per tissue (Fisher + GSEA)

## Reproducibility

- All random seeds fixed at 42
- Pipeline version tracked in `summary_statistics.json`
- Full run time: ~11 minutes on recommended hardware (excluding data download)
- All intermediate results are checkpointed for exact reproduction

## Tests

```bash
python -m pytest tests/ -v
```

28 unit tests covering correlation computation, graph construction, cascade propagation, and synthetic data validation.

## Citation

If you use OrganoChron in your research, please cite:

```bibtex
@article{organochron2025,
  title={OrganoChron: A Causal Inter-Organ Crosstalk Map Reveals Aging Pacemaker Organs and Cascade Drug Repurposing Targets},
  author={[Author]},
  year={2025},
  doi={PLACEHOLDER}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

This work uses publicly available data from the GTEx Consortium, LINCS/CMap, STRING, Human Protein Atlas, DrugBank, GenAge, and DrugAge. We thank these projects for making their data freely accessible.
