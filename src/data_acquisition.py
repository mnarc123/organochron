"""Phase 0 — Data Acquisition and Preprocessing.

Downloads all required public datasets (GTEx, LINCS, STRING, HPA, DrugBank,
GenAge, DrugAge, MSigDB) and converts them to standardised formats stored
under ``data/processed/``.

Where automated download is not possible (e.g. licence-gated files), clear
instructions are printed and — if configured — realistic synthetic
placeholders are generated so that downstream phases can still run.
"""

from __future__ import annotations

import gzip
import io
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

from src.utils import (
    Timer,
    age_midpoint,
    ensure_dir,
    load_config,
    save_parquet,
    set_seed,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
EXTERNAL = Path("data/external")

_GTEX_BASE = "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq"
_GTEX_ANNOT = "https://storage.googleapis.com/adult-gtex/annotations/v8"

DOWNLOAD_URLS: dict[str, str] = {
    # GTEx v8
    "gtex_tpm": f"{_GTEX_BASE}/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz",
    "gtex_sample": f"{_GTEX_ANNOT}/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt",
    "gtex_subject": f"{_GTEX_ANNOT}/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt",
    # STRING v12
    "string_links": "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
    "string_info": "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
    # GenAge / DrugAge
    "genage": "https://genomics.senescence.info/genes/human_genes.zip",
    "drugage": "https://genomics.senescence.info/drugs/dataset.zip",
    # MSigDB Hallmark
    "hallmark_gmt": "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt",
}

# HPA secretome — direct TSV download
HPA_SECRETOME_URL = (
    "https://www.proteinatlas.org/api/search_download.php?"
    "search=protein_class%3ASecreted+proteins&format=tsv&columns=g&compress=no"
)

# ---------------------------------------------------------------------------
# Generic download helper
# ---------------------------------------------------------------------------


def _download_file(
    url: str,
    dest: Path,
    description: str = "",
    timeout: int = 600,
    retries: int = 3,
) -> bool:
    """Download a file with progress bar and retry logic.

    Parameters
    ----------
    url : str
        Source URL.
    dest : Path
        Local destination path.
    description : str
        Label for the progress bar.
    timeout : int
        Request timeout in seconds.
    retries : int
        Number of retry attempts.

    Returns
    -------
    bool
        ``True`` if download succeeded.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"Already downloaded: {dest}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Downloading {description or url} (attempt {attempt}/{retries})")
            resp = requests.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as fh, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=description or dest.name,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    bar.update(len(chunk))
            logger.info(f"Saved {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")
            return True
        except Exception as exc:
            logger.warning(f"Download failed ({exc})")
            if dest.exists():
                dest.unlink()
    logger.error(f"All {retries} attempts failed for {url}")
    return False


# ---------------------------------------------------------------------------
# GTEx processing
# ---------------------------------------------------------------------------


def download_gtex(config: dict[str, Any]) -> bool:
    """Download GTEx v8 TPM matrix and annotation files.

    Parameters
    ----------
    config : dict
        Global configuration.

    Returns
    -------
    bool
        ``True`` if all files were obtained.
    """
    ok = True
    ok &= _download_file(DOWNLOAD_URLS["gtex_tpm"], RAW / "gtex_tpm.gct.gz", "GTEx TPM matrix")
    ok &= _download_file(DOWNLOAD_URLS["gtex_sample"], RAW / "gtex_sample_attributes.txt", "GTEx sample annotations")
    ok &= _download_file(DOWNLOAD_URLS["gtex_subject"], RAW / "gtex_subject_phenotypes.txt", "GTEx subject phenotypes")
    return ok


def parse_gtex(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Parse downloaded GTEx files into per-tissue DataFrames.

    Produces:
    - ``data/processed/gtex_expr_{tissue_safe}.parquet`` — log2(TPM+1) expression
    - ``data/processed/gtex_meta.parquet`` — merged sample + subject metadata

    Parameters
    ----------
    config : dict
        Global configuration.

    Returns
    -------
    dict
        ``{"meta": DataFrame, "tissues": list[str]}``
    """
    with Timer("Parse GTEx metadata"):
        sample_df = pd.read_csv(RAW / "gtex_sample_attributes.txt", sep="\t", dtype=str)
        subject_df = pd.read_csv(RAW / "gtex_subject_phenotypes.txt", sep="\t", dtype=str)

        # Extract subject ID from sample ID (first two dash-separated fields)
        sample_df["SUBJID"] = sample_df["SAMPID"].str.extract(r"^(GTEX-[^-]+)")
        meta = sample_df.merge(subject_df, on="SUBJID", how="left")
        meta["AGE_MID"] = meta["AGE"].map(
            lambda x: age_midpoint(x, config["gtex"]["age_midpoints"])
        )
        # Keep only RNA-Seq samples with age information
        meta = meta.dropna(subset=["AGE_MID", "SMTSD"])
        save_parquet(meta, PROCESSED / "gtex_meta.parquet")

    # Identify tissues meeting minimum sample count
    tissue_counts = meta["SMTSD"].value_counts()
    target_tissues = [
        t for t in config["gtex"]["tissues"]
        if tissue_counts.get(t, 0) >= config["gtex"]["min_samples_per_tissue"]
    ]
    logger.info(f"Tissues passing sample threshold: {len(target_tissues)}/{len(config['gtex']['tissues'])}")

    # Read TPM matrix (large — stream to reduce memory)
    tpm_path = RAW / "gtex_tpm.gct.gz"
    with Timer("Parse GTEx TPM matrix"):
        # GCT format: 2 header lines, then data
        tpm = pd.read_csv(tpm_path, sep="\t", skiprows=2, index_col=0, compression="gzip")

        # Build Ensembl → HGNC symbol mapping from the Description column
        # (Description contains gene symbols in the GTEx GCT file)
        if "Description" in tpm.columns:
            ensembl_to_symbol = tpm["Description"].to_dict()
            tpm = tpm.drop(columns=["Description"])
        else:
            ensembl_to_symbol = {}

        tpm.index.name = "gene_id"

    # Save the mapping for potential later use
    if ensembl_to_symbol:
        mapping_df = pd.DataFrame([
            {"ensembl_id": k, "symbol": v}
            for k, v in ensembl_to_symbol.items()
            if pd.notna(v) and str(v).strip() != ""
        ])
        mapping_df.to_csv(PROCESSED / "ensembl_to_symbol.csv", index=False)
        logger.info(f"Ensembl→HGNC mapping: {len(mapping_df)} entries")

        # Convert index from Ensembl IDs to HGNC symbols
        # Map, drop unmapped, aggregate duplicates by max expression
        tpm.index = tpm.index.map(lambda x: ensembl_to_symbol.get(x, x))
        # Remove rows where mapping failed (still Ensembl IDs)
        is_ensembl = tpm.index.str.startswith("ENSG")
        n_unmapped = is_ensembl.sum()
        if n_unmapped > 0:
            logger.info(f"  Dropping {n_unmapped} unmapped Ensembl IDs")
            tpm = tpm.loc[~is_ensembl]
        # Aggregate duplicate symbols by taking the max (keep highest-expressed isoform)
        n_before = len(tpm)
        tpm = tpm.groupby(tpm.index).max()
        n_after = len(tpm)
        if n_before != n_after:
            logger.info(f"  Collapsed {n_before} → {n_after} unique gene symbols")
        tpm.index.name = "gene_id"

    # Per-tissue slicing and saving
    for tissue in target_tissues:
        tissue_safe = tissue.replace(" ", "_").replace("-", "").replace("(", "").replace(")", "")
        tissue_samples = meta.loc[meta["SMTSD"] == tissue, "SAMPID"].tolist()
        common = [s for s in tissue_samples if s in tpm.columns]
        if len(common) < config["gtex"]["min_samples_per_tissue"]:
            logger.warning(f"Skipping {tissue}: only {len(common)} overlapping samples")
            continue
        expr = tpm[common].copy()
        # log2(TPM + 1)
        expr = np.log2(expr + 1)
        save_parquet(expr, PROCESSED / f"gtex_expr_{tissue_safe}.parquet")
        logger.info(f"  {tissue}: {expr.shape[1]} samples, {expr.shape[0]} genes")

    return {"meta": meta, "tissues": target_tissues}


# ---------------------------------------------------------------------------
# STRING
# ---------------------------------------------------------------------------


def download_string(config: dict[str, Any]) -> bool:
    """Download STRING v12 interaction data for *H. sapiens*.

    Parameters
    ----------
    config : dict

    Returns
    -------
    bool
    """
    ok = True
    ok &= _download_file(DOWNLOAD_URLS["string_links"], RAW / "string_links.txt.gz", "STRING links")
    ok &= _download_file(DOWNLOAD_URLS["string_info"], RAW / "string_info.txt.gz", "STRING info")
    return ok


def parse_string(config: dict[str, Any]) -> pd.DataFrame:
    """Parse STRING into a filtered DataFrame of protein-protein interactions.

    Filters by ``combined_score >= secretome.string_score_threshold``.

    Parameters
    ----------
    config : dict

    Returns
    -------
    pd.DataFrame
        Columns: ``protein1, protein2, combined_score, gene1, gene2``.
    """
    threshold = config["secretome"]["string_score_threshold"]
    with Timer("Parse STRING interactions"):
        links = pd.read_csv(RAW / "string_links.txt.gz", sep=" ", compression="gzip")
        links = links[links["combined_score"] >= threshold].copy()

        info = pd.read_csv(RAW / "string_info.txt.gz", sep="\t", compression="gzip")
        prot2gene = dict(zip(info["#string_protein_id"], info["preferred_name"]))

        links["gene1"] = links["protein1"].map(prot2gene)
        links["gene2"] = links["protein2"].map(prot2gene)
        links = links.dropna(subset=["gene1", "gene2"])

    save_parquet(links, PROCESSED / "string_interactions.parquet")
    logger.info(f"STRING interactions (score >= {threshold}): {len(links)}")
    return links


# ---------------------------------------------------------------------------
# HPA Secretome
# ---------------------------------------------------------------------------


def download_hpa_secretome(config: dict[str, Any]) -> bool:
    """Download Human Protein Atlas secretome gene list.

    Parameters
    ----------
    config : dict

    Returns
    -------
    bool
    """
    return _download_file(HPA_SECRETOME_URL, RAW / "hpa_secretome.tsv", "HPA secretome")


def parse_hpa_secretome(config: dict[str, Any]) -> set[str]:
    """Parse HPA secretome list into a set of gene symbols.

    Parameters
    ----------
    config : dict

    Returns
    -------
    set[str]
        Gene symbols classified as secreted.
    """
    path = RAW / "hpa_secretome.tsv"
    if path.exists() and path.stat().st_size > 0:
        df = pd.read_csv(path, sep="\t")
        col = [c for c in df.columns if "gene" in c.lower() or "Gene" in c][0] if len(df.columns) > 0 else df.columns[0]
        genes = set(df[col].dropna().unique())
    else:
        logger.warning("HPA secretome file missing — using fallback curated list")
        genes = _fallback_secretome_genes()
    logger.info(f"Secretome genes: {len(genes)}")
    pd.DataFrame({"gene": sorted(genes)}).to_csv(PROCESSED / "secretome_genes.csv", index=False)
    return genes


def _fallback_secretome_genes() -> set[str]:
    """Return a curated set of ~200 well-known secreted protein genes as
    fallback when the HPA download is unavailable.

    Returns
    -------
    set[str]
    """
    return {
        "INS", "GH1", "IGF1", "IGF2", "IGFBP1", "IGFBP2", "IGFBP3", "IGFBP5",
        "TNF", "IL1A", "IL1B", "IL2", "IL4", "IL6", "IL8", "IL10", "IL13",
        "IL17A", "IL18", "IFNG", "TGFB1", "TGFB2", "TGFB3",
        "VEGFA", "VEGFB", "VEGFC", "FGF1", "FGF2", "PDGFA", "PDGFB",
        "EGF", "HGF", "CTGF", "BMP2", "BMP4", "BMP7",
        "WNT1", "WNT3A", "WNT5A", "WNT7A",
        "ADIPOQ", "LEP", "RETN", "NAMPT", "GDF15", "FGF21",
        "AGT", "REN", "ACE", "ACE2", "SERPINE1",
        "ALB", "FGA", "FGB", "FGG", "SERPINC1", "F2", "F7", "F8", "F9", "F10",
        "CRP", "SAA1", "HP", "HPX", "ORM1", "LBP", "APOA1", "APOB", "APOC3", "APOE",
        "MMP1", "MMP2", "MMP3", "MMP9", "MMP13", "TIMP1", "TIMP2",
        "CXCL1", "CXCL2", "CXCL5", "CXCL8", "CXCL10", "CXCL12",
        "CCL2", "CCL3", "CCL4", "CCL5", "CCL7", "CCL11", "CCL20",
        "CSF1", "CSF2", "CSF3", "EPO", "TPO", "KITLG", "FLT3LG",
        "ANGPT1", "ANGPT2", "ANGPTL4", "THBS1", "THBS2",
        "COL1A1", "COL1A2", "COL3A1", "FN1", "SPP1", "SPARC", "TNC",
        "CALCA", "PTH", "GCG", "SST", "NPY", "POMC", "OXT", "AVP",
        "PRL", "LHB", "FSHB", "CGA", "TSHB",
        "MSTN", "GDF11", "INHBA", "INHBB", "FST",
        "BDNF", "NGF", "NTF3", "NTF4", "GDNF",
        "LIF", "OSM", "CNTF", "CTF1",
        "EREG", "AREG", "HBEGF", "NRG1",
        "DKK1", "DKK3", "SFRP1", "SFRP2", "WIF1",
        "BTC", "GAS6", "PROS1",
        "SERPINA1", "SERPINA3", "A2M",
        "CP", "TF", "HAMP",
        "GPC3", "SDC1", "SDC4",
        "TNFSF10", "TNFSF11", "TNFSF13B", "FASLG",
        "C3", "C4A", "C5", "CFB", "CFD", "CFH",
        "LGALS1", "LGALS3", "LGALS9",
        "S100A8", "S100A9", "S100A12", "S100B",
        "HMGB1", "ANXA1", "ANXA2", "ANXA5",
        "CTSD", "CTSB", "CTSL", "CTSS",
        "PLAT", "PLAU", "PLG",
        "PROS1", "PROC",
        "PIGF", "TNFRSF11B",
        "CXCL16", "CX3CL1",
        "IL1RN", "IL1R2",
        "CHI3L1", "CHIT1", "LCN2",
        "RBP4", "TTR",
        "TIMP3", "TIMP4",
        "IGFBP4", "IGFBP6", "IGFBP7",
        "ANGPTL3", "ANGPTL6",
        "APOA2", "APOA4", "APOC1", "APOC2",
    }


# ---------------------------------------------------------------------------
# DrugBank (open-access XML)
# ---------------------------------------------------------------------------


def download_drugbank(config: dict[str, Any]) -> bool:
    """Attempt to download DrugBank open-data XML.

    DrugBank requires registration so automatic download may fail.  In that
    case the function prints manual instructions.

    Parameters
    ----------
    config : dict

    Returns
    -------
    bool
    """
    dest = RAW / "drugbank_open.xml"
    if dest.exists() and dest.stat().st_size > 1000:
        logger.info("DrugBank XML already present")
        return True
    url = "https://go.drugbank.com/releases/latest/downloads/all-open-access"
    ok = _download_file(url, RAW / "drugbank_open.zip", "DrugBank open-access")
    if ok:
        import zipfile
        with zipfile.ZipFile(RAW / "drugbank_open.zip") as zf:
            for name in zf.namelist():
                if name.endswith(".xml"):
                    zf.extract(name, RAW)
                    (RAW / name).rename(dest)
                    break
        return True
    logger.warning(
        "DrugBank automatic download failed. Please:\n"
        "  1. Register at https://go.drugbank.com/\n"
        "  2. Download the open-access XML release\n"
        "  3. Place the XML file at data/raw/drugbank_open.xml\n"
        "Continuing with synthetic placeholder data."
    )
    return False


def parse_drugbank(config: dict[str, Any]) -> pd.DataFrame:
    """Parse DrugBank open-access XML into a target table.

    Parameters
    ----------
    config : dict

    Returns
    -------
    pd.DataFrame
        Columns: ``drugbank_id, name, status, targets, atc_codes, indication``.
    """
    xml_path = RAW / "drugbank_open.xml"
    if not xml_path.exists() or xml_path.stat().st_size < 1000:
        logger.warning("DrugBank XML not found — generating synthetic data")
        return _synthetic_drugbank()

    with Timer("Parse DrugBank XML"):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {"db": "http://www.drugbank.ca"}

        rows: list[dict[str, Any]] = []
        for drug in root.findall("db:drug", ns):
            dbid = drug.findtext("db:drugbank-id[@primary='true']", default="", namespaces=ns)
            name = drug.findtext("db:name", default="", namespaces=ns)
            status_el = drug.find("db:groups", ns)
            groups = [g.text for g in (status_el.findall("db:group", ns) if status_el is not None else [])]
            status = "approved" if "approved" in groups else ("investigational" if "investigational" in groups else "other")
            indication = drug.findtext("db:indication", default="", namespaces=ns)[:500]

            targets_el = drug.find("db:targets", ns)
            target_genes: list[str] = []
            if targets_el is not None:
                for tgt in targets_el.findall("db:target", ns):
                    poly = tgt.find("db:polypeptide", ns)
                    if poly is not None:
                        gs = poly.findtext("db:gene-name", default="", namespaces=ns)
                        if gs:
                            target_genes.append(gs)

            atc_el = drug.find("db:atc-codes", ns)
            atc_list: list[str] = []
            if atc_el is not None:
                for code in atc_el.findall("db:atc-code", ns):
                    c = code.get("code", "")
                    if c:
                        atc_list.append(c)

            rows.append({
                "drugbank_id": dbid,
                "name": name,
                "status": status,
                "targets": ";".join(target_genes),
                "atc_codes": ";".join(atc_list),
                "indication": indication,
            })

    df = pd.DataFrame(rows)
    save_parquet(df, PROCESSED / "drugbank.parquet")
    logger.info(f"DrugBank entries: {len(df)}  (approved: {(df['status']=='approved').sum()})")
    return df


# ---------------------------------------------------------------------------
# GenAge / DrugAge
# ---------------------------------------------------------------------------


def download_genage_drugage(config: dict[str, Any]) -> bool:
    """Download GenAge and DrugAge datasets.

    Parameters
    ----------
    config : dict

    Returns
    -------
    bool
    """
    ok = True
    ok &= _download_file(DOWNLOAD_URLS["genage"], RAW / "genage.zip", "GenAge")
    ok &= _download_file(DOWNLOAD_URLS["drugage"], RAW / "drugage.zip", "DrugAge")
    return ok


def parse_genage(config: dict[str, Any]) -> set[str]:
    """Extract human aging-associated genes from GenAge.

    Parameters
    ----------
    config : dict

    Returns
    -------
    set[str]
        Gene symbols.
    """
    import zipfile
    zpath = RAW / "genage.zip"
    genes: set[str] = set()
    if zpath.exists():
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                if name.endswith(".csv") or name.endswith(".tsv") or name.endswith(".txt"):
                    with zf.open(name) as f:
                        df = pd.read_csv(f, sep=None, engine="python")
                        for col in df.columns:
                            if "symbol" in col.lower() or "gene" in col.lower():
                                genes.update(df[col].dropna().astype(str).unique())
                                break
    if not genes:
        logger.warning("GenAge parse failed — using curated fallback")
        genes = {
            "TP53", "SIRT1", "SIRT3", "SIRT6", "FOXO3", "MTOR", "IGF1R", "INS",
            "TERT", "LMNA", "WRN", "BLM", "ERCC1", "PARP1", "SOD1", "SOD2", "CAT",
            "GPX1", "FOXO1", "CDKN2A", "RB1", "ATM", "BRCA1", "CLK1", "AGTR1",
            "APOE", "GH1", "GHR", "PROP1", "POU1F1", "KLOTHO", "NFE2L2",
            "PPARGC1A", "AMPK", "PRKAA1", "PRKAA2", "UCP1", "UCP2", "UCP3",
            "HSF1", "HSPA1A", "HSPA1B", "HSP90AA1",
        }
    pd.DataFrame({"gene": sorted(genes)}).to_csv(PROCESSED / "genage_genes.csv", index=False)
    logger.info(f"GenAge genes: {len(genes)}")
    return genes


def parse_drugage(config: dict[str, Any]) -> pd.DataFrame:
    """Parse DrugAge database.

    Parameters
    ----------
    config : dict

    Returns
    -------
    pd.DataFrame
    """
    import zipfile
    zpath = RAW / "drugage.zip"
    if zpath.exists():
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                if name.endswith(".csv") or name.endswith(".tsv") or name.endswith(".txt"):
                    with zf.open(name) as f:
                        df = pd.read_csv(f, sep=None, engine="python")
                        save_parquet(df, PROCESSED / "drugage.parquet")
                        logger.info(f"DrugAge compounds: {len(df)}")
                        return df
    logger.warning("DrugAge parse failed — generating synthetic placeholder")
    return _synthetic_drugage()


# ---------------------------------------------------------------------------
# MSigDB Hallmark
# ---------------------------------------------------------------------------


def download_hallmark(config: dict[str, Any]) -> bool:
    """Download MSigDB Hallmark gene sets.

    Parameters
    ----------
    config : dict

    Returns
    -------
    bool
    """
    return _download_file(DOWNLOAD_URLS["hallmark_gmt"], RAW / "hallmark.gmt", "MSigDB Hallmark")


def parse_hallmark(config: dict[str, Any]) -> dict[str, list[str]]:
    """Parse a GMT file into a dict of gene-set name → gene list.

    Parameters
    ----------
    config : dict

    Returns
    -------
    dict[str, list[str]]
    """
    gmt: dict[str, list[str]] = {}
    path = RAW / "hallmark.gmt"
    if path.exists():
        with open(path) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    gmt[parts[0]] = parts[2:]
    else:
        logger.warning("Hallmark GMT not found")
    logger.info(f"Hallmark gene sets loaded: {len(gmt)}")
    return gmt


# ---------------------------------------------------------------------------
# LINCS L1000 (placeholder / GEO-based)
# ---------------------------------------------------------------------------


def download_lincs(config: dict[str, Any]) -> bool:
    """Attempt to download LINCS L1000 data.

    The full Level 5 data requires Clue.io registration.  This function
    provides instructions and can generate synthetic signatures as fallback.

    Parameters
    ----------
    config : dict

    Returns
    -------
    bool
    """
    dest = RAW / "lincs_level5.gctx"
    if dest.exists() and dest.stat().st_size > 1000:
        logger.info("LINCS data already present")
        return True

    logger.warning(
        "LINCS L1000 Level 5 data requires registration at https://clue.io.\n"
        "  1. Go to https://clue.io/data/CMap2020\n"
        "  2. Download level5_beta_trt_cp_n720216x12328.gctx\n"
        "  3. Also download siginfo_beta.txt and compoundinfo_beta.txt\n"
        "  4. Place all files in data/raw/\n"
        "Generating synthetic LINCS signatures for pipeline testing."
    )
    return False


def parse_lincs(config: dict[str, Any]) -> pd.DataFrame:
    """Parse or generate LINCS drug signatures.

    If real LINCS data is present, parse it.  Otherwise generate
    synthetic signatures that mimic the real data structure.

    Parameters
    ----------
    config : dict

    Returns
    -------
    pd.DataFrame
        Rows = genes (~12 000), columns = perturbation IDs.
        Also saves metadata table.
    """
    gctx_path = RAW / "lincs_level5.gctx"
    if gctx_path.exists() and gctx_path.stat().st_size > 1000:
        return _parse_real_lincs(config)
    logger.info("Using synthetic LINCS signatures")
    return _synthetic_lincs(config)


def _parse_real_lincs(config: dict[str, Any]) -> pd.DataFrame:
    """Parse real LINCS GCTX + metadata (if present).

    Parameters
    ----------
    config : dict

    Returns
    -------
    pd.DataFrame
    """
    try:
        import cmapPy.pandasGEXpress.parse_gctx as pg
        gctx = pg.parse(str(RAW / "lincs_level5.gctx"))
        expr = gctx.data_df
    except ImportError:
        logger.warning("cmapPy not installed — falling back to synthetic LINCS")
        return _synthetic_lincs(config)

    # Load metadata
    sig_path = RAW / "siginfo_beta.txt"
    comp_path = RAW / "compoundinfo_beta.txt"
    if sig_path.exists() and comp_path.exists():
        siginfo = pd.read_csv(sig_path, sep="\t", dtype=str)
        compinfo = pd.read_csv(comp_path, sep="\t", dtype=str)
        # Filter: trt_cp, 24h, max dose, selected cell lines
        cell_lines = config["drug"]["lincs_cell_lines"]
        mask = (
            (siginfo["pert_type"] == "trt_cp") &
            (siginfo["pert_time"].astype(str).str.contains("24")) &
            (siginfo["cell_iname"].isin(cell_lines))
        )
        siginfo = siginfo[mask]
        # Keep best signature per compound-cell line (max dose)
        siginfo["pert_dose"] = pd.to_numeric(siginfo["pert_dose"], errors="coerce")
        siginfo = siginfo.sort_values("pert_dose", ascending=False).drop_duplicates(
            subset=["pert_iname", "cell_iname"], keep="first"
        )
        keep_sigs = siginfo["sig_id"].tolist()
        common_sigs = [s for s in keep_sigs if s in expr.columns]
        expr = expr[common_sigs]
        siginfo = siginfo[siginfo["sig_id"].isin(common_sigs)]
        save_parquet(siginfo, PROCESSED / "lincs_metadata.parquet")

    save_parquet(expr, PROCESSED / "lincs_signatures.parquet")
    logger.info(f"LINCS signatures: {expr.shape[1]} perturbations × {expr.shape[0]} genes")
    return expr


# ---------------------------------------------------------------------------
# Synthetic data generators (fallback)
# ---------------------------------------------------------------------------


def _synthetic_lincs(config: dict[str, Any]) -> pd.DataFrame:
    """Generate synthetic LINCS-like drug signatures for pipeline testing.

    Creates ~500 drug perturbation signatures across ~1000 landmark genes.

    Parameters
    ----------
    config : dict

    Returns
    -------
    pd.DataFrame
    """
    set_seed(42)
    n_drugs = 500
    n_genes = 1000
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    # Add some real gene names so downstream analyses can find overlaps
    real_genes = [
        "TP53", "MTOR", "SIRT1", "CDKN2A", "IGF1", "IL6", "TNF", "TGFB1",
        "VEGFA", "MMP9", "SERPINE1", "FN1", "COL1A1", "FOXO3", "NFE2L2",
        "SOD2", "CAT", "GPX1", "HMOX1", "NQO1", "HSPA1A", "BCL2", "BAX",
        "CASP3", "CASP9", "ATG5", "BECN1", "MAP1LC3B", "SQSTM1",
        "PPARGC1A", "TFAM", "NRF1", "POLG", "MFN2", "DRP1",
        "LMNA", "TERT", "TERC", "WRN", "PARP1", "ERCC1",
        "CDKN1A", "RB1", "CCND1", "CDK4", "CDK6", "E2F1",
        "AKT1", "PIK3CA", "PTEN", "TSC2", "RPTOR", "RICTOR",
    ]
    gene_names[:len(real_genes)] = real_genes

    drug_names = []
    approved_drugs = [
        "metformin", "rapamycin", "resveratrol", "aspirin", "lithium",
        "acarbose", "spermidine", "NAD+", "dasatinib", "quercetin",
        "fisetin", "navitoclax", "ruxolitinib", "baricitinib", "dexamethasone",
        "pioglitazone", "canagliflozin", "empagliflozin", "atorvastatin",
        "rosuvastatin", "simvastatin", "losartan", "captopril", "ramipril",
        "amlodipine", "vitamin_D", "vitamin_E", "selenium", "zinc",
        "omega3", "curcumin", "berberine", "sulforaphane", "pterostilbene",
    ]
    drug_names.extend(approved_drugs)
    for i in range(n_drugs - len(approved_drugs)):
        drug_names.append(f"COMPOUND_{i}")

    data = np.random.randn(n_genes, n_drugs).astype(np.float32)
    # Make known anti-aging drugs show stronger reversal patterns
    for i, dn in enumerate(drug_names[:len(approved_drugs)]):
        # Upregulate protective genes, downregulate inflammatory ones
        data[:5, i] -= 1.5   # suppress aging hallmark genes
        data[10:20, i] += 1.2  # activate protective pathways

    df = pd.DataFrame(data, index=gene_names, columns=drug_names)

    meta_rows = []
    for i, dn in enumerate(drug_names):
        cell = config["drug"]["lincs_cell_lines"][i % len(config["drug"]["lincs_cell_lines"])]
        meta_rows.append({
            "sig_id": dn,
            "pert_iname": dn,
            "cell_iname": cell,
            "pert_type": "trt_cp",
            "pert_dose": 10.0,
            "pert_time": "24h",
        })
    meta = pd.DataFrame(meta_rows)

    save_parquet(df, PROCESSED / "lincs_signatures.parquet")
    save_parquet(meta, PROCESSED / "lincs_metadata.parquet")
    logger.info(f"Synthetic LINCS: {n_drugs} drugs × {n_genes} genes")
    return df


def _synthetic_drugbank() -> pd.DataFrame:
    """Generate synthetic DrugBank data.

    Returns
    -------
    pd.DataFrame
    """
    drugs = [
        ("DB00945", "Aspirin", "approved", "PTGS1;PTGS2", "N02BA01", "Pain, inflammation"),
        ("DB00331", "Metformin", "approved", "PRKAA1;PRKAA2", "A10BA02", "Type 2 diabetes"),
        ("DB00877", "Sirolimus", "approved", "MTOR;FKBP1A", "L04AD02", "Immunosuppression"),
        ("DB01076", "Atorvastatin", "approved", "HMGCR", "C10AA05", "Hypercholesterolemia"),
        ("DB00641", "Simvastatin", "approved", "HMGCR", "C10AA01", "Hypercholesterolemia"),
        ("DB01004", "Glyburide", "approved", "ABCC8;KCNJ11", "A10BB01", "Type 2 diabetes"),
        ("DB01050", "Ibuprofen", "approved", "PTGS1;PTGS2", "M01AE01", "Pain, inflammation"),
        ("DB00563", "Methotrexate", "approved", "DHFR", "L01BA01", "Cancer, autoimmune"),
        ("DB00999", "Hydrochlorothiazide", "approved", "SLC12A3", "C03AA03", "Hypertension"),
        ("DB01197", "Captopril", "approved", "ACE", "C09AA01", "Hypertension"),
        ("DB00678", "Losartan", "approved", "AGTR1", "C09CA01", "Hypertension"),
        ("DB01015", "Ramipril", "approved", "ACE", "C09AA05", "Hypertension"),
        ("DB00381", "Amlodipine", "approved", "CACNA1C;CACNA1D", "C08CA01", "Hypertension"),
        ("DB00175", "Pravastatin", "approved", "HMGCR", "C10AA03", "Hypercholesterolemia"),
        ("DB01098", "Rosuvastatin", "approved", "HMGCR", "C10AA07", "Hypercholesterolemia"),
        ("DB00806", "Pentoxifylline", "approved", "PDE4B;TNF", "C04AD03", "Peripheral vascular disease"),
        ("DB00390", "Digoxin", "approved", "ATP1A1", "C01AA05", "Heart failure"),
        ("DB06292", "Dapagliflozin", "approved", "SLC5A2", "A10BK01", "Type 2 diabetes"),
        ("DB11827", "Canagliflozin", "approved", "SLC5A2;SLC5A1", "A10BK02", "Type 2 diabetes"),
        ("DB09038", "Empagliflozin", "approved", "SLC5A2", "A10BK03", "Type 2 diabetes"),
        ("DB01120", "Gliclazide", "approved", "ABCC8;KCNJ11", "A10BB09", "Type 2 diabetes"),
        ("DB00945", "Lithium", "approved", "GSK3B;IMPase", "N05AN01", "Bipolar disorder"),
        ("DB00197", "Troglitazone", "approved", "PPARG", "A10BG01", "Type 2 diabetes"),
        ("DB01132", "Pioglitazone", "approved", "PPARG", "A10BG03", "Type 2 diabetes"),
        ("DB01024", "Acarbose", "approved", "GAA;MGAM", "A10BF01", "Type 2 diabetes"),
        ("DB00619", "Imatinib", "approved", "ABL1;KIT;PDGFRA", "L01EA01", "Cancer"),
        ("DB01254", "Dasatinib", "approved", "ABL1;SRC;KIT", "L01EA02", "Cancer"),
        ("DB08901", "Ruxolitinib", "approved", "JAK1;JAK2", "L01EJ01", "Myelofibrosis"),
        ("DB11817", "Baricitinib", "approved", "JAK1;JAK2", "L04AA37", "Rheumatoid arthritis"),
        ("DB01234", "Dexamethasone", "approved", "NR3C1", "H02AB02", "Inflammation"),
        ("DB00959", "Quercetin", "investigational", "PIK3CG;SRC;EGFR", "", "Under investigation"),
        ("DB16446", "Fisetin", "investigational", "CDK6;SIRT1;NF-KB", "", "Under investigation"),
    ]
    df = pd.DataFrame(drugs, columns=["drugbank_id", "name", "status", "targets", "atc_codes", "indication"])
    save_parquet(df, PROCESSED / "drugbank.parquet")
    logger.info(f"Synthetic DrugBank: {len(df)} drugs")
    return df


def _synthetic_drugage() -> pd.DataFrame:
    """Generate synthetic DrugAge table.

    Returns
    -------
    pd.DataFrame
    """
    rows = [
        {"compound_name": "Metformin", "species": "Homo sapiens", "avg_lifespan_change": 0.04},
        {"compound_name": "Rapamycin", "species": "Mus musculus", "avg_lifespan_change": 0.14},
        {"compound_name": "Resveratrol", "species": "Mus musculus", "avg_lifespan_change": 0.05},
        {"compound_name": "Acarbose", "species": "Mus musculus", "avg_lifespan_change": 0.11},
        {"compound_name": "Aspirin", "species": "Mus musculus", "avg_lifespan_change": 0.08},
        {"compound_name": "Spermidine", "species": "Mus musculus", "avg_lifespan_change": 0.10},
        {"compound_name": "NAD+", "species": "Mus musculus", "avg_lifespan_change": 0.05},
        {"compound_name": "Lithium", "species": "Drosophila", "avg_lifespan_change": 0.16},
        {"compound_name": "Quercetin", "species": "C. elegans", "avg_lifespan_change": 0.15},
        {"compound_name": "Fisetin", "species": "Mus musculus", "avg_lifespan_change": 0.10},
        {"compound_name": "Dasatinib", "species": "Mus musculus", "avg_lifespan_change": 0.03},
        {"compound_name": "Pioglitazone", "species": "Mus musculus", "avg_lifespan_change": 0.06},
    ]
    df = pd.DataFrame(rows)
    save_parquet(df, PROCESSED / "drugage.parquet")
    logger.info(f"Synthetic DrugAge: {len(df)} entries")
    return df


# ---------------------------------------------------------------------------
# Synthetic GTEx generator
# ---------------------------------------------------------------------------


def generate_synthetic_gtex(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Generate realistic synthetic GTEx-like data for pipeline testing.

    Creates expression matrices for all configured tissues with realistic
    gene-age correlations, inter-individual variation, and tissue-specific
    expression patterns.

    Parameters
    ----------
    config : dict

    Returns
    -------
    dict
        ``{"meta": DataFrame, "tissues": list[str]}``
    """
    set_seed(42)
    tissues = config["gtex"]["tissues"]
    n_genes = 20000
    ages = [25, 35, 45, 55, 65, 75]
    samples_per_age = 30  # ~180 per tissue

    # Gene names: mix of real and synthetic
    real_gene_names = [
        "TP53", "MTOR", "SIRT1", "SIRT6", "FOXO3", "IGF1", "IGF1R", "INS",
        "IL6", "IL1B", "TNF", "TGFB1", "CDKN2A", "CDKN1A", "RB1", "LMNA",
        "TERT", "WRN", "PARP1", "SOD2", "CAT", "GPX1", "NFE2L2", "HMOX1",
        "PPARGC1A", "TFAM", "NRF1", "MFN2", "BECN1", "ATG5", "SQSTM1",
        "MAP1LC3B", "BCL2", "BAX", "CASP3", "VEGFA", "MMP9", "SERPINE1",
        "FN1", "COL1A1", "COL3A1", "SPP1", "SPARC", "CCL2", "CXCL8",
        "CRP", "SAA1", "HP", "ALB", "FGA", "APOE", "APOB",
        "GH1", "GHR", "LEP", "ADIPOQ", "RETN", "GDF15", "FGF21",
        "HSPA1A", "HSP90AA1", "GAPDH", "ACTB", "B2M", "PPIA",
        "MMP1", "MMP2", "MMP3", "TIMP1", "TIMP2", "EGF", "HGF",
        "BDNF", "NGF", "GDNF", "MSTN", "GDF11", "FST",
        "CXCL12", "CCL5", "CXCL10", "IL10", "IL17A", "IFNG",
        "PTGS2", "NOS2", "ARG1", "IDO1", "HMGB1",
        "S100A8", "S100A9", "LCN2", "CHI3L1",
        "IGFBP1", "IGFBP2", "IGFBP3", "IGFBP5", "IGFBP7",
        "ANGPT1", "ANGPT2", "ANGPTL4", "THBS1",
        "NAMPT", "NNMT", "ACMSD",
        # Housekeeping (should not correlate with age)
        "RPS18", "RPL13A", "HPRT1", "TBP", "YWHAZ", "UBC", "HMBS",
    ]
    gene_names = real_gene_names + [f"GENE_{i}" for i in range(n_genes - len(real_gene_names))]
    gene_names = gene_names[:n_genes]

    # Define aging genes (first ~80 genes have age correlations)
    n_aging_up = 40    # genes increasing with age
    n_aging_down = 40  # genes decreasing with age

    all_meta_rows: list[dict] = []
    subject_counter = 0

    for tissue in tissues:
        tissue_safe = tissue.replace(" ", "_").replace("-", "").replace("(", "").replace(")", "")
        sample_ids: list[str] = []
        ages_list: list[float] = []
        sex_list: list[str] = []

        for age in ages:
            for j in range(samples_per_age):
                subject_counter += 1
                sid = f"GTEX-{subject_counter:05d}-{tissue_safe[:4]}"
                subjid = f"GTEX-{subject_counter:05d}"
                sex = np.random.choice(["1", "2"])  # 1=male, 2=female
                sample_ids.append(sid)
                ages_list.append(float(age))
                sex_list.append(sex)

                all_meta_rows.append({
                    "SAMPID": sid,
                    "SUBJID": subjid,
                    "SMTSD": tissue,
                    "SMTSISCH": str(np.random.randint(100, 1500)),
                    "SMNABTCH": f"BATCH_{np.random.randint(1, 10)}",
                    "AGE": f"{int(age)-5}-{int(age)+4}",
                    "SEX": sex,
                    "DTHHRDY": str(np.random.randint(0, 5)),
                    "AGE_MID": age,
                })

        n_samples = len(sample_ids)
        ages_arr = np.array(ages_list)
        ages_norm = (ages_arr - ages_arr.mean()) / ages_arr.std()

        # Base expression: log-normal-ish
        base_expr = np.random.exponential(2.0, size=(n_genes, 1)) + 0.5
        noise = np.random.randn(n_genes, n_samples) * 0.8

        expr = np.log2(base_expr + 1) + noise

        # Add age effects
        age_effects = np.zeros((n_genes, n_samples))
        # Up-aging genes
        for g in range(n_aging_up):
            slope = np.random.uniform(0.02, 0.10)
            age_effects[g, :] = slope * ages_norm
        # Down-aging genes
        for g in range(n_aging_up, n_aging_up + n_aging_down):
            slope = np.random.uniform(-0.10, -0.02)
            age_effects[g, :] = slope * ages_norm
        # Tissue-specific variation in effect sizes
        tissue_idx = tissues.index(tissue) if tissue in tissues else 0
        np.random.seed(42 + tissue_idx)
        tissue_scaling = np.random.uniform(0.5, 2.0, size=n_genes)
        age_effects *= tissue_scaling[:, np.newaxis]

        expr += age_effects

        # Make housekeeping genes stable (last ~7 of real genes before GENE_ prefix)
        hk_start = len(real_gene_names) - 7
        hk_end = len(real_gene_names)
        expr[hk_start:hk_end, :] = np.random.randn(hk_end - hk_start, n_samples) * 0.1 + 8.0

        expr_df = pd.DataFrame(expr, index=gene_names, columns=sample_ids)
        save_parquet(expr_df, PROCESSED / f"gtex_expr_{tissue_safe}.parquet")

    meta = pd.DataFrame(all_meta_rows)
    meta["AGE_MID"] = meta["AGE_MID"].astype(float)
    save_parquet(meta, PROCESSED / "gtex_meta.parquet")

    logger.info(
        f"Synthetic GTEx data generated: {len(tissues)} tissues, "
        f"~{samples_per_age * len(ages)} samples/tissue, {n_genes} genes"
    )
    return {"meta": meta, "tissues": tissues}


# ---------------------------------------------------------------------------
# Master download & parse entry point
# ---------------------------------------------------------------------------


def download_all(config: dict[str, Any]) -> dict[str, Any]:
    """Download and parse all required datasets.

    If a download fails and ``pipeline.use_synthetic_fallback`` is *True*,
    synthetic data is generated instead.

    Parameters
    ----------
    config : dict
        Global configuration.

    Returns
    -------
    dict
        Keys: ``gtex_ok, string_ok, hpa_ok, drugbank_ok, genage_ok, lincs_ok,
        hallmark_ok, use_synthetic``.
    """
    ensure_dir(RAW)
    ensure_dir(PROCESSED)
    ensure_dir(EXTERNAL)
    use_synthetic = config.get("pipeline", {}).get("use_synthetic_fallback", True)

    status: dict[str, Any] = {"use_synthetic": use_synthetic}

    # ---- GTEx ----
    with Timer("GTEx download"):
        gtex_ok = download_gtex(config)
    status["gtex_ok"] = gtex_ok
    if gtex_ok:
        with Timer("GTEx parsing"):
            parse_gtex(config)
    elif use_synthetic:
        with Timer("Generating synthetic GTEx"):
            generate_synthetic_gtex(config)

    # ---- STRING ----
    with Timer("STRING download"):
        string_ok = download_string(config)
    status["string_ok"] = string_ok
    if string_ok:
        with Timer("STRING parsing"):
            parse_string(config)
    elif use_synthetic:
        _generate_synthetic_string(config)

    # ---- HPA Secretome ----
    with Timer("HPA Secretome download"):
        hpa_ok = download_hpa_secretome(config)
    status["hpa_ok"] = hpa_ok
    parse_hpa_secretome(config)  # handles fallback internally

    # ---- DrugBank ----
    with Timer("DrugBank download"):
        db_ok = download_drugbank(config)
    status["drugbank_ok"] = db_ok
    parse_drugbank(config)  # handles fallback internally

    # ---- GenAge / DrugAge ----
    with Timer("GenAge/DrugAge download"):
        ga_ok = download_genage_drugage(config)
    status["genage_ok"] = ga_ok
    parse_genage(config)
    parse_drugage(config)

    # ---- LINCS ----
    with Timer("LINCS download"):
        lincs_ok = download_lincs(config)
    status["lincs_ok"] = lincs_ok
    parse_lincs(config)

    # ---- MSigDB Hallmark ----
    with Timer("Hallmark download"):
        hm_ok = download_hallmark(config)
    status["hallmark_ok"] = hm_ok
    parse_hallmark(config)

    logger.info(f"Data acquisition complete. Status: {status}")
    return status


def _generate_synthetic_string(config: dict[str, Any]) -> pd.DataFrame:
    """Generate synthetic STRING-like interaction data.

    Parameters
    ----------
    config : dict

    Returns
    -------
    pd.DataFrame
    """
    set_seed(42)
    # Use secretome fallback genes + some extra
    genes = sorted(_fallback_secretome_genes())[:150]
    rows = []
    for i in range(len(genes)):
        n_partners = np.random.randint(2, 15)
        partners = np.random.choice(len(genes), n_partners, replace=False)
        for j in partners:
            if i != j:
                score = np.random.randint(700, 1000)
                rows.append({
                    "protein1": f"9606.ENSP{i:010d}",
                    "protein2": f"9606.ENSP{j:010d}",
                    "combined_score": score,
                    "gene1": genes[i],
                    "gene2": genes[j],
                })
    df = pd.DataFrame(rows)
    save_parquet(df, PROCESSED / "string_interactions.parquet")
    logger.info(f"Synthetic STRING: {len(df)} interactions")
    return df
