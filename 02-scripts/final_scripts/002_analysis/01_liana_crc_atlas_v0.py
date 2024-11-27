# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: liana_2024
#     language: python
#     name: liana_2024
# ---

# # LIANA tumor vs normal core atlas v0

# ## Libraries

import numpy as  np
import pandas as pd
import scanpy as sc
import decoupler as dc
import liana as li
from liana.method import singlecellsignalr, connectome, cellphonedb, natmi, logfc, cellchat, geometric_mean
import sc_atlas_helpers as ah
from tqdm.auto import tqdm
import contextlib
import os
import statsmodels.stats.multitest
import numpy as np
from anndata import AnnData
import scipy.sparse

# ## Define variables, paths and comparison tumor vs normal

resDir ="/data/projects/2022/CRCA/results/v1/final/liana_cell2cell/h5ads/updated/"

resDir

adata = sc.read_h5ad("/data/projects/2022/CRCA/results/v1/downstream_analyses/Prepare_de_analysis/artifacts/paired_tumor_normal-adata.h5ad")

adata.obs.sample_type.value_counts()

adata.obs.sample_type.value_counts()

adata.obs.cell_type_coarse.value_counts()

adata.obs.cell_type_middle.value_counts()

adata.obs.cell_type_fine.value_counts()

set(adata.obs.cell_type_fine)


adata = adata[(adata.obs["sample_type"]=="tumor") & ~(adata.obs["sample_type"].isin(["Enteroendocrine","NKT"]))].copy()

# +
# %%
# Make epithelial-tumor labels the same for comparisson
cluster_annot = {
    "Monocyte classical": "Monocyte classical",
    "Monocyte non-classical": "Monocyte non-classical",
    "Macrophage": "Macrophage",
    "Macrophage cycling": "Macrophage cycling",
    "Myeloid progenitor": "Myeloid progenitor",
    "cDC1": "cDC1",
    "cDC2": "cDC2",
    "DC3": "DC3",
    "pDC": "pDC",
    "DC mature": "DC mature",
    "Granulocyte progenitor": "Neutrophil",
    "Neutrophil": "Neutrophil",
    "Eosinophil": "Eosinophil",
    "Mast cell": "Mast cell",
    "Platelet": "Platelet",
    "CD4": "T cell CD4",
    "Treg": "T cell regulatory",
    "CD8": "T cell CD8",
    "NK": "NK",
    "ILC": "ILC",
    "gamma-delta": "gamma-delta",
    "NKT": "NKT",
    "CD4 naive": "T cell CD4 naive",
    "CD8 naive": "T cell CD8 naive",
    "CD4 stem-like": "T cell CD4 stem-like",
    "CD8 stem-like": "T cell CD8 stem-like",
    "CD4 cycling": "T cell CD4 cycling",
    "CD8 cycling": "T cell CD8 cycling",
    "GC B cell": "GC B cell",
    "B cell naive": "B cell naive",
    "B cell activated naive": "B cell activated",
    "B cell activated": "B cell activated",
    "B cell memory": "B cell memory",
    "Plasma IgA": "Plasma IgA",
    "Plasma IgG": "Plasma IgG",
    "Plasma IgM": "Plasma IgM",
    "Plasmablast": "Plasmablast",
    "Crypt cell": "Cancer Crypt-like",
    "TA progenitor": "Cancer TA-like",
    "Colonocyte": "Cancer Colonocyte-like",
    "Colonocyte BEST4": "Cancer BEST4",
    "Goblet": "Cancer Goblet-like",
    "Tuft": "Tuft",
    "Enteroendocrine": "Enteroendocrine",
    "Cancer Colonocyte-like": "Cancer Colonocyte-like",
    "Cancer BEST4": "Cancer BEST4",
    "Cancer Goblet-like": "Cancer Goblet-like",
    "Cancer Crypt-like": "Cancer Crypt-like",
    "Cancer TA-like": "Cancer TA-like",
    "Cancer cell circulating": "Cancer cell circulating",
    "Endothelial venous": "Endothelial venous",
    "Endothelial arterial": "Endothelial arterial",
    "Endothelial lymphatic": "Endothelial lymphatic",
    "Fibroblast S1": "Fibroblast S1",
    "Fibroblast S2": "Fibroblast S2",
    "Fibroblast S3": "Fibroblast S3",
    "Pericyte": "Pericyte",
    "Schwann cell": "Schwann cell",
    "Hepatocyte": "Hepatocyte",
    "Fibroblastic reticular cell": "Fibroblastic reticular cell",
    "Epithelial reticular cell": "Epithelial reticular cell",
}
adata.obs["cell_type"] = (
    adata.obs["cell_type_fine"].map(cluster_annot)
)

# %%
cluster_annot = {
    "Monocyte classical": "Monocyte",
    "Monocyte non-classical": "Monocyte",
    "Macrophage": "Macrophage",
    "Macrophage cycling": "Macrophage",
    "Myeloid progenitor": "Dendritic cell",
    "cDC1": "Dendritic cell",
    "cDC2": "Dendritic cell",
    "DC3": "Dendritic cell",
    "pDC": "Dendritic cell",
    "DC mature": "Dendritic cell",
    "Granulocyte progenitor": "Neutrophil",
    "Neutrophil": "Neutrophil",
    "Eosinophil": "Eosinophil",
    "Mast cell": "Mast cell",
    "Platelet": "Platelet",
    "CD4": "T cell CD4",
    "Treg": "T cell CD4",
    "CD8": "T cell CD8",
    "NK": "NK",
    "ILC": "ILC",
    "gamma-delta": "gamma-delta",
    "NKT": "NKT",
    "CD4 naive": "T cell CD4",
    "CD8 naive": "T cell CD8",
    "CD4 stem-like": "T cell CD4",
    "CD8 stem-like": "T cell CD8",
    "CD4 cycling": "T cell CD4",
    "CD8 cycling": "T cell CD8",
    "GC B cell": "B cell",
    "B cell naive": "B cell",
    "B cell activated naive": "B cell",
    "B cell activated": "B cell",
    "B cell memory": "B cell",
    "Plasma IgA": "Plasma cell",
    "Plasma IgG": "Plasma cell",
    "Plasma IgM": "Plasma cell",
    "Plasmablast": "Plasma cell",
    "Crypt cell": "Cancer stem-like",
    "TA progenitor": "Cancer stem-like",
    "Colonocyte": "Cancer non-stem-like",
    "Colonocyte BEST4": "Cancer non-stem-like",
    "Goblet": "Cancer non-stem-like",
    "Tuft": "Tuft",
    "Enteroendocrine": "Enteroendocrine",
    "Cancer Colonocyte-like": "Cancer cell",
    "Cancer BEST4": "Cancer cell",
    "Cancer Goblet-like": "Cancer cell",
    "Cancer Crypt-like": "Cancer cell",
    "Cancer TA-like": "Cancer cell",
    "Cancer cell circulating": "Cancer cell circulating",
    "Endothelial venous": "Endothelial cell",
    "Endothelial arterial": "Endothelial cell",
    "Endothelial lymphatic": "Endothelial cell",
    "Fibroblast S1": "Fibroblast",
    "Fibroblast S2": "Fibroblast",
    "Fibroblast S3": "Fibroblast",
    "Pericyte": "Pericyte",
    "Schwann cell": "Schwann cell",
    "Hepatocyte": "Hepatocyte",
    "Fibroblastic reticular cell": "Fibroblastic reticular cell",
    "Epithelial reticular cell": "Epithelial reticular cell",
}
adata.obs["cell_type_coarse"] = (
    adata.obs["cell_type_fine"].map(cluster_annot)
)

# %%
adata.obs["cell_type_lineage"] = adata.obs["cell_type_coarse"].astype(str).replace(
    {
        "Cancer non-stem-like": "Cancer cell",
        "Cancer stem-like": "Cancer cell",
        "Cancer cell circulating": "Cancer cell",
        "T cell CD8": "T cell",
        "T cell CD4": "T cell",
        "gamma-delta": "T cell",
        "NKT": "T cell",
        "Plasma cell": "B cell",
        "Macrophage": "Myeloid cell",
        "Monocyte": "Myeloid cell",
        "Dendritic cell": "Myeloid cell",
        "Granulocyte": "Myeloid cell",
        "Fibroblast": "Stromal cell",
        "Endothelial cell": "Stromal cell",
        "Pericyte": "Stromal cell",
        "Schwann cell": "Stromal cell",
    }
)
# -

set(adata.obs.cell_type)

set(adata.obs.cell_type_coarse)

adata[adata.obs["sample_type"]=="tumor"].obs.cell_type_coarse.value_counts()

set(adata.obs.cell_type_fine)


# ## LIANA- rank agregate

adata.layers["log1p_norm"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e6, layer="log1p_norm")
sc.pp.log1p(adata, base=6, layer="log1p_norm")

# Run rank_aggregate 
li.mt.rank_aggregate(adata, groupby='cell_type_coarse', expr_prop=0.1,resource_name='consensus',  verbose=True,key_added='rank_aggregate', n_jobs=6, layer = "log1p_norm", use_raw = False)

#Save adata with new ranked information
adata.write_h5ad(f"{resDir}/adata_rank_agregate.h5ad")


