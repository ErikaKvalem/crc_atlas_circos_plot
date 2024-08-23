# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: liana_2024
#     language: python
#     name: liana_2024
# ---

# %%

# %%
import numpy as  np
import pandas as pd
import scanpy as sc
import decoupler as dc

# import liana
import liana as li
from liana.method import singlecellsignalr, connectome, cellphonedb, natmi, logfc, cellchat, geometric_mean

import sc_atlas_helpers as ah
#from scanpy_helper_submodule import scanpy_helpers as sh

# %%
from tqdm.auto import tqdm
import contextlib
import os
import statsmodels.stats.multitest
import numpy as np
from anndata import AnnData
import scipy.sparse


# %% [markdown]
# # Functions

# %%
def fdr_correction(df, pvalue_col="pvalue", *, key_added="fdr", inplace=False):
    """Adjust p-values in a data frame with test results using FDR correction."""
    if not inplace:
        df = df.copy()

    df[key_added] = statsmodels.stats.multitest.fdrcorrection(df[pvalue_col].values)[1]

    if not inplace:
        return df


# %%
"""Plotting functions for group comparisons"""

import altair as alt
import pandas as pd
import numpy as np


def plot_lm_result_altair(
    df,
    p_cutoff=0.1,
    p_col="fdr",
    x="variable",
    y="group",
    color="coef",
    title="heatmap",
    cluster=False,
    value_max=None,
    configure=lambda x: x.configure_mark(opacity=1),
    cmap="redblue",
    reverse=True,
    domain=lambda x: [-x, x],
    order=None,
):
    """
    Plot a results data frame of a comparison as a heatmap
    """
    df_filtered = df.loc[lambda _: _[p_col] < p_cutoff, :]
    df_subset = df.loc[
        lambda _: _[x].isin(df_filtered[x].unique()) & _[y].isin(df[y].unique())
    ]
    if not df_subset.shape[0]:
        print("No values to plot")
        return

    if order is None:
        order = "ascending"
        if cluster:
            from scipy.cluster.hierarchy import linkage, leaves_list

            values_df = df_subset.pivot(index=y, columns=x, values=color)
            order = values_df.columns.values[
                leaves_list(
                    linkage(values_df.values.T, method="average", metric="euclidean")
                )
            ]

    def _get_significance(fdr):
        if fdr < 0.001:
            return "< 0.001"
        elif fdr < 0.01:
            return "< 0.01"
        elif fdr < 0.1:
            return "< 0.1"
        else:
            return np.nan

    df_subset["FDR"] = pd.Categorical([_get_significance(x) for x in df_subset[p_col]])

    if value_max is None:
        value_max = max(
            abs(np.nanmin(df_subset[color])), abs(np.nanmax(df_subset[color]))
        )
    # just setting the domain in altair will lead to "black" fields. Therefore, we constrain the values themselves.
    df_subset[color] = np.clip(df_subset[color], *domain(value_max))
    return configure(
        alt.Chart(df_subset, title=title)
        .mark_rect()
        .encode(
            x=alt.X(x, sort=order),
            y=y,
            color=alt.Color(
                color,
                scale=alt.Scale(scheme=cmap, reverse=reverse, domain=domain(value_max)),
            ),
        )
        + alt.Chart(df_subset.loc[lambda x: ~x["FDR"].isnull()])
        .mark_point(color="white", filled=True, stroke="black", strokeWidth=0)
        .encode(
            x=alt.X(x, sort=order),
            y=y,
            size=alt.Size(
                "FDR:N",
                scale=alt.Scale(
                    domain=["< 0.001", "< 0.01", "< 0.1"],
                    range=4 ** np.array([3, 2, 1]),
                ),
            ),
        )
    )


# %%
from typing import Sequence, Union
from anndata import AnnData, ImplicitModificationWarning
import numpy as np
import pandas as pd
from operator import and_
from functools import reduce
import warnings


def pseudobulk(
    adata,
    *,
    groupby: Union[str, Sequence[str]],
    aggr_fun=np.sum,
    min_obs=10,
) -> AnnData:
    """
    Calculate Pseudobulk of groups

    Parameters
    ----------
    adata
        annotated data matrix
    groupby
        One or multiple columns to group by
    aggr_fun
        Callback function to calculate pseudobulk. Must be a numpy ufunc supporting
        the `axis` attribute.
    min_obs
        Exclude groups with less than `min_obs` observations

    Returns
    -------
    New anndata object with same vars as input, but reduced number of obs.
    """
    if isinstance(groupby, str):
        groupby = [groupby]

    combinations = adata.obs.loc[:, groupby].drop_duplicates()

    if adata.is_view:
        # for whatever reason, the pseudobulk function is terribly slow when operating on a view.
        adata = adata.copy()

    # precompute masks
    masks = {}
    for col in groupby:
        masks[col] = {}
        for val in combinations[col].unique():
            masks[col][val] = adata.obs[col] == val

    expr_agg = []
    obs = []

    for comb in combinations.itertuples(index=False):
        mask = reduce(and_, (masks[col][val] for col, val in zip(groupby, comb)))
        if np.sum(mask) < min_obs:
            continue
        expr_row = aggr_fun(adata.X[mask, :], axis=0)
        obs_row = comb._asdict()
        obs_row["n_obs"] = np.sum(mask)
        # convert matrix to array if required (happens when aggregating spares matrix)
        try:
            expr_row = expr_row.A1
        except AttributeError:
            pass
        obs.append(obs_row)
        expr_agg.append(expr_row)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ImplicitModificationWarning)
        return AnnData(
            X=np.vstack(expr_agg),
            var=adata.var,
            obs=pd.DataFrame.from_records(obs),
        )


# %%
"""Helper functions for cellphonedb analysis

Focuses on differential cellphonedb analysis between conditions.
"""
from typing import List, Literal
import pandas as pd
#from .pseudobulk import pseudobulk
import numpy as np
import scanpy as sc
import altair as alt
#from .compare_groups.pl import plot_lm_result_altair
#from .util import fdr_correction


class CpdbAnalysis:
    def __init__(
        self, cpdb, adata, *, pseudobulk_group_by: List[str], cell_type_column: str
    ):
        """
        Class that handles comparative cellphonedb analysis.

        Parameters
        ----------
        cpdb
            pandas data frame with cellphonedb interactions.
            Required columns: `source_genesymbols`, `target_genesymbol`.
            You can get this from omnipathdb:
            https://omnipathdb.org/interactions/?fields=sources,references&genesymbols=1&databases=CellPhoneDB
        adata
            Anndata object with the target cells. Will use this to derive mean fraction of expressed cells.
            Should contain counts in X.
        pseudobulk_group_by
            See :func:`scanpy_helper.pseudobulk.pseudobulk`. Pseudobulk is used to compute the mean fraction
            of expressed cells by patient
        cell_type_column
            Column in anndata that contains the cell-type annotation.
        """
        self.cpdb = cpdb
        self.cell_type_column = cell_type_column
        self._find_expressed_genes(adata, pseudobulk_group_by)

    def _find_expressed_genes(self, adata, pseudobulk_group_by):
        """Compute the mean expression and fraction of expressed cells per cell-type.
        This is performed on the pseudobulk level, i..e. the mean of means per patient is calculated.
        """
        pb_fracs = pseudobulk(
            adata,
            groupby=pseudobulk_group_by + [self.cell_type_column],
            aggr_fun=lambda x, axis: np.sum(x > 0, axis) / x.shape[axis],  # type: ignore
        )
        fractions_expressed = pseudobulk(
            pb_fracs, groupby=self.cell_type_column, aggr_fun=np.mean
        )
        fractions_expressed.obs.set_index(self.cell_type_column, inplace=True)

        pb = pseudobulk(
            adata,
            groupby=pseudobulk_group_by + [self.cell_type_column],
        )
        sc.pp.normalize_total(pb, target_sum=1e6)
        sc.pp.log1p(pb)
        pb_mean_cell_type = pseudobulk(
            pb, groupby=self.cell_type_column, aggr_fun=np.mean
        )
        pb_mean_cell_type.obs.set_index(self.cell_type_column, inplace=True)

        self.expressed_genes = (
            fractions_expressed.to_df()
            .melt(ignore_index=False, value_name="fraction_expressed")
            .reset_index()
            .merge(
                pb_mean_cell_type.to_df()
                .melt(ignore_index=False, value_name="expr_mean")
                .reset_index(),
                on=[self.cell_type_column, "variable"],
            )
        )

    def significant_interactions(
        self,
        de_res: pd.DataFrame,
        *,
        pvalue_col="pvalue",
        fc_col="log2FoldChange",
        gene_symbol_col="gene_id",
        max_pvalue=0.1,
        min_abs_fc=1,
        adjust_fdr=True,
        min_frac_expressed=0.1,
        de_genes_mode: Literal["ligand", "receptor"] = "ligand",
    ) -> pd.DataFrame:
        """
        Generates a data frame of differentiall cellphonedb interactions.

        This function will extract all known ligands (or receptors, respectively) from a list of differentially expressed
        and find all receptors (or ligands, respectively) that are expressed above a certain cutoff in all cell-types.

        Parameters:
        -----------
        de_res
            List of differentially expressed genes
        pvalue_col
            column in de_res that contains the pvalue or false discovery rate
        gene_id_col
            column in de_res that contains the gene symbol
        min_frac_expressed
            Minimum fraction cells that need to express the receptor (or ligand) to be considered a potential interaction
        de_genes_mode
            If the list of de genes provided are ligands (default) or receptors. In case of `ligand`, cell-types
            that express corresonding receptors above the threshold will be identified. In case of `receptor`,
            cell-types that express corresponding ligands above the threshold will be identified.
        adjust_fdr
            If True, calculate false discovery rate on the pvalue, after filtering for genes that are contained
            in the cellphonedb.
        """
        if de_genes_mode == "ligand":
            cpdb_de_col = "source_genesymbol"
            cpdb_expr_col = "target_genesymbol"
        elif de_genes_mode == "receptor":
            cpdb_de_col = "target_genesymbol"
            cpdb_expr_col = "source_genesymbol"
        else:
            raise ValueError("Invalud value for de_genes_mode!")

        de_res = de_res.loc[lambda x: x[gene_symbol_col].isin(self.cpdb[cpdb_de_col])]
        if adjust_fdr:
            de_res = fdr_correction(de_res, pvalue_col=pvalue_col, key_added="fdr")
            pvalue_col = "fdr"

        significant_genes = de_res.loc[
            lambda x: (x[pvalue_col] < max_pvalue) & (np.abs(x[fc_col]) >= min_abs_fc),
            gene_symbol_col,
        ].unique()  # type: ignore
        significant_interactions = self.cpdb.loc[
            lambda x: x[cpdb_de_col].isin(significant_genes)
        ]

        res_df = (
            self.expressed_genes.loc[
                lambda x: x["fraction_expressed"] >= min_frac_expressed
            ]  # type: ignore
            .merge(
                significant_interactions,
                left_on="variable",
                right_on=cpdb_expr_col,
            )
            .drop(columns=["variable"])
            .merge(de_res, left_on=cpdb_de_col, right_on=gene_symbol_col)
            .drop(columns=[gene_symbol_col])
        )

        return res_df

    def plot_result(
        self,
        cpdb_res,
        *,
        pvalue_col="fdr",
        group_col="group",
        fc_col="log2FoldChange",
        title="CPDB analysis",
        aggregate=True,
        clip_fc_at=(-5, 5),
        label_limit=100,
        cluster: Literal["heatmap", "dotplot"] = "dotplot",
        de_genes_mode: Literal["ligand", "receptor"] = "ligand",
    ):
        """
        Plot cpdb results as heatmap

        Parameters
        ----------
        cpdb_res
            result of `significant_interactions`. May be further filtered or modified.
        group_col
            column to be used for the y axis of the heatmap
        aggregate
            whether to merge multiple targets of the same ligand into a single column
        de_genes_mode
            If the list of de genes provided are ligands (default) or receptors. If receptor, will show the dotplot
            at the top (source are expressed ligands) and the de heatmap at the bottom (target are the DE receptors).
            Otherwise the other way round.
        """
        if de_genes_mode == "ligand":
            cpdb_de_col = "source_genesymbol"
            cpdb_expr_col = "target_genesymbol"
        elif de_genes_mode == "receptor":
            cpdb_de_col = "target_genesymbol"
            cpdb_expr_col = "source_genesymbol"
        else:
            raise ValueError("Invalud value for de_genes_mode!")

        cpdb_res[fc_col] = np.clip(cpdb_res[fc_col], *clip_fc_at)

        # aggregate if there are multiple receptors per ligand
        if aggregate:
            cpdb_res = (
                cpdb_res.groupby(
                    [
                        self.cell_type_column,
                        cpdb_de_col,
                        fc_col,
                        pvalue_col,
                        group_col,
                    ]
                )
                .agg(
                    n=(cpdb_expr_col, len),
                    fraction_expressed=("fraction_expressed", np.max),
                    expr_mean=("expr_mean", np.max),
                )
                .reset_index()
                .merge(
                    cpdb_res.groupby(cpdb_de_col).agg(
                        **{
                            cpdb_expr_col: (
                                cpdb_expr_col,
                                lambda x: "|".join(np.unique(x)),
                            )
                        }
                    ),
                    on=cpdb_de_col,
                )
            )

        cpdb_res["interaction"] = [
            f"{s}_{t}" for s, t in zip(cpdb_res[cpdb_de_col], cpdb_res[cpdb_expr_col])
        ]

        # cluster heatmap
        if cluster is not None:
            from scipy.cluster.hierarchy import linkage, leaves_list

            _idx = self.cell_type_column if cluster == "dotplot" else group_col
            _values = "fraction_expressed" if cluster == "dotplot" else fc_col
            _columns = "interaction"
            values_df = (
                cpdb_res.loc[:, [_idx, _values, _columns]]
                .drop_duplicates()
                .pivot(
                    index=_idx,
                    columns=_columns,
                    values=_values,
                )
                .fillna(0)
            )
            order = values_df.columns.values[
                leaves_list(
                    linkage(values_df.values.T, method="average", metric="euclidean")
                )
            ]
        else:
            order = "ascending"

        p1 = plot_lm_result_altair(
            cpdb_res,
            color=fc_col,
            p_col=pvalue_col,
            x="interaction",
            configure=lambda x: x,
            title="",
            order=order,
            p_cutoff=1,
        ).encode(
            x=alt.X(
                title=None,
                axis=alt.Axis(
                    labelExpr="split(datum.label, '_')[0]",
                    orient="top" if de_genes_mode == "receptor" else "bottom",
                ),
            )
        )

        p2 = (
            alt.Chart(cpdb_res)
            .mark_circle()
            .encode(
                x=alt.X(
                    "interaction",
                    axis=alt.Axis(
                        grid=True,
                        orient="bottom" if de_genes_mode == "receptor" else "top",
                        title=None,
                        labelExpr="split(datum.label, '_')[1]",
                        labelLimit=label_limit,
                    ),
                    sort=order,
                ),
                y=alt.Y(self.cell_type_column, axis=alt.Axis(grid=True), title=None),
                size=alt.Size("fraction_expressed"),
                color=alt.Color("expr_mean", scale=alt.Scale(scheme="cividis")),
            )
        )

        if de_genes_mode == "receptor":
            p1, p2 = p2, p1

        return (
            alt.vconcat(p1, p2, title=title)
            .resolve_scale(size="independent", color="independent", x="independent")
            .configure_mark(opacity=1)
            .configure_concat(spacing=label_limit - 130)
        )


# %%
import scanpy as sc

# %%
adata = sc.read_h5ad("/data/projects/2022/CRCA/results/v0.1/crc-atlas-dataset/latest/ds_analyses/liana_cell2cell/core_atlas/adata_rank_agregate.h5ad")

# %%
adata.obs.columns

# %%
import pandas as pd

# %%
df = pd.read_csv("/data/projects/2022/CRCA/results/v0.1/crc-atlas-dataset/latest/ds_analyses/liana_cell2cell/core_atlas/tumor_normal/03_deseq2/tumor_normal_matched_B_cell_tumor_vs_normal_DESeq2_result.tsv", sep="\t")

# %%
list(set(adata.obs.cell_type))

# %%
modified_list = [item.replace(" ", "_") for item in list(set(adata.obs.cell_type))]

# %%
modified_list

# %%
import os
import pandas as pd


# Folder containing the files
folder_path = "/data/projects/2022/CRCA/results/v0.1/crc-atlas-dataset/latest/ds_analyses/liana_cell2cell/core_atlas/tumor_normal/03_deseq2/"

# Initialize an empty list to hold all DataFrames
df_list = []

# Loop through each element in the modified_list
for element in modified_list:
    # Find the file that starts with the element name
    for file_name in os.listdir(folder_path):
        if file_name.startswith(element):
            # Read the CSV file into a DataFrame
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path,sep="\t")

            # Add a new column "cell_name" with the value of the element
            df["cell_name"] = element

            # Append the DataFrame to the list
            df_list.append(df)
            break  # Break after finding the matching file

# Concatenate all DataFrames into one
final_df = pd.concat(df_list, ignore_index=True)

# Now `final_df` contains the concatenated DataFrame with the "cell_name" column added
print(final_df.head())


# %%
final_df

# %%
de_res = final_df

# %%
#result of `significant_interactions`. May be further filtered or modified.
cpdb_res = adata.uns['rank_aggregate'].loc[
        lambda x: x["specificity_rank"] <= 0.01
    ]

# %%
# rename columns in liana results 
cpdb_res=cpdb_res.rename(columns={"ligand_complex":"source_genesymbol","receptor_complex":"target_genesymbol"})

# %%
cpdb_res.columns

# %%
# use scanpy helper class CpdbAnalysis to compute pseudobulk, cell fraction and 
cpdba = CpdbAnalysis(
    cpdb_res,
    adata,
    pseudobulk_group_by=["patient_id"],
    cell_type_column="cell_type"
)

# %%
cpdba

# %%
cpdb_sig_int = cpdba.significant_interactions(
    de_res, max_pvalue=0.1
)

# %%
## This is input for CIRCOS PLOT 
cpdb_sig_int.to_csv("epithelial_cancer_old_atlas.csv")

# %%
