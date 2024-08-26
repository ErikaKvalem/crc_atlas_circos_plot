#!/usr/bin/env Rscript


# load required packages
suppressPackageStartupMessages({
library(BiocParallel)
library(conflicted)
library(readr)
library(tibble)
library(dplyr)
library(stringr)
library(forcats)
library(DESeq2)
library(IHW)
library(limma)
})


input_path="/data/projects/2022/CRCA/results/v1/final/liana_cell2cell/core_atlas/tumor_normal/02_pseudobulk/"
resDir = "/data/projects/2022/CRCA/results/v1/final/liana_cell2cell/core_atlas/tumor_normal/03_deseq2/"
count_mat <- read_csv(paste0(input_path,"epithelial_cancer_count_mat.csv"))
colData <- read_csv(paste0(input_path,"epithelial_cancer_colData.csv"))
rowData <- read_csv(paste0(input_path,"epithelial_cancer_rowData.csv"))


# Load parameters
prefix <- "tumor_vs_normal"

sample_col <- "sample_col"
cond_col <- "sample_type"
covariate_formula <- "patient_id +"

sum2zero = FALSE

if(!sum2zero) {
  c1 <- "tumor"
  c2 <- "normal"
} else {
  c1 = NULL
  c2 = NULL
}

n_cpus <- 4


design_formula <- as.formula(paste0("~", covariate_formula, " ", cond_col))

register(MulticoreParam(workers = n_cpus))

countData = count_mat |> column_to_rownames(var = "gene_id") |> ceiling()
colData = colData |> column_to_rownames(var = sample_col)



if (length(unique(colData[[cond_col]])) < 2 ) {
  print(paste0("Categories in cond col",length(unique(colData[[cond_col]])) ))
  quit()
}

if(sum2zero) {
  design_mat = model.matrix(design_formula, contrasts.arg = structure(as.list("contr.sum"), names=cond_col), data=colData)  
} else {
  design_mat = design_formula
}

dds <- DESeqDataSetFromMatrix(
  countData = countData,
  colData = colData,
  rowData = rowData,
  design = design_formula
)
# define reference level (not really necessary when uisng contrasts)
if(!sum2zero) {
dds[[cond_col]] <- relevel(dds[[cond_col]], ref = c2)
} 

## keep only genes where we have >= 10 reads per samplecondition in at least 2 samples
dds <- estimateSizeFactors(dds)
keep <- rowSums(counts(dds, normalized = TRUE) >= 10) >= 2
dds <- dds[keep, ]

# save normalized filtered count file
norm_mat <- counts(dds, normalized = TRUE) |> as_tibble(rownames = "gene_id")
write_tsv(norm_mat,  paste0(resDir,prefix, "_NormalizedCounts.tsv"))

# save normalized batch corrected filtered count file
vst <- vst(dds, nsub=sum( rowMeans( counts(dds, normalized=TRUE)) > 5 ), blind = FALSE) # to avoid this error " less than 'nsub' rows with mean normalized count > 5, " 


if(!sum2zero) {
batch <- gsub("\\+", "", covariate_formula) |> str_squish()
assay(vst) <- limma::removeBatchEffect(x = assay(vst), batch = vst[[batch]])
write_tsv(assay(vst) |> as_tibble(rownames = "gene_id"), paste0(resDir,prefix, "_vst_batch_corrected_NormalizedCounts.tsv"))
}
# run DESeq
dds <- DESeq(dds, parallel = (n_cpus > 1))

if(sum2zero) {
  # order needs to be the one of the levels of the factor (same as for contrast matrix)
  unique_conditions = levels(as.factor(colData[[cond_col]]))
  n_unique = length(unique_conditions)
  # with sum2zero we test that a single coefficient != 0
  # a coefficient corresponds to the difference from the overall mean
  # the intercept correponds to the overall mean
  contr_mat = diag(n_unique - 1) 
  # make list with one contrast per item
  contrasts = lapply(seq_len(n_unique - 1), function(i) { contr_mat[, i] }) 
  # the above added n-1 comparisons, we need to construct the last (redundant) one manually
  contrasts = append(contrasts, list(-apply(contr_mat, MARGIN = 1, sum) / (n_unique - 1)))
  # pad end of vector with zeros (required if there are covariates in the design).
  # we can assume that the "condition columns" always come at the front since
  # it is the first argument of the formula
  contrasts = lapply(contrasts, function(x) {
    c(0, x, rep.int(0, length(resultsNames(dds)) - n_unique))
  })
  # set names of contrasts
  names(contrasts) = unique_conditions
} else {
  contrasts = list(c(cond_col, c1, c2))
  names(contrasts) = sprintf("%s_vs_%s", c1, c2)
}



resIHW <- lapply(names(contrasts), function(name) {
  contrast <- contrasts[[name]]
  results(dds, filterFun = ihw, contrast = contrast) |>
    as_tibble(rownames = "gene_id") |>
    mutate(comparison = name) |>
    arrange(pvalue)
}) |> bind_rows()


if(!sum2zero) {
write_tsv(resIHW, paste0(resDir,prefix, "_", names(contrasts), "_DESeq2_result.tsv"))
}else if (sum2zero) {
  write_tsv(resIHW, paste0(resDir,prefix, "_DESeq2_result.tsv"))
}
