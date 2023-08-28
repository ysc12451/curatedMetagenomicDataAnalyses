#### IBD
load("data/IBD.rda")
tse_IBD = se

otu_data_IBD <- assays(tse_IBD)$relative_abundance 
dim(otu_data_IBD)

#### CRC
load("data/CRC.rda")
tse_CRC = se
rm(se)
otu_data_CRC <- assays(tse_CRC)$relative_abundance 
dim(otu_data_CRC)

inter = intersect(rownames(otu_data_IBD), rownames(otu_data_CRC))
length(inter)

dim(otu_data_IBD[inter,])  # 782, 1260 OTUs, sample
dim(otu_data_CRC[inter,])  # 782, 1395 OTUs, sample

otu_data = cbind(otu_data_IBD[inter,], otu_data_CRC[inter,])
dim(otu_data)
### preprocess
library(edgeR)
library(caret)

group <- factor(c(rep(1,ncol(otu_data)))) # if you have a grouping variable, replace the appropriate values
y <- DGEList(counts=otu_data, group=group)
y <- calcNormFactors(y, method="TMM")
normalized_otu_data <- cpm(y)

normalized_otu_data = t(normalized_otu_data)
dim(normalized_otu_data)

# PCA
# 1. Identify and remove constant columns
constant_cols <- which(apply(normalized_otu_data, 2, var) == 0)
filtered_otu_data <- normalized_otu_data[,-constant_cols]
dim(filtered_otu_data)
# 2. Apply PCA on the modified data
set.seed(123)  # for reproducibility
pca_res <- prcomp(filtered_otu_data, center = TRUE, scale. = TRUE)

top_300_PCs <- pca_res$x[, 1:300]

dim(top_300_PCs)

normalized_otu_data = top_300_PCs

class(normalized_otu_data)

save(normalized_otu_data, file = "normalized_otu_data_IBD_CRC.RData")
