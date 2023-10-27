library(curatedMetagenomicData)
library(curatedMetagenomicAnalyses)
library(dplyr)


load("data/CRC.rda")
tse = se
rm(se)

otu_data <- assays(tse)$relative_abundance 

sample_data <- colData(tse)

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
# Extract sample metadata
sample_data <- colData(tse)

sample_id = which(sample_data$disease %in% c('healthy', 'CRC'))

normalized_otu_data = normalized_otu_data[sample_id,]
dim(normalized_otu_data)

response <- sample_data$disease 
response = response[sample_id]


length(response)

length(which(response %in% c('healthy')))

# 2. Model Fitting

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(response, p=0.8, list=FALSE)
otu_train <- normalized_otu_data[trainIndex,]
response_train <- response[trainIndex]

otu_test <- normalized_otu_data[-trainIndex, ]
response_test <- response[-trainIndex]

# Fit logistic regression model


# Check dimensions again
dim(otu_train) # Should now be 1009 x 900 or 1009*300 if use PCA

# Create the data frame for the model
training_data <- data.frame(response_train, otu_train)
response_train_binary <- ifelse(response_train == "CRC", 1, 0)
training_data$response_train <- response_train_binary
model <- glm(response_train ~ ., data = training_data, family = "binomial")

# 3. Evaluation

# Predict on the test set
dim(otu_test)
predictions_prob <- predict(model, newdata = data.frame(otu_test), type = "response")
predictions <- ifelse(predictions_prob > 0.5, "CRC", "healthy")

# Confusion matrix

cmtx = confusionMatrix(as.factor(predictions), as.factor(response_test))
print(cmtx)

#### single PCA: 0.5806 


##################### ##################### ##################### ##################### ##################### ##################### ##################### 
##################### ##################### ##################### ##################### ##################### ##################### ##################### 
##################### start from combine end, extract PCA
rm(list = ls())
load("normalized_otu_data_IBD_CRC.RData")

dim(normalized_otu_data)
normalized_otu_data = normalized_otu_data[1261:2655,]
dim(normalized_otu_data)


load("data/CRC.rda")
tse = se
rm(se)
# Extract sample metadata
sample_data <- colData(tse)

sample_id = which(sample_data$disease %in% c('healthy', 'CRC'))

normalized_otu_data = normalized_otu_data[sample_id,]
dim(normalized_otu_data)

response <- sample_data$disease 
response = response[sample_id]

# 2. Model Fitting

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(response, p=0.8, list=FALSE)
otu_train <- normalized_otu_data[trainIndex,]
response_train <- response[trainIndex]

otu_test <- normalized_otu_data[-trainIndex, ]
response_test <- response[-trainIndex]

# Fit logistic regression model


# Check dimensions again
dim(otu_train) # Should now be 1009 x 900 or 1009*300 if use PCA

# Create the data frame for the model
training_data <- data.frame(response_train, otu_train)
response_train_binary <- ifelse(response_train == "CRC", 1, 0)
training_data$response_train <- response_train_binary
model <- glm(response_train ~ ., data = training_data, family = "binomial")

# 3. Evaluation

# Predict on the test set
dim(otu_test)
predictions_prob <- predict(model, newdata = data.frame(otu_test), type = "response")
predictions <- ifelse(predictions_prob > 0.5, "CRC", "healthy")

# Confusion matrix
response_test <- ifelse(response_test == "CRC", "CRC", "healthy")

cmtx = confusionMatrix(as.factor(predictions), as.factor(response_test))

print(cmtx)
#### 0.629



