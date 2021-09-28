
# 1) INSTALL THE PACKAGES

#install.packages("RKEEl")
#install.packages("RKEELdata")
#install.packages("RKEELjars")
#install.packages("psych")
#install.packages("Hmisc")
#install.packages("skimr")
#install.packages("visdat")
#install.packages("naniar")
#install.packages("caret") --> Classification And REgression Training (CARET) --> https://topepo.github.io/caret/
#install.packages("dplyr")
#install.packages("recipes")



# 2) IMPORT THE DATASET

RKEELdata::getKeelDatasetList() # the Dataset list
RKEELdata::getDataPath() # get the data path 

# load the dataset
hepatitis = RKEEL::read.keel( "/home/msi/R/x86_64-pc-linux-gnu-library/4.1/RKEELdata/datasets/hepatitis.dat")




# 3) Some changes and EDA (Exploratory Data Analysis)

# more information about the dataset --->  https://archive.ics.uci.edu/ml/datasets/hepatitis

# numeric 
hepatitis = data.frame(apply(hepatitis, MARGIN = 2, FUN=as.numeric))
# class is always a factor
hepatitis$Class = as.factor(hepatitis$Class)

# factor
for (i in 2:13){
  hepatitis[,  i] = as.factor(hepatitis[, i])
}
hepatitis$Histology = as.factor(hepatitis$Histology)

# descriptive measures 
summary(hepatitis)
str(hepatitis)
psych::describe(hepatitis)
Hmisc::describe(hepatitis) 
skimr::skim(hepatitis)

# plots for numeric features
library(caret)
featurePlot(x=hepatitis[, 14:18], y=hepatitis$Class, plot="box", 
            scales = list(x=list(relation="free"), y = list(relation="free")))
featurePlot(x=hepatitis[, 14:18], y=hepatitis$Class, plot="density", 
            scales = list(x=list(relation="free"), y = list(relation="free")))


# Missing values 
visdat::vis_dat(hepatitis)
naniar::vis_miss(hepatitis)
naniar::gg_miss_var(hepatitis)
naniar::miss_case_summary(hepatitis)



# Class immbalance 
imbalance::imbalanceRatio(hepatitis, classAttr = "Class")
table(hepatitis$Class)
32/123
123/123


# 4) Preprocessing 

# Protime and AlkPhosphate have too many missing values, but first is usefull 

hepatitis2 = dplyr::select(hepatitis, -c("AlkPhosphate"))

# Imputation of data (knn for numeric features and mode for binary features)
library(recipes)
hepatitis2 = recipe(Class ~ ., data=hepatitis2) %>% 
  step_impute_knn(all_numeric()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  prep() %>% bake(new_data=NULL)

# Recode the levels of class
hepatitis2$Class = factor(hepatitis2$Class, levels=c(1,2), labels=c("Die", "Live"))

# Plots to check the missing values
naniar::vis_miss(hepatitis2)


# 5) Modelling the data 

#   5.1)Choose the models

names(getModelInfo()) # --> available models using caret 

# I HAVE CHOSEN FOR THIS MINI-TUTORIAL: 

# Decision tree (DT): J48, PART, rpart

# Rules based models (RL): JRip

# Bagging (BAG) : treebag 

# Random Forest (RF): parRF, rf

# Boosting (BOOST): C5.0


# If I want information about their parameters:
modelLookup("J48") 

#   5.2) Split the dataset into train1 and test 
set.seed(12345)
resamples_train = createDataPartition(hepatitis2$Class, times=1, p=0.75, list=FALSE)

train = hepatitis2[c(resamples_train), ]
test = hepatitis2[-c(resamples_train), ]

# estratificated partitions
prop.table(table(train$Class))
prop.table(table(test$Class))

#  5.3) Creating models and tune the parameters

# We will use 5 fold cross validation because if we will use k = 10 we 
# would have 2 positive examples in each fold

table(train$Class)
24/10
24/5

# The metric which we will use to tune the parameters, would be AUC score

# control1 
set.seed(12345)
control = trainControl(method="cv", number=5,
                       savePrediction=TRUE, 
                       classProbs = TRUE,  
                       summaryFunction = twoClassSummary)
# control2 UNDERSAMPLING 
set.seed(12345)
control = trainControl(method="cv", number=5,
                       savePrediction=TRUE, 
                       classProbs = TRUE,  
                       summaryFunction = twoClassSummary, sampling="down")
## DT 
### J48
set.seed(12345)
j48 = train(Class ~ . , data=train, method="J48", metric="ROC",
            trControl= control)

bt_j48 = as.numeric(rownames(j48$bestTune))
result_j48 = j48$results[bt_j48, ]

### PART
set.seed(12345)
PART = train(Class ~ . , data=train, method="PART", metric="ROC",
            trControl= control)

bt_PART = as.numeric(rownames(PART$bestTune))
result_PART = PART$results[bt_PART, ]

### rpart

set.seed(12345)
rpart = train(Class ~ . , data=train, method="rpart", metric="ROC",
             trControl= control)

bt_rpart = as.numeric(rownames(rpart$bestTune))
result_rpart = rpart$results[bt_rpart, ]


## RL

### JRip

set.seed(12345)
jrip = train(Class ~ . , data=train, method="JRip", metric="ROC",
              trControl= control)

bt_jrip = as.numeric(rownames(jrip$bestTune))
result_jrip = jrip$results[bt_jrip, ]


## BAG 

### treebag 

set.seed(12345)
treebag = train(Class ~ . , data=train, method="treebag", metric="ROC",
             trControl= control)

bt_treebag = as.numeric(rownames(treebag$bestTune))
result_treebag = treebag$results[bt_treebag, ]

### rf 

set.seed(12345)
rf = train(Class ~ . , data=train, method="rf", metric="ROC",
                trControl= control)

bt_rf = as.numeric(rownames(rf$bestTune))
result_rf = rf$results[bt_rf, ]

### parRF

set.seed(12345)
parRF = train(Class ~ . , data=train, method="parRF", metric="ROC",
           trControl= control)

bt_parRF = as.numeric(rownames(parRF$bestTune))
result_parRF = rf$results[bt_parRF, ]


## BOOST
set.seed(12345)
C50 = train(Class ~ . , data=train, method="C5.0", metric="ROC",
              trControl= control)

bt_C50 = as.numeric(rownames(C50$bestTune))
result_C50 = C50$results[bt_C50, ]

# 6) Comparing the performance between different models 

comparaison = resamples(list(J48 = j48, PART = PART, rpart= rpart, JRip=jrip, 
                             treebag = treebag, rf= rf, parRF=parRF, C50 =C50))

summary(comparaison)

scales = list(x = list(relation="free"), y=list(relation="free"))
bwplot(comparaison, scales=scales)
# summary(diff(comparaison))

# 7) Evaluating the models with test data 
set.seed(12345)
control = trainControl(method="none",
                       savePrediction=TRUE, 
                       classProbs = TRUE,  
                       summaryFunction = twoClassSummary, sampling="down")

rf$bestTune
parRF$bestTune
j48$bestTune

rfs_param = expand.grid(mtry=2)
j48_param = expand.grid(C=0.255, M=3)

## Using the best parameters to create one model from all training1 data 
set.seed(12345)
rf_def = train(Class ~ . , data=train, method="rf", metric="ROC",
               trControl= control, tuneGrid=rfs_param)
set.seed(12345)
parRF_def = train(Class ~ . , data=train, method="parRF", metric="ROC",
               trControl= control, tuneGrid=rfs_param)
set.seed(12345)
j48_def = train(Class ~ . , data=train, method="J48", metric="ROC",
                trControl= control, tuneGrid=j48_param)

## Prediction with test data 

pred_rf = predict(rf_def, test)
pred_parRF = predict(parRF_def, test)
pred_J48 = predict(j48_def, test)

## Confussion matrix 


confusionMatrix(pred_rf, test$Class)
confusionMatrix(pred_parRF, test$Class)
confusionMatrix(pred_J48, test$Class)



# 8) Other methods which are similar to k fold cross validation 

# Leave One Out Cross Validation

set.seed(12345)
control_loo = trainControl(method="LOOCV", 
                       savePrediction=TRUE, 
                       classProbs = TRUE,  
                       summaryFunction = twoClassSummary, sampling="down")

# Leave Group Out Cross Validation / Monte Carlo Cross Validation

set.seed(12345)
control_lgo = trainControl(method="LGOCV", p=0.8,
                       savePrediction=TRUE, 
                       classProbs = TRUE,  
                       summaryFunction = twoClassSummary, sampling="down")

# Bootstrap

set.seed(12345)
control_boot = trainControl(method="boot", p=0.8,
                       savePrediction=TRUE, 
                       classProbs = TRUE,  
                       summaryFunction = twoClassSummary, sampling="down")

# J48 with LOO

set.seed(12345)
j48_loo = train(Class ~ . , data=train, method="J48", metric="ROC",
               trControl= control_loo)

# J48 with LGO
set.seed(12345)
j48_lgo = train(Class ~ . , data=train, method="J48", metric="ROC",
               trControl= control_lgo)

# j48 with Bootstrap
set.seed(12345)
j48_boot = train(Class ~ . , data=train, method="J48", metric="ROC",
               trControl= control_boot)

# best parameters in each methodology
j48$bestTune
j48_loo$bestTune
j48_lgo$bestTune
j48_boot$bestTune

# and the different results
j48$results
j48_loo$results
j48_lgo$results
j48_boot$results


# 9) R version and loaded packages 
sessionInfo()




