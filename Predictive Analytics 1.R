#Use former PRNG
RNGkind(sample.kind = "Rounding")

#Load libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(caret)

#Read data
df <- read.csv("June 17 data.csv")
summary(df)

#EXPLORE DATA

#Create histogram for age
ggplot(df, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "royalblue", col = "royalblue")

#Bulk of ages are between 25 and 60. Distribution is right skewed somewhat right. Consider log transformation to normalize data


# Create bar chart for job.
ggplot(df, aes(x = job)) +
  geom_bar(stat = "count", fill = "royalblue", col = "royalblue") +
  theme(axis.text = element_text(size = 6))

#Highest proportion of customers have an administrative job, followed by blue collar and technician jobs. 
#Those missing job values are relatively small but impact on purchase rate should be considered. Many categories may lead to high variance
#Some job values may need to be combined for more reliable predictions based on job

# Create a bar chart for edu_years.
ggplot(df, aes(x = edu_years)) +
  geom_bar(stat = "count", fill = "royalblue", col = "royalblue") +
  theme(axis.text = element_text(size = 6))

#Edu_years takes integer values from 1 to 16 with gaps between values. There are a few outliers that have 1 year of education which may need to be removed

# Create a boxplot of the age distribution for different jobs.
boxplot(age ~ job, data = df, ylab = "Age Distribution", cex.axis = 0.5)

#Retired and student groups are immediately different from other groups of age by job, the former being higher age and the latter being much younger
#Age and job have some degree of codependence. Be careful when dealing with these two variables together

#Create a graph showing purchasing price by age
ggplot(df) +
  aes(x = age, fill = factor(purchase)) +
  labs(y = "Proportion of Purchases") +
  ggtitle("Proportion of Purchases by age") +
  geom_bar(position = "fill")

#Proportion of purchase trends downward frmo ages 17 to 50 and then drops off with a significant jump at age 60
# GLM models will have trouble fitting this with single age variable. Consider age squared as additional variable
# Decision trees will not need additional variables for these shapes

# Create a graph showing the proportion purchasing by edu_years.
ggplot(df) +
  aes(x = edu_years, fill = factor(purchase)) +
  labs(y = "Proportion of Purchases") +
  ggtitle("Proportion of Purchases by edu_years") +
  geom_bar(position = "fill")

#For proportion of purchases, there is a bowed shape. A linear model may miss this because of higher frequencies with individuals with more education
#Consider using GLM model or using tree based model that can handle non-linear patterns as well

#Dimension reduction is necessary to avoid overfitting. The original education variable has 6 variables in addition to baseline
#Creating new numeric variable edu_years has one dimension, reducing risk of overfitting
#This is optimal for GLM models because they can better discern lienar trends when converted to numerical variables
#For decision trees, conversion to numerical does not reduce dimensionality as much because the variables can be split

# Check missing values. Display missing proportions for each variable that has them.
missing_proportion <- colMeans(is.na(df))
missing_data <- data.frame(colnames = colnames(df), missing_proportion = missing_proportion)
missing_data %>%
  filter(missing_proportion > 0) %>%
  ggplot(aes(x = colnames, y = missing_proportion, label = missing_proportion)) +
  geom_bar(stat = "identity", fill = "royalblue", col = "royalblue")

# The code below calculates the proportion of purchases for NAs and for nonNAs for each variable that has NAs.
print("Purchase Proportions by variable, for missing and non missing values")
print(sprintf("%10s %15s %15s", "Variable", "PP_for_NAs", "PP_for_non_NAs"))
varnames <- c("housing", "job", "loan", "marital", "edu_years")
for (t in varnames)
{
  ind <- is.na(df[t])
  print(sprintf("%10s %15.2f %15.2f", t, mean(df["purchase"][ind]), mean(df["purchase"][!ind])))
}

#Remove rows witih NA values
df <- df[!is.na(df$marital), ]
df <- df[!is.na(df$job), ]

# Convert missing values to "unknown" (only works for factor variables)
# Create new level for housing variable.
levels(df$housing)[length(levels(df$housing)) + 1] <- "unknown"
# Use new level to convert NA to unknown.
df$housing[is.na(df$housing)] <- "unknown"


# Create new level for loan variable.
levels(df$loan)[length(levels(df$loan)) + 1] <- "unknown"
# Use new level to convert NA to unknown.
df$loan[is.na(df$loan)] <- "unknown"

# Impute using the mean (works for numeric variable edu_years)
df$edu_years[is.na(df$edu_years)] <- mean(df$edu_years, na.rm = TRUE)

summary(df)

#Edu_years: impute using mean
#Almost 5% of values are missing and the purchase proportion is significantly higher for missing values
#Since the variable is numeric, converting to unknown doesn't work and so imputing missing value as mean is all that's left

#Housing is converted to unknown despite not much difference in purchasing proportions in missing/non-missing values

#Job: remove rows since only 1% of data is missing and purchase proportion is nearly the same for missing and non-missing values

#Loans: convert to unknown; non-missing and missing values are noticeably different and therefore may be predictive

#Marital: remove rows. Less than 0.5% are missing and the purchase and purchase proportion is similar to non-missing values



#Investigate Correlations

tmp <- dplyr::select(df, age, edu_years, CPI, CCI, irate, employment)
cor(tmp, use = "complete.obs")

#The most notable correlations are between irate and employment variables and between CPI and irate
#These correlations and others are not so concerning for decision trees. For example, irate and employment are
#heavily correlated and so little to no information can be gained from splitting on employment after splitting on irate

#For GLM models, these correlations are more concerning since they do not handle collinear variables well
#Large and offseting coefficients may make interpretation of coefficients difficult 
#Consider using PCA to handle the correlated variables; alternatively delete redundant variables

#Principal Component Analysis

tmp <- dplyr::select(df, CPI, CCI, irate, employment)
apply(tmp, 2, mean)
apply(tmp, 2, sd)
pca <- prcomp(tmp, scale = TRUE)
pca


#Will be used to obtain indepdendent and uncorrelated variables. PCA was performed on CPI, CCi, irate, employment. 
#Scale parameter should be set to true so variables can be scaled to have unit variance. 
#This is important to get an accurate picture of which variables are contributing most to variance

# 	Create a bi-plot.
biplot(pca, cex = 0.8, xlabs = rep(".", nrow(tmp)))

#In the bi-plot, employment and irate have nearly identical positions, showing that PC1 and PC2 don't show much difference between them
# In PC1, similar movements in these two variables and CPI are grouped together with little emphasis on CCI, while PC2 highlights movements
#in CCI, combined with some opposing movement in CPI. The variation from PC2 is visible in the black PC
#scores, with a wide variation of PC2 scores for PC1 scores between 0.01 and 0.02

#Calculate variance explained by PCA
vars_pca <- apply(pca$x, 2, var)
vars_pca / sum(vars_pca)


#83% of the variance is explained by PC1 and PC2 and so the GLM should use two principal components

pred <- as.data.frame(predict(pca, newdata = df[, c("CPI", "CCI", "irate", "employment")]))
df <- cbind(df, pred[, 1:2])

#Split data into training and test sets. Check for set that looks reasonable

set.seed(1875)
train_ind <- createDataPartition(df$purchase, p = 0.7, list = FALSE)
data_train <- df[train_ind, ]
data_test <- df[-train_ind, ]
rm(train_ind)

print("Mean value of purchase on train and test data splits")
mean(data_train$purchase)
mean(data_test$purchase)

#Create GLM with just age

glm <- glm(purchase ~ age,
           data = data_train,
           family = binomial(link = "logit")
)

summary(glm)

library(pROC)

glm_probs <- predict(glm, data_train, type = "response")
glm_probs_test <- predict(glm, data_test, type = "response")

# Evaluate GLM: construct ROC and calculate AUC
roc <- roc(data_train$purchase, glm_probs)
par(pty = "s")
plot(roc)
pROC::auc(roc)

roc <- roc(data_test$purchase, glm_probs_test)
plot(roc)
pROC::auc(roc)

#Age in insolation has a low p-value, indicating that is statistically significant
#The ROC curve for the test data is 0.5099, which is barely any better than an intercept only model (0.5)

#Create GLM with full set of independent variables

glm <- glm(purchase ~ age + job + marital + edu_years + housing + loan + phone + month + weekday + PC1 + PC2,
           data = data_train,
           family = binomial(link = "logit")
)

summary(glm)

# Evaluate GLM: construct ROC and calculate AUC

glm_probs <- predict(glm, data_train, type = "response")
glm_probs_test <- predict(glm, data_test, type = "response")

roc <- roc(data_train$purchase, glm_probs)
par(pty = "s")
plot(roc)
pROC::auc(roc)

roc <- roc(data_test$purchase, glm_probs_test)
plot(roc)
pROC::auc(roc)

#Model is performing much better with AUC of 0.7665, showing that the model is more effective with extra predictors
#Age is now not statistically significant as it was in the previous regression

#Step selection can construct model using either forward or backward selection to quickly find an adequate set of predictors


#Forward selection starts with one variable and adds more until there is no improvement in selected criterion
#Forward selection usually results in a simpler model

library(MASS)

glm_none <- glm(purchase ~ 1,
                data = data_train,
                family = binomial(link = "logit")
)

#AIC and forward selection
stepAIC(glm_none,
        direction = "forward",
        k=2,
        scope = list(upper = glm, lower = glm_none)
)

#BIC and forward selection
stepAIC(glm_none,
        direction = "forward",
        k=log(nrow(data_train)),
        scope = list(upper = glm, lower = glm_none)
)

#AIC and backward selection
stepAIC(glm,
        direction = "backward",
        k=2,
        scope = list(upper = glm, lower = glm_none)
)

#BIC and backward selection
stepAIC(glm,
        direction = "backward",
        k=log(nrow(data_train)),
        scope = list(upper = glm, lower = glm_none)
)

#When fitting models by maximum likelihood, additional variables never decrease the loglikelihood
#For AIC, adding a variable requires an increase in the loglikelihood of 2 times number of parameters added
#for BIC, the required per parameter increase is the logarithm of the number of observations

#Using BIC results in only two variables and so AIC is used to provide more useful results
#Forward and backward selection arrive at the same model and so either method can be employed

#Construct GLM with recommended variables

# To evaluate the selected model, change the variable list to those selected by stepAIC.
glm.reduced <- glm(purchase ~ PC1 + month + weekday + job,
                   data = data_train,
                   family = binomial(link = "logit")
)

summary(glm.reduced)

# Evaluate GLM: construct ROC and calculate AUC

glm_probs <- predict(glm.reduced, data_train, type = "response")
glm_probs_test <- predict(glm.reduced, data_test, type = "response")

roc <- roc(data_train$purchase, glm_probs)
par(pty = "s")
plot(roc)
pROC::auc(roc)

roc <- roc(data_test$purchase, glm_probs_test)
plot(roc)
pROC::auc(roc)

#Final model has AUCs of 0.7780 and 0.7769 on the training and test model respectively

#This is a fairly accurate model which does not overfit because it's between 0.5 and 1, but closer to 1
#0.5 would be no better than a random and 1.0 would be a correct prediction every time
#Logit function is the natural log of odds (log(p)/ (1-p)). Correct way of expressing coefficients is to exponentiate them

#For example, PC1 is 0.644 and so a 1 unit change in PCA would increase odds of purchase by exp(0.644) = 190%

#Elastic Net
#Adds to the loglikelihood a penalty based on the magnitude of the estimated coefficients
#Penalty includes both a term based on the sum of squares of the coefficients (Ridge regression)
#And a term based on the sum of the absolute value of the coefficients (LASSO Regression)

#An alpha hyperparameter controls how much of each type of term is included
#A lambda hyperparameter controls the size of the overall penalty

#Penalty induces shrinkage in the estimated coefficients when they are being optimized and an inclusion of an absolute value term
#allows this shrinkage to go all the way to 0

library(glmnet)
set.seed(42)

# Use full set of variables for training and test sets
X.train <- model.matrix(purchase ~ age + job + marital + edu_years + housing + loan + phone + month + weekday + PC1 + PC2,
                        data = data_train
)
X.test <- model.matrix(purchase ~ age + job + marital + edu_years + housing + loan + phone + month + weekday + PC1 + PC2,
                       data = data_test
)

m <- cv.glmnet(
  x = X.train,
  y = data_train$purchase,
  family = "binomial",
  type.measure = "class",
  alpha = 0.5
) # alpha = 1 implies LASSO, alpha = 0 implies ridge, values between 0 and 1 imply elastic net
plot(m)


#Use the cross-validation results to run the final elastic net regression model.


m.final <- glmnet(
  x = X.train,
  y = data_train$purchase,
  family = "binomial",
  lambda = m$lambda.min,
  alpha = 0.5
)

# List variables
m.final$beta

# Evaluate against train and test sets

# Predict on training data
enet.pred.train <- predict(m.final, X.train, type = "response")

roc <- roc(as.numeric(data_train$purchase), enet.pred.train[, 1])
par(pty = "s")
plot(roc)
pROC::auc(roc)

# Predict on test data
enet.pred.test <- predict(m.final, X.test, type = "response")

roc <- roc(as.numeric(data_test$purchase), enet.pred.test[, 1])
par(pty = "s")
plot(roc)
pROC::auc(roc)

#AUC is 0.7742 and 0.7659 on the training and test sets respectively 

#Elastic net model includes PC1, Month, Weekday, Job, Marital Status, and Phone
#Binarization removes coefficients within weekday and job and new variables Marital and Phone are introduced

#Construct a decision tree

# Load the two needed libraries
library(rpart)
library(rpart.plot)

set.seed(1234)

#Using underlying variables instead of the principal components derived from them is helpful here
#because it is harder to interpret those PCA variables and decision trees are not adversely affected by the
#presence of highly correlated variables. However, there is no or little information can be gained from
#splitting on employment after having split on irate, given that “irate” and “employment” are very highly
#correlated, and therefore, I felt dropping the empnloyment variable was the best choice

formula <- "purchase ~ age + job + marital + edu_years + housing + loan + phone + month + weekday + CPI + CCI + irate"

tree1 <- rpart(formula,
               data = data_train, method = "class",
               control = rpart.control(minbucket = 5, cp = 0.0005, maxdepth = 7),
               parms = list(split = "gini")
)

rpart.plot(tree1, type = 0, digits = 2)
tree1


# Obtain predicted probabilities for train and for test.
pred.prob.tr <- predict(tree1, type = "prob")
pred.prob.te <- predict(tree1, type = "prob", newdata = data_test)


library(pROC)
print("Training ROC and AUC")
roc <- roc(data_train$purchase, pred.prob.tr[, "1"])
par(pty = "s")
plot(roc)

pROC::auc(roc)


# Do the same for test.
print("Test ROC and AUC")
roc2 <- roc(data_test$purchase, pred.prob.te[, "1"])
par(pty = "s")
plot(roc2)

pROC::auc(roc2)

#Use complexity pruning to construct a smaller tree

tree1$cptable # This code displays the complexity parameter table for tree1.
# Select the optimal pruning parameter from the table.
plotcp(tree1)

#The complexity parameter (CP) is used to find the optimal tree size and reduce the overfitting seen
#above.  The optimal CP is the one that minimizes the cross validation error (in the xerror column). 
#Row 6 accomplishes that, with CP = 0.0021705426. Pruning with this CP value will result in a tree with 7 splits and so 8 leaves

#Prune tree with optimal cp parameter
tree2 <- prune(tree1, cp = .0022, "CP")

# Show the pruned tree.
rpart.plot(tree2)
tree2

# Obtain predicted probabilities for train and for test.
pred.prune.prob.tr <- predict(tree2, type = "prob")
pred.prune.prob.te <- predict(tree2, type = "prob", newdata = data_test)

# Construct ROC and calculate AUC for the training data.
library(pROC)
print("Training ROC and AUC")
roc <- roc(data_train$purchase, pred.prune.prob.tr[, "1"])
par(pty = "s")
plot(roc)

pROC::auc(roc)

# Do the same for test.
print("Test ROC and AUC")
roc2 <- roc(data_test$purchase, pred.prune.prob.te[, "1"])
par(pty = "s")
plot(roc2)

pROC::auc(roc2)

#The train and test AUCs are 0.7682 and 0.7593 respectively 

#While the train AUC had to come down
#from the 0.7744 of the previous tree due to being a simpler model, the test AUC actually came up from
#0.7500 of the previous tree despite the similar model. The overfitting of the prior tree has been reduced
#with this pruned tree and the simpler model has been shown to be the better predictor on new data

#52% of the past experience used for training falls into the leaf farthest to the left, when interest
#rates exceed 2.92% in the months from May to August or in November. Only 25% of these made
#a purchase, and the model predicts no purchase for future prospects in this situation. The
#combination of high interest rates and summer months appear to be a particularly poor
#combination for marketing our products, as this group had the lowest historical purchase rate of all eight groups

#Another 25% of the past experience used for training falls into the leaf farthest to the right,
#when interest rates are less than 2.92% and the consumer confidence index (CCI) is higher
#(greater than or equal to -43.5). 84% of these made a purchase, regardless of the month, and
#the model predicts a purchase for future prospects in this situation. High consumer confidence
#is a good time to market our products but only when interest rates are low enough

#Based on the final results, I believe the pruned decision tree is the best model  to use
#The pruned decision tree is a far simpler model where seemingly important interaction effects are noted. 
#Its predictive power is not far off from that of the more complex GLM, and the ease of explaining this model, without worrying about odds ratios,
#making this the preferred choice
