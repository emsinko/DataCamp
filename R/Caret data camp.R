# Fit lm model using 10-fold CV: model
model <- train(
  price ~ . , diamonds,
  method = "lm",
  trControl = trainControl(
    method = "cv", number = 10,
    verboseIter = TRUE
  )
)

# Print model to console
model


###################################################x
# Fit lm model using 5-fold CV: model
model <- train(
  medv ~ . , Boston,
  method = "lm",
  trControl = trainControl(
    method = "cv", number = 5,
    verboseIter = TRUE
  )
)

# Print model to console
model

###################################################x

# Fit lm model using 5 x 5-fold CV: model
model <- train(
  medv ~ ., Boston,
  method = "lm",
  trControl = trainControl(
    method = "cv", number = 5,
    repeats = 5, verboseIter = TRUE
  )
)

# Print model to console
model

# Predict on full Boston dataset
predict(model, Boston)
###################################################x


#LOGISTIC ON SONAR

# Shuffle row indices: rows
rows <- sample(nrow(Sonar))

# Randomly order data: Sonar
Sonar <- Sonar[rows,]

# Identify  split on: split
split <- round(nrow(Sonar) * 0.6)

# Create train
train <- Sonar[1:split,]

# Create test
test <- Sonar[-(1:split),]

# Fit glm model: model
model <- glm(Class ~ . , family = "binomial", train)

# Predict on test: p
p <- predict(model,test, type  = "response")
p


# Calculate class probabilities: p_class
p_class <- ifelse(p > 0.5, "M", "R")

# Create confusion matrix
confusionMatrix(p_class, test$Class)

confusionMatrix(p_class, test[["Class"]])
confusionMatrix(p_class, test[["Class"]])


##############################################x

#Customizing train control

# Create trainControl object: myControl
myControl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE
)

# Train glm with custom trainControl: model
model <- train(Class~ . , data = Sonar,method = "glm",trControl = myControl)


# Print model to console
model


# Fit random forest: model
model <- train(
  quality ~ . ,
  tuneLength = 1,
  data = wine, method = "ranger",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)

# Print model to console
model


##############################################

# Fit random forest: model
model <- train(
  quality ~ . ,
  tuneLength = 3,
  data = wine, method = "ranger",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)

# Print model to console
print(model)

# Plot model
plot(model)


#############################################

# Fit random forest: model
model <- train(
  quality ~ .,
  data = wine, method = "ranger",
  tuneGrid = data.frame(mtry = c(2, 3, 7)),
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)

# Print model to console
model

# Plot model
plot(model)


#############################################
# GLM NET MODEL

# Create custom trainControl: myControl
myControl <- trainControl(
  method = "cv", number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE
)

# Fit glmnet model: model
model <- train(
  y ~ . , data = overfit,
  method = "glmnet",
  trControl = myControl
)

# Print model to console
model

# Print maximum ROC statistic
max(model[["results"]]$ROC)


# Train glmnet with custom trainControl and tuning: model
model <- train(
  y ~ . , data = overfit,
  tuneGrid = expand.grid(alpha = 0:1, lambda = seq(0.0001, 1, length = 20)),
  method = "glmnet",
  trControl = myControl
)

# Print model to console
model

# Print maximum ROC statistic
max(model[["results"]][["ROC"]])



################################xxxxxx

#PRE-PROCESSING 

# Apply median imputation: model
model <- train(
  x = breast_cancer_x, y = breast_cancer_y,
  method = "glm",
  trControl = myControl,
  preProcess = "medianImpute"
)

# Print model to console
model


# Apply KNN imputation: model2
model2 <- train(
  x = breast_cancer_x, y = breast_cancer_y,
  method = "glm",
  trControl = myControl,
  preProcess = "knnImpute"
)

# Print model to console
model2


#Porovnanie
dotplot(resamples, metric = "ROC")




# Fit glm with median imputation: model1
model1 <- train(
  x = breast_cancer_x, y = breast_cancer_y,
  method = "glm",
  trControl = myControl,
  preProcess = "medianImpute"
)

# Print model1
model1

# Fit glm with median imputation and standardization: model2
model2 <- train(
  x = breast_cancer_x, y = breast_cancer_y,
  method = "glm",
  trControl = myControl,
  preProcess = c("medianImpute","center", "scale")
)

# Print model2
model2



# Identify near zero variance predictors: remove_cols
remove_cols <- nearZeroVar(bloodbrain_x, names = TRUE, 
                           freqCut = 2, uniqueCut = 20)

# Get all column names from bloodbrain_x: all_cols
all_cols <- names(bloodbrain_x)

# Remove from data: bloodbrain_x_small
bloodbrain_x_small <- bloodbrain_x[ , setdiff(all_cols, remove_cols)]





# Fit model on reduced data: model
model <- train(x = bloodbrain_x_small, y = bloodbrain_y, method = "glm")

# Print model to console
model




# Fit glm model using PCA: model
model <- train(
  x = bloodbrain_x, y = bloodbrain_y,
  method = "glm", preProcess = "pca"
)

# Print model to console
model

#PCA ZVLADA LEPSIE NEAR ZERO, ako keby sme ich uplne vyhodili





# POSLEDNA KAPITOLA

# Create custom indices: myFolds
myFolds <- createFolds(churn_y, k = 5)

# Create reusable trainControl object: myControl
myControl <- trainControl(
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE,
  savePredictions = TRUE,
  index = myFolds
)


# Fit glmnet model: model_glmnet
model_glmnet <- train(
  x = churn_x, y = churn_y,
  metric = "ROC",
  method = "glmnet",
  trControl = myControl
)

plot(model_glmnet)



# Fit random forest: model_rf
model_rf <- train(
  x = churn_x, y = churn_y,
  metric = "ROC",
  method = "ranger",
  trControl = myControl
)


# COMPARING MODELS

# Create model_list
model_list <- list(item1 = model_glmnet, item2 = model_rf)

# Pass model_list to resamples(): resamples
resamples <- resamples(model_list)

# Summarize the results
summary(resamples)


# Create bwplot
bwplot(resamples, metric = "ROC")
xyplot(resamples, metric = "ROC")
summary(resamples)

########################xx
#STACKING

# Create ensemble model: stack
stack <- caretStack(model_list, method = "glm") 

# Look at summary
summary(stack)