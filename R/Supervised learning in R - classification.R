# Supervised learning in R - classification

# Chapter 1: ----


# Load the 'class' package
#install.packages("class")
library(class)

# Create a vector of labels
sign_types <- signs$sign_type

# Classify the next sign observed
knn(train = signs[-1], test = next_sign, cl = sign_types)  # k je defaultne rovne 1 v tomto baliku


# Examine the structure of the signs dataset
str(signs)

# Count the number of signs of each type
table(signs$sign_type)

# Check r10's average red level by sign type
aggregate(r10 ~ sign_type, data = signs, mean)


##
# Use kNN to identify the test road signs
sign_types <- signs$sign_type
signs_pred <- knn(train = signs[-1], test = test_signs[-1], cl = sign_types)

# Create a confusion matrix of the actual versus predicted values
signs_actual <- test_signs$sign_type 
table(signs_pred, signs_actual)

# Compute the accuracy
mean(test_signs$sign_type == signs_pred)


####### 
## k mal� --> prisposobuje sa hranicna ciara,  k velke ---> podoba saa na priamkove rozhranie 

# Compute the accuracy of the baseline model (default k = 1)
k_1 <- knn(train = signs[-1], test = signs_test[-1], cl = sign_types)
mean(k_1 == signs_actual)

# Modify the above to set k = 7
k_7 <- knn(train = signs[-1], test = signs_test[-1], cl = sign_types, k = 7)
mean(k_7 == signs_actual)

# Set k = 15 and compare to the above
k_15 <- knn(train = signs[-1], test = signs_test[-1], cl = sign_types, k = 15)
mean(k_15 == signs_actual)




# Use the prob parameter to get the proportion of votes for the winning class
sign_pred <- knn(train = signs[-1], test = signs_test[-1],cl = sign_types, k = 7, prob = TRUE)

# Get the "prob" attribute from the predicted classes
sign_prob <- attr(sign_pred, "prob")

# Examine the first several predictions
head(sign_pred)

# Examine the proportion of votes for the winning class
head(sign_prob)




# Chapter 2 : ----

### Naive Bayes:

where9am <-structure(list(daytype = structure(c(1L, 1L, 1L, 2L, 2L, 1L, 
                                                1L, 1L, 1L, 1L, 2L, 2L, 1L, 1L, 1L, 1L, 1L, 2L, 2L, 1L, 1L, 1L, 
                                                1L, 1L, 2L, 2L, 1L, 1L, 1L, 1L, 1L, 2L, 2L, 1L, 1L, 1L, 1L, 1L, 
                                                2L, 2L, 1L, 1L, 1L, 1L, 1L, 2L, 2L, 1L, 1L, 1L, 1L, 1L, 2L, 2L, 
                                                1L, 1L, 1L, 1L, 1L, 2L, 2L, 1L, 1L, 1L, 1L, 1L, 2L, 2L, 1L, 1L, 
                                                1L, 1L, 1L, 2L, 2L, 1L, 1L, 1L, 1L, 1L, 2L, 2L, 1L, 1L, 1L, 1L, 
                                                1L, 2L, 2L, 1L, 1L), .Label = c("weekday", "weekend"), class = "factor"), 
                          location = structure(c(4L, 4L, 4L, 3L, 3L, 2L, 3L, 1L, 4L, 
                                                 4L, 3L, 3L, 4L, 3L, 4L, 4L, 2L, 3L, 3L, 2L, 4L, 4L, 4L, 4L, 
                                                 3L, 3L, 3L, 3L, 4L, 3L, 4L, 3L, 3L, 4L, 4L, 4L, 4L, 4L, 3L, 
                                                 3L, 2L, 3L, 3L, 4L, 3L, 3L, 3L, 4L, 4L, 4L, 2L, 3L, 3L, 3L, 
                                                 4L, 4L, 4L, 2L, 4L, 3L, 3L, 4L, 4L, 4L, 3L, 4L, 3L, 3L, 2L, 
                                                 3L, 2L, 4L, 4L, 3L, 3L, 4L, 4L, 3L, 2L, 2L, 3L, 3L, 3L, 3L, 
                                                 4L, 4L, 3L, 3L, 3L, 4L, 4L), .Label = c("appointment", "campus", 
                                                                                         "home", "office"), class = "factor")), .Names = c("daytype", 
                                                                                                                                           "location"), row.names = c(NA, -91L), class = "data.frame")


# Compute P(A) 
p_A <- nrow(subset(where9am, location == "office")) / nrow(where9am)

# Compute P(B)
p_B <- nrow(subset(where9am, daytype == "weekday")) / nrow(where9am)

# Compute the observed P(A and B)
p_AB <- nrow(subset(where9am,location == "office" & daytype == "weekday"))/ nrow(where9am)

# Compute P(A | B) and print its value
p_A_given_B <- p_AB/p_B
print(p_A_given_B)



# Load the naivebayes package
# install.packages("naivebayes")
library(naivebayes)

# Build the location prediction model
locmodel <- naive_bayes(location ~ daytype, data = where9am)

### Predict Thursday's 9am location

thursday9am <- data.frame(daytype = "weekday")
predict(locmodel, thursday9am)

### Predict Saturdays's 9am location

thursday9am <- data.frame(daytype = "weekend")  
predict(locmodel, saturday9am)



# The 'naivebayes' package is loaded into the workspace
# and the Naive Bayes 'locmodel' has been built

# Examine the location prediction model
locmodel

# Obtain the predicted probabilities for Thursday at 9am
predict(locmodel, thursday9am , type = "prob")

# Obtain the predicted probabilities for Saturday at 9am
predict(locmodel, saturday9am , type = "prob")




# The 'naivebayes' package is loaded into the workspace already

# Build a NB model of location
locmodel <- naive_bayes(location ~ daytype + hourtype, data = locations)

# Predict Brett's location on a weekday afternoon
predict(locmodel, newdata = weekday_afternoon)

# Predict Brett's location on a weekday evening
predict(locmodel, newdata = weekday_evening)


######
## Laplaceova korekcia pre Naive Bayes

# Naive bayes assumption: Pri viacej premennych sa nerobi joint probability vsetkych, ale nasobia pre vsetky premenne prieniky targetom pricom sa beru ako independent events.
#     Problem je taky, ze ak mame nejaku premennu, ktora m� 0 eventov (byt v pr�ci cez v�kend) tak automaticky sa pri nasobeni 0 dostane 0-pravdepodobnost
#     To sa vyriesi Laplaceovou korekciou, kotra prida jeden event do intersectionu

'Preparing for unforeseen circumstances
While Brett was tracking his location over 13 weeks, he never went into the office during the weekend. Consequently, 
the joint probability of P(office and weekend) = 0. Explore how this impacts the predicted probability that Brett may 
go to work on the weekend in the future. Additionally, you can see how using the Laplace correction will allow a small 
chance for these types of unforeseen circumstances. '

# The 'naivebayes' package is loaded into the workspace already
# The Naive Bayes location model (locmodel) has already been built

# Observe the predicted probabilities for a weekend afternoon
predict(locmodel, newdata = weekend_afternoon, type = "prob")

# Build a new model using the Laplace correction
locmodel2 <- naive_bayes(formula(locmodel)
                         ,data = locations, laplace = 1)

# Observe the new predicted probabilities for a weekend afternoon
predict(locmodel2, newdata = weekend_afternoon, type = "prob")

# Kazdy event tym padom moze nastat


#####
### Pouzitelnost modelu

# Naive Bayes je dobry pouzivat ked mame numericke data rozkategorizovane. Kedze v pozadi sa pocita vzdy pravdepodobnost interceptu mmdezi predictorom a targetom
# Je dobre predpripravit data do modelu roznym binningom. Pre text data je fajn pouzivat koncept "bag of words" - nebere do uvahy semantiku, postupnost slov, gramatiku


####################################### -
##### Logistic regression Chapter 3 ###
####################################### -

# Examine the dataset to identify potential independent variables
str(donors)

# Explore the dependent variable
table(donors$donated)

# Build the donation model
donation_model <- glm(donated ~ bad_address + interest_religion + interest_veterans, 
                      data = donors, family = "binomial")

# Summarize the model results
summary(donation_model)


# Estimate the donation probability
donors$donation_prob <- predict(donation_model,donors, type = "response")

# Find the donation probability of the average prospect
mean(donors$donated)

# Predict a donation if probability of donation is greater than average (0.0504)
donors$donation_pred <- ifelse(donors$donation_prob > 0.0504, 1, 0)

# Calculate the model's accuracy
mean(donors$donation_pred == donors$donated)


# Load the pROC package
#install.packages("pROC")
library(pROC)

# Create a ROC curve
ROC <- roc(donors$donated, donors$donation_prob)  # actual vs predicted

# Plot the ROC curve
plot(ROC, col = "blue")

# Calculate the area under the curve (AUC)
auc(ROC)   #Area under the curve: 0.5102
 


# Convert the wealth rating to a factor
donors$wealth_rating <- factor(donors$wealth_rating, levels = c(0,1,2,3), labels = c("Unknown","Low","Medium","High"))

# Use relevel() to change reference category 
donors$wealth_rating <- relevel(donors$wealth_rating, ref = "Medium")

# See how our factor coding impacts the model
summary(glm(donated ~ wealth_rating, data = donors, family = "binomial"))




# Find the average age among non-missing values
summary(donors$age)

# Impute missing age values with mean(age)
donors$imputed_age <- ifelse(is.na(donors$age),round(mean(donors$age, na.rm = TRUE),2),donors$age)

# Create missing value indicator for age
donors$missing_age <- ifelse(is.na(donors$age),1,0)




# Build a recency, frequency, and money (RFM) model
rfm_model <- glm(donated ~ money + recency * frequency, data = donors, family = "binomial")

# Summarize the RFM model to see how the parameters were coded
summary(rfm_model)

# Compute predicted probabilities for the RFM model
rfm_prob <- predict(rfm_model, newdata = donors, type = "response")

# Plot the ROC curve and find AUC for the new model
library(pROC)
ROC <- roc(donors$donated, rfm_prob)
plot(ROC, col = "red")
auc(ROC)


### STEPWISE-REGRESSION

# Specify a null model with no predictors
null_model <- glm(donated ~ 1 , data = donors, family = "binomial")

# Specify the full model using all of the potential predictors
full_model <- glm(donated ~ . , data = donors, family = "binomial")


# Use a forward stepwise algorithm to build a parsimonious model
step_model <- step(null_model, scope = list(lower = null_model, upper = full_model), direction = "forward")

# Estimate the stepwise donation probability
step_prob <- predict(step_model, newdata = donors, type = "response")

# Plot the ROC of the stepwise model
library(pROC)
ROC <- roc(donors$donated, step_prob)
plot(ROC, col = "red")
auc(ROC)


######################################x
#######################################
### CHAPTER 4  - classification trees

# Load the rpart package
library(rpart)

# Build a lending model predicting loan outcome versus loan amount and credit score
loan_model <- rpart(outcome ~ loan_amount + credit_score, data = loans, method = "class", control = rpart.control(cp = 0))

#good_credit <- head(subset(loans, credit_score == "HIGH"),1)
#bad_credit <- head(subset(loans, credit_score == "LOW"),1)

# Make a prediction for someone with good credit
predict(loan_model, newdata = good_credit, type = "class")

# Make a prediction for someone with bad credit
predict(loan_model, newdata = bad_credit, type = "class")


# Examine the loan_model object
loan_model

# Load the rpart.plot package
library(rpart.plot)

# Plot the loan_model with default settings
rpart.plot(loan_model)

# Plot the loan_model with customized settings
rpart.plot(loan_model, type = 3, box.palette = c("red", "green"), fallen.leaves = TRUE)



# Determine the number of rows for training
nrow(loans) * 0.75

# Create a random sample of row IDs
sample_rows <- sample(11312, 8484)

# Create the training dataset
loans_train <- loans[sample_rows,]

# Create the test dataset
loans_test <- loans[-sample_rows,]




# Grow a tree using all of the available applicant data
loan_model <- rpart(outcome ~ . , data = loans_train, method = "class", control = rpart.control(cp = 0))

# Make predictions on the test dataset
loans_test$pred <- predict(loan_model, newdata = loans_test, type = "class")

# Examine the confusion matrix
table(loans_test$pred, loans_test$outcome)

# Compute the accuracy on the test dataset
mean(loans_test$pred == loans_test$outcome)



# Grow a tree with maxdepth of 6
loan_model <- rpart(outcome ~ ., data = loans_train, method = "class",control = rpart.control(cp = 0, maxdepth = 6))

# Make a class prediction on the test set
loans_test$pred <- predict(loan_model, newdata = loans_test, type = "class")

# Compute the accuracy of the simpler tree
mean(loans_test$pred == loans_test$outcome)

# Swap maxdepth for a minimum split of 500 
loan_model <- rpart(outcome ~ ., data = loans_train, method = "class", control = rpart.control(cp = 0, maxdepth = 6))

# Run this. How does the accuracy change?
loans_test$pred <- predict(loan_model, loans_test, type = "class")
mean(loans_test$pred == loans_test$outcome)





# Grow an overly complex tree
loan_model <- rpart(outcome ~ ., data = loans_train, method = "class",control = rpart.control(cp = 0))

# Examine the complexity plot
plotcp(loan_model)
#printcp(loan_model)

# Prune the tree
loan_model_pruned <- prune(loan_model, cp = 0.0014)

# Compute the accuracy of the pruned tree
loans_test$pred <- predict(loan_model_pruned, newdata = loans_test, type = "class")
mean(loans_test$pred == loans_test$outcome)

### Random forest

# Load the randomForest package
library(randomForest)

# Build a random forest model
loan_model <- randomForest(outcome ~ .,  data = loans_train)

# Compute the accuracy of the random forest
loans_test$pred <- predict(loan_model, loans_test, type = "class")
mean(loans_test$pred == loans_test$outcome)










