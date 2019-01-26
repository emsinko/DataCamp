#####################################
### CHAPTER 1 - LINEAR REGRESSION ###
#####################################

unemployment <- structure(list(male_unemployment = c(2.900000095, 6.699999809, 
                                     4.900000095, 7.900000095, 9.800000191, 6.900000095, 6.099999905, 
                                     6.199999809, 6, 5.099999905, 4.699999809, 4.400000095, 5.800000191
), female_unemployment = c(4, 7.400000095, 5, 7.199999809, 7.900000095, 
                           6.099999905, 6, 5.800000191, 5.199999809, 4.199999809, 4, 4.400000095, 
                           5.199999809)), .Names = c("male_unemployment", "female_unemployment"
                           ), class = "data.frame", row.names = c(NA, -13L))


# unemployment is loaded in the workspace
summary(unemployment)

# Define a formula to express female_unemployment as a function of male_unemployment
fmla <- as.formula("female_unemployment ~ male_unemployment")

# Print it
fmla

# Use the formula to fit a model: unemployment_model
unemployment_model <- lm(fmla,  data = unemployment)

# Print it
unemployment_model



# broom and sigr are already loaded in your workspace
#install.packages("broom")
#install.packages("sigr")
library(broom)
library(sigr)

# Print unemployment_model
unemployment_model

# Call summary() on unemployment_model to get more details
summary(unemployment_model)

# Call glance() on unemployment_model to see the details in a tidier form
broom::glance(x = unemployment_model)

# Call wrapFTest() on unemployment_model to see the most relevant details
sigr::wrapFTest(unemployment_model)



# unemployment is in your workspace
summary(unemployment)

# newrates is in your workspace
newrates <- data.frame(male_unemployment = 5)

# Predict female unemployment in the unemployment data set
unemployment$prediction <-  predict(unemployment_model, data = unemployment)

# load the ggplot2 package
library(ggplot2)

# Make a plot to compare predictions to actual (prediction on x axis). 
ggplot(unemployment, aes(x = prediction, y = female_unemployment)) + 
  geom_point() +
  geom_abline(color = "blue")

# Predict female unemployment rate when male unemployment is 5%
pred <- predict(unemployment_model, newdata = newrates)
# Print it
pred



bloodpressure <- structure(list(blood_pressure = c(132L, 143L, 153L, 162L, 154L, 
                                                   168L, 137L, 149L, 159L, 128L, 166L), age = c(52L, 59L, 67L, 73L, 
                                                                                                64L, 74L, 54L, 61L, 65L, 46L, 72L), weight = c(173L, 184L, 194L, 
                                                                                                                                               211L, 196L, 220L, 188L, 188L, 207L, 167L, 217L)), .Names = c("blood_pressure", 
                                                                                                                                                                                                            "age", "weight"), class = "data.frame", row.names = c(NA, -11L
                                                                                                                                                                                                            ))
# bloodpressure is in the workspace
summary(bloodpressure)

# Create the formula and print it
fmla <- as.formula("blood_pressure ~ age + weight")
print(fmla)

# Fit the model: bloodpressure_model
bloodpressure_model <- lm(fmla, data = bloodpressure)

# Print bloodpressure_model and call summary() 
print(bloodpressure_model)
summary(bloodpressure_model)


# bloodpressure_model is in your workspace
bloodpressure_model

# predict blood pressure using bloodpressure_model :prediction
bloodpressure$prediction <- predict(bloodpressure_model, newdata = bloodpressure)

# plot the results
ggplot(aes(x = prediction, y = blood_pressure), data = bloodpressure) + 
  geom_point() +
  geom_abline(color = "blue")


#####################################
### CHAPTER 2 EVALUATION OF MODEL ###
#####################################

# unemployment is in the workspace
summary(unemployment)

# unemployment_model is in the workspace
summary(unemployment_model)

# Make predictions from the model
unemployment$predictions <- predict(unemployment_model, unemployment)


# Fill in the blanks to plot predictions (on x-axis) versus the female_unemployment rates
ggplot(unemployment, aes(x = predictions, y = female_unemployment)) + 
  geom_point() + 
  geom_abline()

# Calculate residuals
unemployment$residuals <- unemployment$female_unemployment - unemployment$predictions

# Fill in the blanks to plot predictions (on x-axis) versus the residuals
ggplot(unemployment, aes(x = predictions, y = residuals)) + 
  geom_pointrange(aes(ymin = 0, ymax = residuals)) + 
  geom_hline(yintercept = 0, linetype = 3) + 
  ggtitle("residuals vs. linear model prediction")






# Load the package WVPlots
# install.packages("WVPlots")
library(WVPlots)

# Plot the Gain Curve
WVPlots::GainCurvePlot(unemployment, xvar = "predictions",truthVar = "female_unemployment", title = "Unemployment model")



# unemployment is in the workspace
summary(unemployment)

# For convenience put the residuals in the variable res
res <- unemployment$residuals

# Calculate RMSE, assign it to the variable rmse and print it
(rmse <- sqrt(mean(res^2)))

# Calculate the standard deviation of female_unemployment and print it
(sd_unemployment <- sd(unemployment$female_unemployment))

#### POROVNAVANIE RMSE(res) a SD(y) je ekvivalentne! Porovnanie uboheho modelu (SD(y)) s inym modelom RMSE



# Calculate mean female_unemployment: fe_mean. Print it
(fe_mean <- mean(unemployment$female_unemployment))

# Calculate total sum of squares: tss. Print it
(tss <- sum((unemployment$female_unemployment - fe_mean)^2))

# Calculate residual sum of squares: rss. Print it
(rss <- sum((unemployment$residuals)^2) )

# Calculate R-squared: rsq. Print it. Is it a good fit?
(rsq <- 1 - rss / tss)

# Get R-squared from glance. Print it
(rsq_glance <- glance(unemployment_model)$r.squared)


##### Korelacia vs R2

# unemployment is in your workspace
summary(unemployment)

# unemployment_model is in the workspace
summary(unemployment_model)



# Get the correlation between the prediction and true outcome: rho and print it
(rho <- cor(unemployment$predictions, unemployment$female_unemployment))

# Square rho: rho2 and print it
(rho2 <- rho^2)

# Get R-squared from glance and print it
(rsq_glance <- glance(unemployment_model)$r.squared)



# mpg is in the workspace
summary(mpg)
dim(mpg)

# Use nrow to get the number of rows in mpg (N) and print it
(N <- nrow(mpg))

# Calculate how many rows 75% of N should be and print it
# Hint: use round() to get an integer
(target <- round(0.75 * N))

# Create the vector of N uniform random variables: gp
gp <- runif(N)

# Use gp to create the training set: mpg_train (75% of data) and mpg_test (25% of data)
mpg_train <-  mpg[gp < 0.75, ]
mpg_test <- mpg[gp >= 0.75, ]

# Use nrow() to examine mpg_train and mpg_test
nrow(mpg_train)
nrow(mpg_test)




# mpg_train is in the workspace
summary(mpg_train)

# Create a formula to express cty as a function of hwy: fmla and print it.
(fmla <- as.formula("cty~ hwy"))

# Now use lm() to build a model mpg_model from mpg_train that predicts cty from hwy 
mpg_model <- lm(fmla , data = mpg_train)

# Use summary() to examine the model
summary(mpg_model)

















###################################################
### CHAPTER 4 : Dealing with Non-Linear Responses
################################################

sparrow <- structure(list(status = structure(c(2L, 2L, 2L, 2L, 2L, 2L, 2L, 
                                               2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 
                                               2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 1L, 1L, 1L, 1L, 
                                               1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 
                                               1L, 1L, 1L, 1L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 2L, 
                                               2L, 2L, 2L, 2L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L
), .Label = c("Perished", "Survived"), class = "factor"), age = c("adult", 
                                                                  "adult", "adult", "adult", "adult", "adult", "adult", "adult", 
                                                                  "adult", "adult", "adult", "adult", "adult", "adult", "adult", 
                                                                  "adult", "adult", "adult", "adult", "adult", "adult", "adult", 
                                                                  "adult", "adult", "adult", "adult", "adult", "adult", "adult", 
                                                                  "adult", "adult", "adult", "adult", "adult", "adult", "adult", 
                                                                  "adult", "adult", "adult", "adult", "adult", "adult", "adult", 
                                                                  "adult", "adult", "adult", "adult", "adult", "adult", "adult", 
                                                                  "adult", "adult", "adult", "adult", "adult", "adult", "adult", 
                                                                  "adult", "adult", "juvenile", "juvenile", "juvenile", "juvenile", 
                                                                  "juvenile", "juvenile", "juvenile", "juvenile", "juvenile", "juvenile", 
                                                                  "juvenile", "juvenile", "juvenile", "juvenile", "juvenile", "juvenile", 
                                                                  "juvenile", "juvenile", "juvenile", "juvenile", "juvenile", "juvenile", 
                                                                  "juvenile", "juvenile", "juvenile", "juvenile", "juvenile", "juvenile"
), total_length = c(154L, 160L, 155L, 154L, 156L, 161L, 157L, 
                    159L, 158L, 158L, 160L, 162L, 161L, 160L, 159L, 158L, 159L, 166L, 
                    159L, 160L, 161L, 163L, 156L, 165L, 160L, 158L, 160L, 157L, 159L, 
                    160L, 158L, 161L, 160L, 160L, 153L, 165L, 160L, 161L, 162L, 163L, 
                    162L, 163L, 161L, 160L, 162L, 160L, 161L, 162L, 165L, 161L, 161L, 
                    162L, 164L, 158L, 162L, 156L, 166L, 165L, 166L, 156L, 156L, 163L, 
                    163L, 160L, 156L, 162L, 163L, 164L, 163L, 160L, 160L, 158L, 158L, 
                    158L, 155L, 160L, 156L, 158L, 166L, 165L, 157L, 164L, 166L, 167L, 
                    161L, 166L, 161L), wingspan = c(241L, 252L, 243L, 245L, 247L, 
                                                    253L, 251L, 247L, 247L, 252L, 252L, 253L, 243L, 250L, 247L, 253L, 
                                                    247L, 253L, 247L, 248L, 252L, 251L, 242L, 251L, 247L, 244L, 242L, 
                                                    245L, 244L, 253L, 245L, 247L, 247L, 247L, 241L, 249L, 245L, 249L, 
                                                    246L, 250L, 247L, 246L, 246L, 242L, 246L, 249L, 250L, 248L, 252L, 
                                                    243L, 244L, 248L, 244L, 247L, 253L, 239L, 251L, 253L, 250L, 246L, 
                                                    245L, 248L, 248L, 250L, 237L, 253L, 254L, 251L, 244L, 247L, 250L, 
                                                    247L, 249L, 243L, 237L, 249L, 236L, 240L, 245L, 255L, 238L, 250L, 
                                                    256L, 255L, 246L, 254L, 251L), weight = c(24.5, 26.9, 26.9, 24.3, 
                                                                                              24.1, 26.5, 24.6, 24.2, 23.6, 26.2, 26.2, 24.8, 25.4, 23.7, 25.7, 
                                                                                              25.7, 26.5, 26.7, 23.9, 24.7, 28, 27.9, 25.9, 25.7, 26.6, 23.2, 
                                                                                              25.7, 26.3, 24.3, 26.7, 24.9, 23.8, 25.6, 27, 24.7, 26.5, 26.1, 
                                                                                              25.6, 25.9, 25.5, 27.6, 25.8, 24.9, 26, 26.5, 26, 27.1, 25.1, 
                                                                                              26, 25.6, 25, 24.6, 25, 26, 28.3, 24.6, 27.5, 31, 28.3, 24.6, 
                                                                                              25.5, 24.8, 26.3, 24.4, 23.3, 26.7, 26.4, 26.9, 24.3, 27, 26.8, 
                                                                                              24.9, 26.1, 26.6, 23.3, 24, 26.8, 23.5, 26.9, 28.6, 24.7, 27.3, 
                                                                                              25.7, 29, 25, 27.5, 26), beak_head = c(31.2, 30.8, 30.6, 31.7, 
                                                                                                                                     31.5, 31.8, 31.1, 31.4, 29.8, 32, 32, 32.3, 31.8, 29.8, 31.4, 
                                                                                                                                     31.9, 31.6, 32.5, 31.4, 31.3, 31.8, 31.9, 32, 32.2, 32.4, 31.6, 
                                                                                                                                     31.6, 32.2, 31.5, 32.1, 31.4, 31.4, 32.3, 32, 32.2, 31, 32, 32.3, 
                                                                                                                                     32.3, 32.5, 31.8, 31.4, 30.5, 31, 31.5, 31.4, 31.6, 31.9, 32.3, 
                                                                                                                                     32.5, 31.3, 31, 31.2, 32, 31.8, 30.5, 31.5, 32.4, 32.4, 32, 32.1, 
                                                                                                                                     32.2, 33, 31.5, 30.6, 32, 32, 32, 31.3, 31.5, 32.5, 32.4, 32.2, 
                                                                                                                                     32.4, 30.2, 30.4, 30.2, 31, 31.7, 31.5, 31.2, 31.8, 31.7, 32.2, 
                                                                                                                                     31.5, 31.4, 31.5), humerus = c(0.69, 0.74, 0.73, 0.74, 0.71, 
                                                                                                                                                                    0.78, 0.74, 0.73, 0.7, 0.75, 0.74, 0.77, 0.72, 0.73, 0.73, 0.74, 
                                                                                                                                                                    0.73, 0.77, 0.75, 0.75, 0.77, 0.77, 0.72, 0.75, 0.73, 0.73, 0.71, 
                                                                                                                                                                    0.74, 0.72, 0.74, 0.73, 0.74, 0.76, 0.75, 0.73, 0.74, 0.74, 0.74, 
                                                                                                                                                                    0.74, 0.75, 0.73, 0.69, 0.74, 0.75, 0.72, 0.73, 0.74, 0.74, 0.73, 
                                                                                                                                                                    0.71, 0.7, 0.71, 0.7, 0.73, 0.75, 0.66, 0.72, 0.76, 0.75, 0.74, 
                                                                                                                                                                    0.76, 0.74, 0.74, 0.75, 0.69, 0.76, 0.77, 0.75, 0.72, 0.76, 0.76, 
                                                                                                                                                                    0.75, 0.74, 0.75, 0.69, 0.74, 0.69, 0.71, 0.71, 0.77, 0.68, 0.76, 
                                                                                                                                                                    0.75, 0.76, 0.74, 0.76, 0.73), femur = c(0.67, 0.71, 0.7, 0.69, 
                                                                                                                                                                                                             0.71, 0.74, 0.74, 0.72, 0.67, 0.74, 0.72, 0.75, 0.72, 0.7, 0.72, 
                                                                                                                                                                                                             0.7, 0.71, 0.76, 0.72, 0.74, 0.73, 0.75, 0.71, 0.74, 0.71, 0.71, 
                                                                                                                                                                                                             0.7, 0.73, 0.7, 0.71, 0.7, 0.69, 0.75, 0.74, 0.68, 0.7, 0.71, 
                                                                                                                                                                                                             0.72, 0.71, 0.73, 0.72, 0.66, 0.73, 0.71, 0.7, 0.69, 0.71, 0.72, 
                                                                                                                                                                                                             0.71, 0.71, 0.69, 0.7, 0.69, 0.71, 0.72, 0.66, 0.69, 0.75, 0.72, 
                                                                                                                                                                                                             0.74, 0.72, 0.73, 0.7, 0.71, 0.66, 0.73, 0.75, 0.74, 0.68, 0.73, 
                                                                                                                                                                                                             0.73, 0.72, 0.74, 0.71, 0.65, 0.72, 0.67, 0.7, 0.69, 0.74, 0.68, 
                                                                                                                                                                                                             0.73, 0.75, 0.75, 0.71, 0.74, 0.71), legbone = c(1.02, 1.18, 
                                                                                                                                                                                                                                                              1.15, 1.15, 1.13, 1.14, 1.15, 1.13, 1.08, 1.15, 1.13, 1.13, 1.13, 
                                                                                                                                                                                                                                                              1.1, 1.14, 1.15, 1.15, 1.23, 1.11, 1.18, 1.19, 1.17, 1.12, 1.16, 
                                                                                                                                                                                                                                                              1.11, 1.14, 1.12, 1.14, 1.11, 1.12, 1.12, 1.1, 1.13, 1.17, 1.09, 
                                                                                                                                                                                                                                                              1.1, 1.11, 1.13, 1.13, 1.2, 1.11, 1.07, 1.14, 1.11, 1.09, 1.1, 
                                                                                                                                                                                                                                                              1.12, 1.15, 1.14, 1.12, 1.08, 1.09, 1.07, 1.14, 1.15, 1.04, 1.12, 
                                                                                                                                                                                                                                                              1.18, 1.18, 1.17, 1.15, 1.16, 1.15, 1.17, 1.01, 1.2, 1.16, 1.17, 
                                                                                                                                                                                                                                                              1.08, 1.18, 1.12, 1.14, 1.15, 1.16, 1.01, 1.13, 1.07, 1.11, 1.11, 
                                                                                                                                                                                                                                                              1.17, 1.16, 1.17, 1.19, 1.2, 1.12, 1.12, 1.12), skull = c(0.59, 
                                                                                                                                                                                                                                                                                                                        0.6, 0.6, 0.58, 0.57, 0.61, 0.61, 0.61, 0.6, 0.61, 0.62, 0.63, 
                                                                                                                                                                                                                                                                                                                        0.6, 0.59, 0.59, 0.6, 0.61, 0.6, 0.6, 0.6, 0.59, 0.62, 0.61, 
                                                                                                                                                                                                                                                                                                                        0.61, 0.59, 0.58, 0.62, 0.6, 0.62, 0.59, 0.58, 0.6, 0.61, 0.63, 
                                                                                                                                                                                                                                                                                                                        0.59, 0.61, 0.61, 0.6, 0.61, 0.62, 0.6, 0.6, 0.58, 0.6, 0.61, 
                                                                                                                                                                                                                                                                                                                        0.6, 0.63, 0.59, 0.61, 0.61, 0.6, 0.59, 0.61, 0.61, 0.6, 0.57, 
                                                                                                                                                                                                                                                                                                                        0.61, 0.61, 0.61, 0.59, 0.62, 0.61, 0.61, 0.6, 0.59, 0.63, 0.61, 
                                                                                                                                                                                                                                                                                                                        0.62, 0.61, 0.62, 0.63, 0.59, 0.6, 0.61, 0.59, 0.62, 0.56, 0.6, 
                                                                                                                                                                                                                                                                                                                        0.6, 0.61, 0.6, 0.59, 0.6, 0.64, 0.59, 0.6, 0.59), sternum = c(0.83, 
                                                                                                                                                                                                                                                                                                                                                                                       0.84, 0.85, 0.84, 0.82, 0.89, 0.86, 0.79, 0.82, 0.86, 0.89, 0.92, 
                                                                                                                                                                                                                                                                                                                                                                                       0.89, 0.82, 0.93, 0.86, 0.92, 0.88, 0.82, 0.8, 0.88, 0.86, 0.89, 
                                                                                                                                                                                                                                                                                                                                                                                       0.87, 0.84, 0.89, 0.79, 0.85, 0.85, 0.86, 0.85, 0.78, 0.9, 0.87, 
                                                                                                                                                                                                                                                                                                                                                                                       0.88, 0.85, 0.84, 0.83, 0.87, 0.89, 0.87, 0.84, 0.8, 0.8, 0.81, 
                                                                                                                                                                                                                                                                                                                                                                                       0.85, 0.85, 0.84, 0.89, 0.83, 0.87, 0.84, 0.8, 0.8, 0.86, 0.81, 
                                                                                                                                                                                                                                                                                                                                                                                       0.85, 0.9, 0.92, 0.85, 0.82, 0.85, 0.84, 0.89, 0.77, 0.88, 0.89, 
                                                                                                                                                                                                                                                                                                                                                                                       0.89, 0.89, 0.85, 0.84, 0.87, 0.82, 0.89, 0.79, 0.84, 0.83, 0.81, 
                                                                                                                                                                                                                                                                                                                                                                                       0.85, 0.85, 0.77, 0.86, 0.86, 0.86, 0.85, 0.91, 0.83)), .Names = c("status", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                          "age", "total_length", "wingspan", "weight", "beak_head", "humerus", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                          "femur", "legbone", "skull", "sternum"), row.names = c(NA, -87L
                                                                                                                                                                                                                                                                                                                                                                                                                                        ), class = "data.frame")
# sparrow is in the workspace
summary(sparrow)

# Create the survived column
sparrow$survived <- sparrow$status == "Survived"

# Create the formula
(fmla <- as.formula("survived ~ total_length + weight + humerus"))

# Fit the logistic regression model
sparrow_model <- glm(fmla, data = sparrow, family = binomial)

# Call summary
summary(sparrow_model)

# Call glance
(perf <- glance(sparrow_model))

# Calculate pseudo-R-squared
(pseudoR2 <- 1 - perf$deviance / perf$null.deviance )
 


### Predikcie probability + Gain Curve

# Make predictions
sparrow$pred <- predict(sparrow_model, type = "response")
# ak nedame response, tak predict vrati log odds 

# Look at gain curve
WVPlots::GainCurvePlot(frame = sparrow, xvar = "pred" , truthVar = "survived" , title = "sparrow survival model")


########
## Modelovanie pomocou poisson - quasipoisson

# Poznamky: Poisson predpoklada mean(X) = var(x). Ak to neplati, tak je to quasipoisson
#           Modelujeme county, outcome je integer. Ak je rates/count >> 0, tak je regularna regresia v poriadku 


# Data set bikesJuly predstavuje dataset a pocte vypozicanych bicyklov (cnt). Vars: pocasie, prazdniny, hodina v dni....
# bikesJuly is in the workspace
str(bikesJuly)

# The outcome column
outcome 

# The inputs to use
vars 

# Create the formula string for bikes rented as a function of the inputs
(fmla <- paste(outcome, "~", paste(vars, collapse = " + ")))

# Calculate the mean and variance of the outcome
(mean_bikes <- mean(bikesJuly[[outcome]]))
(var_bikes <- var(bikesJuly[[outcome]]))

# Fit the model
bike_model <- glm(fmla, data = bikesJuly, family = quasipoisson)

# Call glance
(perf <- glance(x = bike_model))

## + validacia na out of sample data 

# bikesAugust is in the workspace
str(bikesAugust)

# bike_model is in the workspace
summary(bike_model)

# Make predictions on August data
bikesAugust$pred  <- predict(bike_model, newdata = bikesAugust, type = "response")

# Calculate the RMSE
bikesAugust %>% 
  mutate(residual = pred - cnt) %>%
  summarize(rmse  = sqrt(mean(residual^2)))

# Plot predictions vs cnt (pred on x-axis)
ggplot(bikesAugust, aes(x = pred, y = cnt)) +
  geom_point() + 
  geom_abline(color = "darkblue")



# Calculate pseudo-R-squared
(pseudoR2 <- 1 - perf$deviance / perf$null.deviance )


# Ulohou je zobrazit cnt a pred pocas prvych 14 dni v auguste.
# Instant - the time index -- number of hours since beginning of data set 

# Plot predictions and cnt by date/time
bikesAugust %>% 
  # set start to 0, convert unit to days
  mutate(instant = (instant - min(instant))/24) %>%  
  # gather cnt and pred into a value column
  gather(key = valuetype, value = value, cnt, pred) %>%
  filter(instant < 14) %>% # restric to first 14 days
  # plot value by instant
  ggplot(aes(x = instant, y = value, color = valuetype, linetype = valuetype)) + 
  geom_point() + 
  geom_line() + 
  scale_x_continuous("Day", breaks = 0:14, labels = 0:14) + 
  scale_color_brewer(palette = "Dark2") + 
  ggtitle("Predicted August bike rentals, Quasipoisson model")


#####
## Generalized aditive models (GAM modely). GAM() z baliku mgcv

'y ~ b0 + s1(x1) + s2(x2) + ... ' #kde s() su nejake smoothing funkcie transformacne. Konkretne je to skratka od splines

# Priklad, ked sme mali regresiu v ktorej sme si neboli isty ci je lepsie pouzit kvadraticku, alebo kubicku regresiu, tak je mozne pouzit GAM
# Pouzitie pomocou funkcie gam(formula, data, family) , kde family: gaussian(regular), binomial = probabilities, poisson/quasipoisson = cnt

### s() funkcia

# Aby sme specifikovali, ze input/output variable maju nelinearne zavislost medzi sebou, pouzijeme y ~ s(x), t.j. s() funkciu
# s() teba pouzivat so spojitymi premennymi a neodporuca sa pouzit s premennymi, ktore nemaju aspon 10 unique values
# GAM sa vie lahko overfitnut pri malych datasetoch, preto je odporucane mat viac observations pri uceni

# soybean_train is in the workspace
summary(soybean_train)

# Plot weight vs Time (Time on x axis)
ggplot(soybean_train, aes(x = Time, y = weight)) + 
  geom_point()

# Load the package mgcv
library(mgcv)

# Create the formula 
(fmla.gam <- as.formula("weight ~ s(Time)"))

# Fit the GAM Model
model.gam <- gam(fmla.gam, data = soybean_train, family = gaussian)

# Call summary() on model.lin and look for R-squared
summary(model.lin)

# Call summary() on model.gam and look for R-squared
summary(model.gam)

# Call plot() on model.gam
plot(model.gam)




# soybean_test is in the workspace
summary(soybean_test)

# Get predictions from linear model
soybean_test$pred.lin <- predict(model.lin, newdata = soybean_test)

# Get predictions from gam model
soybean_test$pred.gam <- as.numeric(predict(model.gam, newdata = soybean_test))


# Gather the predictions into a "long" dataset
soybean_long <- soybean_test %>%
  gather(key = modeltype, value = pred, pred.lin, pred.gam)

# Calculate the rmse
soybean_long %>%
  mutate(residual = weight - pred) %>%     # residuals
  group_by(modeltype) %>%                  # group by modeltype
  summarize(rmse = sqrt(mean(residual^2))) # calculate the RMSE

# Compare the predictions against actual weights on the test data
soybean_long %>%
  ggplot(aes(x = Time)) +                          # the column for the x axis
  geom_point(aes(y = weight)) +                    # the y-column for the scatterplot
  geom_point(aes(y = pred, color = modeltype)) +   # the y-column for the point-and-line plot
  geom_line(aes(y = pred, color = modeltype, linetype = modeltype)) + # the y-column for the point-and-line plot
  scale_color_brewer(palette = "Dark2")




#####################################
### Chapter 4. Tree based methods ###
#####################################

# bikesJuly is in the workspace
str(bikesJuly)

# Random seed to reproduce results
seed

# The outcome column
(outcome <- "cnt")

# The input variables
(vars <- c("hr", "holiday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"))

# Create the formula string for bikes rented as a function of the inputs
(fmla <- paste(outcome, "~", paste(vars, collapse = " + ")))

# Load the package ranger
library(ranger)

# Fit and print the random forest model
(bike_model_rf <- ranger(fmla, # formula 
                         bikesJuly, # data
                         num.trees = 500, 
                         respect.unordered.factors = "order", 
                         seed = seed))






# bikesAugust is in the workspace
str(bikesAugust)

# bike_model_rf is in the workspace
bike_model_rf

# Make predictions on the August data
bikesAugust$pred <- predict(bike_model_rf, bikesAugust)$predictions

# Calculate the RMSE of the predictions
bikesAugust %>% 
  mutate(residual = cnt - pred)  %>% # calculate the residual
  summarize(rmse  = sqrt(mean(residual^2)))      # calculate rmse

# Plot actual outcome vs predictions (predictions on x-axis)
ggplot(bikesAugust, aes(x = pred, y = cnt)) + 
  geom_point() + 
  geom_abline()





# Print quasipoisson_plot
print(quasipoisson_plot)

# Plot predictions and cnt by date/time
bikesAugust %>% 
  mutate(instant = (instant - min(instant))/24) %>%  # set start to 0, convert unit to days
  gather(key = valuetype, value = value, cnt, pred ) %>%
  filter(instant < 14) %>% # first two weeks
  ggplot(aes(x = instant, y = value, color = valuetype, linetype = valuetype)) + 
  geom_point() + 
  geom_line() + 
  scale_x_continuous("Day", breaks = 0:14, labels = 0:14) + 
  scale_color_brewer(palette = "Dark2") + 
  ggtitle("Predicted August bike rentals, Random Forest plot")


#############
### ONE - HOT - ENCODING
#############

# Dolezita poznamka je, ze nie vsetky programovacie jazyky / baliky vedia pracovat s kategorickymi premennymi.
# Vacsina balikov to automaticky podporuje (v pozdai model.matrix())
# Je potrebne kodovanie na dummy premenne. Typickym prikladom je balik xgboost, ktory nevie pracovat s kategorickymi premennymi

# Balik vtreat ma funkciu designtreatmentZ()
# Idea: 


dframe <- structure(list(color = structure(c(1L, 3L, 3L, 3L, 3L, 1L, 3L, 
                                             2L, 1L, 1L), .Label = c("b", "g", "r"), class = "factor"), 
                                            size = c(13, 11, 15, 14, 13, 11, 9, 12, 7, 12), 
                                            popularity = c(1.07850879384205, 1.39562453143299, 0.921798789640889, 1.20254531921819, 1.08386615267955, 0.804352725623176, 1.10354404407553, 0.874633217928931, 0.694705831585452, 
                                                                                                                                                                   0.883250207640231)), .Names = c("color", "size", "popularity"
                                                                                                                                                                   ), row.names = c(NA, -10L), class = "data.frame")structure(list(color = structure(c(1L, 3L, 3L, 3L, 3L, 1L, 3L,2L, 1L, 1L), .Label = c("b", "g", "r"), class = "factor"), size = c(13, 
                                                                                                                                                                                                                                                                                                                           11, 15, 14, 13, 11, 9, 12, 7, 12), popularity = c(1.07850879384205, 
                                                                                                                                                                                                                                                                                                                                                                             1.39562453143299, 0.921798789640889, 1.20254531921819, 1.08386615267955, 
                                                                                                                                                                                                                                                                                                                                                                             0.804352725623176, 1.10354404407553, 0.874633217928931, 0.694705831585452, 
                                                                                                                                                                                                                                                                                                                                                                             0.883250207640231)), .Names = c("color", "size", "popularity"
                                                                                                                                                                                                                                                                                                                                                                             ), row.names = c(NA, -10L), class = "data.frame")
# dframe is in the workspace
dframe

# Create a vector of variable names
(vars <- c("color", "size"))

# Load the package vtreat
library(vtreat)

# Create the treatment plan
treatplan <- designTreatmentsZ(dframe, vars)

# Examine the scoreFrame - obsahuje info: originalne a nove premenne, spolu s typom (ci je to kategoricka/spojita atd.)
(scoreFrame <- treatplan %>%
    use_series(scoreFrame) %>%
    select(varName, origName, code))

# We only want the rows with codes "clean" or "lev"
(newvars <- scoreFrame %>%
    filter(code %in% c("clean", "lev")) %>%
    use_series(varName))

# Create the treated training data
(dframe.treat <- prepare(treatplan, dframe, varRestriction = newvars))  

# Prepare bere ako prvy parameter object z designTreatmentsZ, potom dataframe na kodovanie a pripadne dalsie podmienky na premenne

# _lev su  kategoricke premenne a _clean su ciste spojite premenne


'When a level of a categorical variable is rare, sometimes it will fail to show up in training data. 
If that rare level then appears in future data, downstream models may not know what to do with it. 
When such novel levels appear, using model.matrix or caret::dummyVars to one-hot-encode will not work correctly.'

'vtreat is a "safer" alternative to model.matrix for one-hot-encoding, because it can manage novel levels safely. 
vtreat also manages missing values in the data (both categorical and continuous).'


# Vyuzitie vtreat encodera. Ak trenujes na datach ktore neobsahuju vsetky mozne kategorie, ktore by sa mohli 
# vyskytnut v buducnosti. Riesenie je vo varRestriction = newvars.
# Nove premenne koduje ako same nuly 



# The outcome column
(outcome <- "cnt")

# The input columns
(vars <- c("hr", "holiday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"))

# Load the package vtreat
library(vtreat)

# Create the treatment plan from bikesJuly (the training data)
treatplan <- designTreatmentsZ(bikesJuly, vars, verbose = FALSE)

# Get the "clean" and "lev" variables from the scoreFrame
(newvars <- treatplan %>%
    use_series(scoreFrame) %>%        
    filter(code %in% c("lev", "clean")) %>%  # get the rows you care about
    use_series("varName"))           # get the varName column

# Prepare the training data
bikesJuly.treat <- prepare(treatplan,bikesJuly,varRestriction = newvars)

# Prepare the test data
bikesAugust.treat <- prepare(treatplan,bikesAugust,varRestriction = newvars)

# Call str() on the treated data
str(bikesJuly.treat)
str(bikesAugust.treat)

#### Teraz su uz data pripravene do podoby, ktore vyzaduje xgboost funkcia. V treated datach sa nenachadza outcome premenna y 


##############
#### Gradient boosting machines
##############


# 1) Fit a shallow tree (plytky strom) T1 to data: M1 = T1
# 2) Fit a tree T2 to the residuals. Find gama such that M2 = M1 + gama*T2 is the best fit to data 
# Regularization : learnning rate n€(0,1). Larger n -> faster learning, smaller n -> less risk of overfit
# 3 ) Repeat (2) until stopping condition met.... Final model M ´= M1 + n * SUM (gama_i * T_1)

## Kedze XGBOOST optimalizuje rezidua na train sete, je velmi lahke overfitnut data.
## Je good practice pozerat sa na cros validovany error a sledovat pocet pouzitych stromov 
## Training error sa stale bude zmensovat, avsak test error sa niekde zasekne po niekolkych stromov --- Danov learning curve

# The July data is in the workspace
ls()

## Load the package xgboost
#install.packages("xgboost")
library(xgboost)

# Run xgb.cv  -- robi aj out-of-sample learning error pri postupnom pridavani stromov
cv <- xgb.cv(data = as.matrix(bikesJuly.treat), 
             label = bikesJuly$cnt,   # vector of outcomes(also numeric)
             nrounds = 100, #the maximum number of rounds (trees to build)
             nfold = 5, #the maximum number of rounds (trees to build)
             objective = "reg:linear", #"reg:linear" for continuous outcomes.
             eta = 0.3, #the learning rate 
             max_depth = 6, #depth of trees
             early_stopping_rounds = 10, #after this many rounds without improvement, stop.
             verbose = 0    # to stay silent
)

# str(cv)

# Get the evaluation log 
elog <- cv$evaluation_log

# Determine and print how many trees minimize training and test error
elog %>% 
  summarize(ntrees.train = which.min(train_rmse_mean),   # find the index of min(train_rmse_mean)
            ntrees.test  = which.min(test_rmse_mean))   # find the index of min(test_rmse_mean)

# ntrees pre test vysiel najlepsie pre 84


# Examine the workspace
ls()

# The number of trees to use, as determined by xgb.cv
ntrees <- 84

# Run xgboost
bike_model_xgb <- xgboost(data = as.matrix(bikesJuly.treat), # training data as matrix
                          label = bikesJuly$cnt,  # column of outcomes
                          nrounds = ntrees,       # number of trees to build
                          objective = "reg:linear", # objective
                          eta = 0.3,
                          depth = 6,
                          verbose = 0  # silent
)

# Make predictions
bikesAugust$pred <- predict(bike_model_xgb, as.matrix(bikesAugust.treat))

# Plot predictions (on x axis) vs actual bike rental count
ggplot(bikesAugust, aes(x = pred, y = cnt)) + 
  geom_point() + 
  geom_abline()

# vyzera to fajn, avsak xgboost vratil aj nechativne predikcie 


### Vypocet RMSE

# bikesAugust is in the workspace
str(bikesAugust)

# Calculate RMSE
bikesAugust %>%
  mutate(residuals = cnt - pred) %>%
  summarize(rmse = sqrt(mean(residuals^2)))

"Even though this gradient boosting made some negative predictions, overall it makes smaller errors than the previous 
two models. Perhaps rounding negative predictions up to zero is a reasonable tradeoff."

# Print quasipoisson_plot
print(quasipoisson_plot)

# Print randomforest_plot
print(randomforest_plot)

# Plot predictions and actual bike rentals as a function of time (days)
bikesAugust %>% 
  mutate(instant = (instant - min(instant))/24) %>%  # set start to 0, convert unit to days
  gather(key = valuetype, value = value, cnt, pred) %>%
  filter(instant < 14) %>% # first two weeks
  ggplot(aes(x = instant, y = value, color = valuetype, linetype = valuetype)) + 
  geom_point() + 
  geom_line() + 
  scale_x_continuous("Day", breaks = 0:14, labels = 0:14) + 
  scale_color_brewer(palette = "Dark2") + 
  ggtitle("Predicted August bike rentals, Gradient Boosting model")



