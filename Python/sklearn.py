# Inštalácia balíkov do Rodea (v spyderi by to malo byť automaticky)

import pip
pip.main(["install","scipy"])
pip.main(["install","sklearn"])
pip.main(["install","seaborn"])


#intro video (nie cvicenie):
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")   # 1. looks great , 2. R-ggplot

iris = datasets.load_iris()
type(iris)   ## output: sklearn.datasets.base.Buch  .... niečo ako dictionary
print(iris.keys())   ## output: dict_keys(['data','target', 'target_names', 'DESCR', 'data', 'feature_names'])

X = iris.data
y = iris.target
iris.target_names
iris.feature_names

iris.data.shape       # (150, 4)
df = pd.DataFrame(X, columns= iris.feature_names)
_ = pd.scatter_matrix(df, c = y, figsize = [8, 8], s = 150, marker = "D" )  
# c ako color ... je to mapovanie roznej farby v zavislosti od kategorie species (y), s ako shape
    
########
## EDA 
########

# Rýchly pohlad na dáta:
df.head()
df.info()
df.describe()


### COUNT PLOT
# Ako keby histogram podla premennej education sa spocitaju county pre party = {republican, democratic}
# Dá sa povedať, že je to vizualizovaná kontingenčná tabuľka

import seaborn as sns
plt.figure()    # aby sme vymazali predosly plot
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

#####################################
######## Classification  ############
#####################################

#Intro ... nejaky základ z videa, skript viac menej od Import KNeigh... = cvičenie
#.predict()
#from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors = 6)
#knn.fit(iris["data"], iris["target"])

#prediction = knn.predict(X_new)


# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party'].values     # aby sme vytvorili array miesto dataframe pouzijeme .values
X = df.drop('party', axis=1).values   #axis 1/"column" je, že stlpec ma vyhodiť. Defaul je 0/"index" 

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors = 6)  # vytvori objekt, ktory ma k=6

# Fit the classifier to the data
knn.fit(X,y)  # fitne model podla X a y

# Predict the labels for the training data X
y_pred = knn.predict(X)  # predikuje

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))   #toto su len nejake frajeriny na print
#https://pyformat.info/

a = 2
print(f"ahoj{a}")
print("ahoj{}".format(a))

########################
### MNIST dataframe: ručne napísané číslice prevedené do 8x8 mriežky pixelov 
########################

## Kreslenie:

# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())  # komponenty datasetu: dict_keys(["data","target", "DESCR", "target_names", "images")
print(digits.DESCR)   # description

# Print the shape of the images and data keys
print(digits.images.shape)   ### (1794, 8, 8)
print(digits.data.shape)     ### (1794, 64)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

### Klasifikácia:

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data     
y = digits.target     # Note: tento krát má dataset v sebe zabudované takto komponenty. T.j. nemusíme ručne vyberať (X) a (y) z dát  

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)  
# train_test_split funkcia returnuje 4 premenné v hentakom poradí
# test_size je %-podiel v testovacej sade 
# stratify = y,  je to zachovanie pomeru y-kategórii v train/teste

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors = 7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))    # výpočet accurary

###########
### Hladanie optimálneho počtu susedov

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)   # cisla od 1 do 8
train_accuracy = np.empty(len(neighbors))  # empty nedava prazdne vektory, ale nahodne cisla 
test_accuracy = np.empty(len(neighbors))


#A new built-in function, enumerate(), will make certain loops a bit clearer. enumerate(thing), where thing is either an iterator or a sequence, returns a iterator that will return (0, thing[0]), (1, thing[1]), (2, thing[2]), and so forth.

for i, k in enumerate(neighbors):
    print(i)
    print(k)
# i je index = [0,1,2...7], k je cislo z neighbors [1,2,...8]

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors = k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

###################################

import numpy as np
a = np.array([1,2,3,4,5,6])
print(a.reshape(1,6))
print(a.reshape(-1,6))
print(a.reshape(-1,3))   # -1 znamená, že nevieme dimenziu a Python si ju má domyslieť (6/3 v tomto prípade)
#It simply means that it is an unknown dimension and we want numpy to figure it out. 
#And numpy will figure this by looking at the 'length of the array and remaining dimensions' 
#and making sure it satisfies the above mentioned criteria    

###########################
#### LINEAR REGRESSION ####
###########################

#Intro: 

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv("gapminder.csv")

# Create arrays for features and target variable
y = df["life"].values
X = df["fertility"].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))    ## (139,)
print("Dimensions of X before reshaping: {}".format(X.shape))    ## (139,)

# Reshape X and y
y = y.reshape(-1,1)   
X = X.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))    ## (139,1)  urobí z toho vektor
print("Dimensions of X after reshaping: {}".format(X.shape))    ## (139,1)


# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()  # treba ako keby vytvoriť objekt (model) a postupne v nom vytvárať zlozky ako fit, predict atd ..

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)  # linspace je z matlabu... vytvorí sekvenciu od-do s krokom 1 (defaultne)

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)    # toto len na to aby sme vedeli nakreslit regresnu priamku

# Print R^2 
print(reg.score(X_fertility, y ))   # R-squared

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()

## FULL model + RMSE

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))      #rezidua 
print("Root Mean Squared Error: {}".format(rmse))

############################
### CROSS-VALIDÁCIA
###########################

# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X,y, cv = 5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

###################################
##### LASSO A RIDGE regresia
###################################

# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha = 0.4, normalize = True)   ##aby boli rovnakej škály 

# Fit the regressor to the data
lasso.fit(X,y)  # aby boli rovnakej škály

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()

###################################
# RIDGE

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)

# Definicia display_plotu:
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

#####################
#### MODEL PERFORMANCE - CONFUSION-MATRIX
#####################

#Intro:
Precision = TP /(TP + FP)    #  #správne odhalenych defaulte / #všetkych určenych defaultov  (resp. faktor jedničkový)
Recall = TP / (TP + FN)  # SENZITIVITA
F1 SCORE = 2 * (Precision * Recall)/(Precision + Recall)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)        # Musí to byť v tomto poradí
classification_report(y_test, y_pred)   # vypíše precision recall f1-score support

#Príklad
# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors = 6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#######################################
### LOGISTIC REGRESSION + ROC krivka
#######################################

## Rekapitulácia ROC:   x = False positive rate = 1- Specificita = FP / (FP + TN) 
#                       y = True positive rate  = Senzitivity    = TP / (TP + FN)
# ROC je parametrizovaná s cutoff : 1 -> 0

from sklearn.metrics import roc_curve
y_pred_prob = logreg.predict_proba(X_test)[:,1]   # 2 stĺpec je P(c=1) a   1. stĺpec je P(c=0)  \\\\ v R.ku to má tiež predict(type = class/prob..)
fpr, tpr, tresholds = roc_curve(y_test, y_pred_prob)


# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.metrics import roc_curve

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, tresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')  # k ako blacK
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


################################################
###### CROSS VALIDACIA + AUC 

# Import necessary modules  -- treba este dalsie veci ako LogisticRegression atd..
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X,y,cv = 5, scoring = "roc_auc")

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


#####################################
#### HYPER PARAMETER TUNING
#####################################

from sklearn.model_selection import GridSearchCV
param_grid = {"n_neighbors": np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv = 5)
knn_cv.fit(X,y)
knn_cv.best_params_
knn_cv.best_score_


# Import necessary modules
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Vraj logistická regresia má nejaký regularizačný parameter C (nepočul som ešte o tom)


# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid,cv=5)

# Fit it to the data
logreg_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))


#####################################
#### PREPROCESSING
#####################################

# Scikit-learn neberie categorické dáta (factory), ale je treba transformácia na dummy premen--né

#Dummy variable
sklearn : OneHotEncoder()
pandas  : get_dummies()   #toto budeme používať

# Cars dataset má jednu premennú kategorickú: "origin"= {US,ASIA,EUROPE}

#Flow: 
import pandas as pd
df = pr.read_csv("auto.csv")
df_origin = pd.get_dummies(df)   # Vytvorí tri stĺpce: origin_US, origin_Asia, origin_Europe
df_origin = df_origin.drop("origin_Asia", axis = 1) #dropne stĺpec Asia, nakoľko vieme explicitne povedať už z hodnôt 0/1 pre US a EU či to je Ázia

axis=1 # riadok, 
axis=0 # stlpec
##############################

#Cviká:

# Import pandas
import pandas as pd

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv("gapminder.csv")

# Create a boxplot of life expectancy per region
df.boxplot("life", "Region", rot=60) # boxplot life rozbitého podľa Regiónu

# Show the plot
plt.show()

###########

# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first = True)   # vymaže prvú kategóriu 

# Print the new columns of df_region
print(df_region.columns)

###########

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha = 0.5, normalize = True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X,y, cv = 5)

# Print the cross-validated scores
print(ridge_cv)

############
## Imputing + Pipelines
############

# Convert '?' to NaN
df[df == "?"] = np.nan

# Print the number of NaNs
print(df.isnull().sum())   # isnull() fachá aj na NaN... Celkovo to vycisli pocet null pre jednotlivke stlpce

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))


# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC  # support vector classification

# Setup the Imputation transformer: imp
imp = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()  # c-support vector machine

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]

####
# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 42, test_size = 0.3 )

# Fit the pipeline to the train set
pipeline.fit(X_train,y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test,y_pred))

### Scaling

# Import scale
from sklearn.preprocessing import scale

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X))) 
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled))) 
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled))) 


# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train,y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test,y_test)))


#######
## PIPELINY SPOLU 

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline
steps = [("scaler", StandardScaler()),
         ("SVM", SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],    # syntax je: Nazov estimatora v pipelinee + __ + nazov hyperparametra
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 21, test_size = 0.2)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline,parameters,cv=3)

# Fit to the training set
cv.fit(X_train,y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))


#####

# Setup the pipeline steps: steps
steps = [("imputation", Imputer(missing_values="NaN", strategy="mean", axis=0)),
         ("scaler", StandardScaler()),
         ("elasticnet", ElasticNet())]
     #ElasticNet : sklearn.linear_model.coordinate_descent.ElasticNet
# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {"elasticnet__l1_ratio": np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.4, random_state = 42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(estimator = pipeline,param_grid = parameters) # Defaultne su 3 CV

# Fit to the training set
gm_cv.fit(X_train,y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
















