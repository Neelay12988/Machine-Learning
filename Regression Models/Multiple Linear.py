# Multiple Linear Regression

# Importing the libraries
import pandas as pd

# Importing the dataset
# X : Matrix of feature
# y : Dependent variable
dataset = pd.read_csv('50 Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# The state column is in test and the mathematical model works only on number
# hence replacing the text states with numerical representation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
# Creating a sparce matrix for the state
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# A dummy variable is created in the sparce matrix hence eliminating the first column
# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
# Since the dataset has 50 observations, training the regressor with 40 observations and testing with 10 observations
# assigning random_state : 0 to get consistant split everytime this code is ran, else we will get different split every single time
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# No need to Feature Scaling as LinearRegression will take care of this

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Checking accuracy of Simple Linear Regression
# Calculating R-square
# Best possible score for R2 is 1.0 
from sklearn.metrics import r2_score
r2_train = r2_score(y_train,regressor.predict(X_train))
r2_test = r2_score(y_test,y_pred)