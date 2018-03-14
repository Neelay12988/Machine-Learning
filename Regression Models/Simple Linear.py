# Simple Linear Regression

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# X : Matrix of feature
# y : Dependent variable
dataset = pd.read_csv('Salary Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
# Since the dataset has 30 observations, training the regressor with 20 observations and testing with 10 observations
# assigning random_state : 0 to get consistant split everytime this code is ran, else we will get different split every single time
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
# No need to Feature Scaling as LinearRegression will take care of this

# Fitting Simple Linear Regression to the Training set
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

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()