# Import relevant libraries and packages.
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns # For all our visualization needs.
import statsmodels.api as sm # The API focuses on models and the most frequently used statistical test, and tools.
from statsmodels.graphics.api import abline_plot # Plots.
from sklearn.metrics import mean_squared_error, r2_score # Calculate MSE and R2 scores.
from sklearn.model_selection import train_test_split #  Splits the dataset into test and train sets.
from sklearn import linear_model, preprocessing #`linear_model` includes a set of methods intended for regression in which the target value is expected to be a linear combination of the features.   `preporcessing` package provides several common utility functions and transformer classes to change raw feature vectors into a representation that is more suitable for the downstream estimators.
import warnings # For handling error messages.
# Don't worry about the following two instructions: they just suppress warnings that could occur later. 
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


# Load the data. 
data = pd.read_csv('wineQualityReds.csv')


# Check out its appearance. 
data.head()


# Another very useful method to call on a recently imported dataset is .info(). Call it here to get a good
# overview of the data
data.info()


# We should also look more closely at the dimensions of the dataset. 
data.shape


# Making a histogram of the quality variable.
data.quality.hist(bins=6)


# Get a basic statistical summary of the variable 
data.quality.describe()

# What do you notice from this summary? 


# Get a list of the values of the quality variable, and the number of occurrences of each. 
data.quality.value_counts()


# Call the .corr() method on the wine dataset 
data.corr()


# Make a pairplot of the wine data
sns.pairplot(data)


# Make a heatmap of the data 
_ _ _



# Plot density against fixed.acidity
_ _ _


# Call the regplot method on your sns object, with parameters: x = 'density', y = 'fixed.acidity'
_ _ _


# Subsetting our data into our dependent and independent variables.
_ _ _

# Split the data. This line uses the sklearn function train_test_split().
# The test_size parameter means we can train with 75% of the data, and test on 25get_ipython().run_line_magic(".", " ")
_ _ _


# We now want to check the shape of the X train, y_train, X_test and y_test to make sure the proportions are right. 
_ _ _


# Create the model: make a variable called rModel, and use it linear_model.LinearRegression appropriately
_ _ _


# We now want to train the model on our test data.
_ _ _


# Evaluate the model  
_ _ _


# Use the model to make predictions about our test data
_ _ _


# Let's plot the predictions against the actual result. Use scatter()
_ _ _


# Create the test and train sets. Here, we do things slightly differently.  
# We make the explanatory variable X as before.
_ _ _

# But here, reassign X the value of adding a constant to it. This is required for Ordinary Least Squares Regression.
# Further explanation of this can be found here: 
# https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.OLS.html
_ _ _


# The rest of the preparation is as before.
_ _ _

# Split the data using train_test_split()
_ _ _


# Create the model
_ _ _

# Fit the model with fit() 
_ _ _


# Evaluate the model with .summary()
_ _ _


# Let's use our new model to make predictions of the dependent variable y. Use predict(), and plug in X_test as the parameter
_ _ _


# Plot the predictions
# Build a scatterplot
_ _ _

# Add a line for perfect correlation. Can you see what this line is doing? Use plot()
_ _ _

# Label it nicely
_ _ _



# Create test and train datasets
# This is again very similar, but now we include more columns in the predictors
# Include all columns from data in the explanatory variables X except fixed.acidity and quality (which was an integer)
_ _ _

# Create constants for X, so the model knows its bounds
_ _ _


# Split the data
_ _ _


# We can use almost identical code to create the third model, because it is the same algorithm, just different inputs
# Create the model
_ _ _

# Fit the model
_ _ _


# Evaluate the model
_ _ _


# Use our new model to make predictions
_ _ _


# Plot the predictions
# Build a scatterplot
_ _ _

# Add a line for perfect correlation
_ _ _

# Label it nicely
_ _ _


# Define a function to check the RMSE. Remember the def keyword needed to make functions? 
_ _ _



# Get predictions from rModel3
_ _ _

# Put the predictions & actual values into a dataframe
_ _ _



# Create test and train datasets
# Include the remaining six columns as predictors
_ _ _

# Create constants for X, so the model knows its bounds
_ _ _

# Split the data

_ _ _


# Create the fifth model
_ _ _
# Fit the model
_ _ _
# Evaluate the model
_ _ _

