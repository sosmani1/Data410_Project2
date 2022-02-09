# Advanced Machine Learning Project 2

# importing some relevant libraries
```
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
```

## DEFINITIONS/INFO ABOUT THE LOCALLY WEIGHTED REGRESSION AND RANDOM FOREST REGRESSION

### Locally Weighted Regression: 

Locally weighted regression is a way of estimating a regression surface through a multivariate smoothing procedure, fitting a function of the independent variables locally and in a moving fashion analogous to how a moving average is computed for a time series. With local fitting we can estimate a much wider class of regression surfaces than with the usual classes of parametric functions, such as polynomials.



### Random Forest Regressions:

Random forest is a type of supervised learning algorithm that uses ensemble methods (bagging) to solve both regression and classification problems. The algorithm operates by constructing a multitude of decision trees at training time and outputting the mean/mode of prediction of the individual trees. This is personally one of my favorite types of regressions methods. This is because of the wide application this can be utilized. Personally I am using this method with boosting methods such as **catboost and XgBoost**. 

### Defining the Kernel and Regression functions:

```
# Tricubic Kernel
def tricubic(x):
  return np.where(np.abs(x)>1,0,70/81*(1-np.abs(x)**3)**3)
```
