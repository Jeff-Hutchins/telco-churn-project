
Google Slides
https://docs.google.com/presentation/d/1tgAGrn60Rt4xNTUMghQKwC4HhX0Z2QKLxmQ3b3zujcM/edit#slide=id.gc6f980f91_0_5

This project is split between three jupyter notebooks(ipynb) and two .py files

Notebook_telco.ipynb - A walk through of my best model, decistion tree classifier, with unscaled as well as scaled/encoded data.
exploration.ipynb - A walk through of visualizations to help visualize the data.
telco_logistic_regression.ipynb - Logistic Regresson model (didn't perform as well as decision tree)

acquire.py
prepare.py

## About this project:
Using the telco_churn database, an analysis was conducted to gain insight into why customers were churning.  Since the target variable is a categorical variable, multiple classification methods were used to model the data.

The data has 21 columns, including customer ID.

## Data Dictionary:
Original column: internet_service_type_id
Encoded to: DLS, Fiber Optic, None

Original column: contract_type_id
Encoded to: Month-to-Month, One Year, Two Year

Combined: phone_service, multiple_lines
As: has_phone_service

Combined: online_security, online_backup, device_protection, tech_support
As: security_package

## Python Scripts:
### Imports:

import numpy as np
import pandas as pd

import env
import acquire
import prepare

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

### acquire.py

If using a jupyter notebook, use the get_telco_churn_data function below to acquire the telco_churn dataset from through sequel_pro

- acquire.get_telco_churn_data()

### prepare.py

#### prepare.clean_data(df)

    Handle erroneous data and/or outliers by replacing with nans.
    Dropped nans.
    Replaced all 'No', 'Yes' with 0 and 1
    Replaced all 'No phone service' with 0
    Replaced all 'No internet service' with 0

    returns df

#### prepare.combine_and_clean_variables(df)

 Created new feature that represents tenure in years.
    Combines:
        'phone_service' and 'multiple_lines' AS 'has_phone_service'
        'online_security', 'online_backup', 'device_protection', 'tech_support' AS 'security_package'
    Drops:
        All columns combined into new column
        'payment_type_id', 'paperless_billing'
    retruns df

#### prepare.split_data(df)
accepts the dataframe, splits between X and y
and
returns X_train, X_test, y_train, y_test

#### prepare.encode(X_train, X_test, col_name)
accepts X_train, X_text, col_name
and
creates labelEncoder object,
encodes col_name,
creates oneHotEncoder object,
encodes col_name to 0s and 1s,
puts new encoded columns into pandas dataframe,
and joins to existing dataframe.
returns X_train, X_test

#### scale_minmax(X_train, X_test, column_list)
accepts X_train, X_test, column_list,
and
creates MinMaxScaler object,
returns X_train, X_test, scaler



