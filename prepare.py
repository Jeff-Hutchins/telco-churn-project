import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

import env
import acquire

# df = get_telco_churn_data()

def clean_data(df):
    '''
    Handles erroneous data and/or outliers by replacing with nans.
    Dropped nans.
    Replaced all 'No', 'Yes' with 0 and 1
    Replaced all 'No phone service' with 0
    Replaced all 'No internet service' with 0
    '''
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df["total_charges"] = df["total_charges"].astype('float')
    df = df.dropna()
    df.replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)
    df.replace(to_replace=['No phone service'], value=[0], inplace=True)
    df.replace(to_replace=['Female', 'Male'], value=[0, 1], inplace=True)
    df.replace(to_replace=['No internet service'], value=[0], inplace=True)
    return df

def combine_and_clean_variables(df):
    '''
    Created new feature that represents tenure in years.
    Combines:
        'phone_service' and 'multiple_lines' AS 'has_phone_service'
        'online_security', 'online_backup', 'device_protection', 'tech_support' AS 'security_package'
    Drops:
        All columns combined into new column
        'payment_type_id', 'paperless_billing'
    '''
    df['tenure_years'] = round(df.tenure/12, 2)
    df['has_phone_service'] = (df.phone_service == 1) | (df.multiple_lines == 1)
    df = df.drop(columns=['phone_service', 'multiple_lines'])
    df = df.drop(columns=['payment_type_id', 'paperless_billing'])
    df['family'] = (df.partner == 1) | (df.dependents == 1)
    df = df.drop(columns=['partner', 'dependents'])
    df['security_package'] = (df.online_security == 1) | (df.online_backup == 1) | (df.device_protection == 1) | (df.tech_support == 1)
    df = df.drop(columns=['online_security', 'online_backup', 'device_protection', 'tech_support'])
    return df

def split_data(df):
    X = df.drop(columns='churn')
    y = df.churn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 123)
    return X_train, X_test, y_train, y_test

def encode(X_train, X_test, col_name):
    
    encoded_values = sorted(list(X_train[col_name].unique()))

    # Integer Encoding
    int_encoder = LabelEncoder()
    X_train.encoded = int_encoder.fit_transform(X_train[col_name])
    X_test.encoded = int_encoder.transform(X_test[col_name])

    # create 2D np arrays of the encoded variable (in train and test)
    X_train_array = np.array(X_train.encoded).reshape(len(X_train.encoded),1)
    X_test_array = np.array(X_test.encoded).reshape(len(X_test.encoded),1)

    # One Hot Encoding
    ohe = OneHotEncoder(sparse=False, categories='auto')
    X_train_ohe = ohe.fit_transform(X_train_array)
    X_test_ohe = ohe.transform(X_test_array)

    # Turn the array of new values into a data frame with columns names being the values
    # and index matching that of train/test
    # then merge the new dataframe with the existing train/test dataframe
    X_train_encoded = pd.DataFrame(data=X_train_ohe,
                            columns=encoded_values, index=X_train.index)
    X_train = X_train.join(X_train_encoded)

    X_test_encoded = pd.DataFrame(data=X_test_ohe,
                               columns=encoded_values, index=X_test.index)
    X_test = X_test.join(X_test_encoded)

    return X_train, X_test

def scale_minmax(X_train, X_test, column_list):
    scaler = MinMaxScaler()
    column_list_scaled = [col + '_scaled' for col in column_list]
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[column_list]), 
                                columns = column_list_scaled, 
                                index = X_train.index)
    X_train = X_train.join(X_train_scaled)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[column_list]), 
                                columns = column_list_scaled, 
                                index = X_test.index)
    X_test = X_test.join(X_test_scaled)
    return X_train, X_test, scaler




