import pandas as pd
import numpy as np
import seaborn as sns
import env

def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def get_telco_churn_data():
    query = '''
    select *
    from customers
    ;
    '''
    df = pd.read_sql(query, get_db_url('telco_churn'))
    return df