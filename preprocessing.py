"""
"""
import pandas as pd
import numpy as np
from utils import load_df

def create_reports_df(fname='copd_{}_data.csv', password=''):
    """
    Creates the un-processed dataframe for performing modelling.
    
    Parameters:
    -----------
    fname,  str
        The filename format for the files on sharepoint.
    
    Returns:
    --------
    """
    
    # names of file to load and merge
    filenames = ['symptom_score', 'cat_score', 'patient', 
                 'activated_user', 'registered_user']
    dfs = []
    for name in filenames: 
        # load the filename
        dfs.append(load_df(fname.format(name), password=password))
        
    # merge cat and symptom score dfs to create report df
    report_df = pd.merge(dfs[0], dfs[1], 
                         left_on=['patient_id','date_on_records'], 
                         right_on=['patient_id','date'], 
                         how='left')
    # drop date column
    report_df.drop('date', inplace=True, axis=1)
    # rename some columns
    report_df.rename(columns={'date_on_records':'date', 'score':'cat_score'}, inplace=True)
    
    # merge patient info, activation and registration dates with a many-to-one merge
    for df in dfs[2:]:
        report_df = pd.merge(report_df, df, 
                             left_on='patient_id', 
                             right_on='patient_id', 
                             how='left', 
                             validate='many_to_one')
        
    # add exacerbation history to data
    #ex_df = load_df(fname.format('exacerbation'), password=password)
    #report_df = pd.merge(report_df, ex_df,
    #                     left_on=['patient_id', 'date'],
    #                     right_on=['patient_id','Data Entered Timestamp'],
    #                     how='left',
    #                     validate='one_to_one')
    
    # sort the dates in ascending order, essential for future operations
    report_df.sort_values(by='date', ascending=True, inplace=True)
    
    # forward fill impute the information
    #col_names = ['Hospital Last Year Times', 'Hospital Last Year']
    #for col in col_names:
    #    report_df[col] = report_df.groupby('patient_id')[col].ffill()
    

    return report_df

def make_target(df, pred_window):
    """
    Creates the target (exacerbation event in pred_window days from current 
    record) for each individual record in df. An exacerbation event is defined 
    as a self-reported symptom score of 3 or 4.

    Parameters:
    ----------
    df, pd.DataFrame
        Record of patient visits to add the target too.

    pred_window, int
        Number of days in future to check for exacerbation event.

    Returns:
    --------
    df, pd.DataFrame
        DataFrame edited in-place with the addition of the target column.
    """

    df['target'] = 0
    df['exacerbation'] = df['symptom_score'].astype(int)>2

    unique_users = np.unique(df['patient_id'])
    # brute force creation of target
    for user in unique_users:
        p_df = df.loc[df['patient_id']==user] # make patient df
    
        # find dates of exacerbation
        exacerbation_dates = p_df.loc[p_df['exacerbation']==True, 'date'].values
 
        # if no exacerbations leave target as 0
        if len(exacerbation_dates)>0:
            for row in p_df.iterrows():
                differences = (row[1]['date'] 
                               - exacerbation_dates)
                differences = np.array([x.days for x in differences])
                mag_mask = np.abs(differences)< pred_window+1
                neg_mask = differences<0
                condition = len(differences[neg_mask & mag_mask])>0
                if condition:
                    df.loc[row[0], 'target'] = 1

    return df

def create_trial_data(df, pat_holdout=0.2, test_duration=2,
                      train_start=pd.to_datetime('01-01-2017'),
                      test_end=pd.to_datetime('03-01-2020'),
                      exclude_size=14, seed=None):
    """
    Creates the training, validation and test sets for testing ML models on 
    the BML myCOPD data.

    Parameters:
    -----------
    df, pd.DataFrame
        The merged DataFrame containing the training data.

    pat_holdout, float
        Fraction of the number of patients to use in the patient hold-out
        test set.

    test_duration,  int
        The size, in months, of the temporal-hold out test set. A month is
        assumed to be 30 days in length.
    
    train_start, pd.DateTime
        The first day to use in the training data. Data before this data is 
        discarded.
    
    test_end, pd.DateTime
        The last date to be used in the test data.
    
    exclude_size, int
        Number of days to remove from the data between the training and temporal
        test set.

    seed, int
        Random seed to be used in sampling. If None default seed is used.

    Returns:
    --------
    X, pd.DataFrame
        The training dataset.

    Xpt, pd.DataFrame
        The patient-wise hold-out test set.

    Xtt, pd.DataFrame
        The temporal hold-out test set. It covers the last test_duration months.
    """
    if seed is not None:
        np.random.seed(seed)
        
    if test_end is not None:
        print('Removing dates coinciding with Covid-19')
        df = df[df.date<test_end]

    # create the temporal test set
    last_date = df.date.values[-1]
    test_start_date = last_date - pd.Timedelta(test_duration*30, unit='days')
    tt_mask = df.date > test_start_date
    Xtt = df[tt_mask]

    # remove test set and excluded data
    train_end_date = (test_start_date - pd.Timedelta(exclude_size, unit='days'))
    df = df[df.date<train_end_date]

    # get hold-out patient ids
    unq_patients = np.unique(df['patient_id'])
    holdout_size = int(len(unq_patients)*pat_holdout)
    test_patients = np.random.choice(unq_patients, size=holdout_size,
                                     replace=False)

    # create the patient holdout test set
    Xpt = df[df.patient_id.isin(test_patients)]

    # create the training set
    X = df[~df.patient_id.isin(test_patients)]
    X = X[X['date']>=train_start]
    Xpt = Xpt[Xpt['date']>=train_start]
    return X, Xpt, Xtt

def drop_bad_reports(df, x=3):
    """Drops reports not wanted in the training or test sets. These include,
    users who have anomalous registration / activation dates, users who report
    just exacerbation events, and reports where there were no reports within x 
    days.
    
    Parameters:
    -----------
    df : pd.DataFrame 
        The report dataframe containing all possible reports. Must contain 
        columns
        named 'patient_id' and 'date'
    
    x : int, 
        The number of days to reject a report if there are no other reports 
        after x days.
    
    Returns:
    --------
    df : pd.DataFrame, 
        The report dataframe edited in-place.
    """
    
    # To do: check the dates are in ascending order
    
    # drop reports with no report in three days
    print(f'Starting reports: {df.shape[0]}, removing isolated reports')
    next_report_date = (df.groupby('patient_id')['date'].shift(-1) 
                        - df['date']).dt.days
    df = df[next_report_date<x+1]
    print(f'Reports left: {df.shape[0]}')
    
    # drop users who report just either 3 or 4s
    print(f'Starting reports: {df.shape[0]}, removing exacerbating only users')
    df['exacerbation'] = df['symptom_score']>2
    bad_patients = df.groupby('patient_id')['exacerbation'].all()
    bad_patients = bad_patients[bad_patients==True].index
    df = df.set_index('patient_id').drop(bad_patients).reset_index()
    print(f'Reports left: {df.shape[0]}')
    
    # drop users with anomalous regisitration / activation dates (reports made 
    # before registration)
    print(f'Starting reports: {df.shape[0]}, removing anomalous users')
    df = df[(df['date']-df['Registered']).dt.days>0]
    df = df[(df['date']-df['Activated']).dt.days>0]
    print(f'Reports left: {df.shape[0]}')
    
    return df

def create_feature_set(df):
    """
    Takes the merged dataset and creates the model feature set.

    Notable excluded features include:
    - post code (my initial models did not find it useful)

    Parameters:
    ----------
    df, pd.DataFrame
        Pandas dataframe which is the outer join of the copd, sympytom and CAT
        score dataframes. 

    Returns:
    -------
    df, pd.DataFrame, 
        DataFrame, edited in-place with generated features added.
    """
    df['date'] = pd.to_datetime(df['date'])

    # post code isn't included as my models didn't use it
    # make if user reported previous exacerbation event
    fname = 'exacerbation_history'
    if not df.columns.isin(['exacerbation']).any():
        df['exacerbation'] = df['symptom_score']>2
    df[fname] = df['exacerbation']
    df.loc[df[fname]==False, fname] = np.nan
    df.loc[:,fname] = df.groupby('patient_id')[fname].ffill()
    df.loc[df[fname].isna(), fname] = 0
    
    # forward-fill cat scores
    
    df['cat_score'] = df.groupby('patient_id')['cat_score'].ffill()

    # was the previous nth event an exacerbation
    #df.loc[:,'last_ex_1'] = df.groupby('anonymkey')['exacerbation'].shift(1)
    #df.loc[:,'last_ex_2'] = df.groupby('anonymkey')['exacerbation'].shift(2)

    # time since nth last user report
    for n in [1,2]:
        df.loc[:, f'time_from_last_{n}'] = (df.loc[:,'date']
                                            - df.groupby('patient_id')['date']
                                              .shift(n)).dt.days

    # previous nth symptom score of user
    for n in [1,2,3,4]:
        fname = 'last_symp_{}'.format(n)
        df.loc[:, fname] = df.groupby('patient_id')['symptom_score'].shift(n)

    # previous nth cat score of user
    for n in [1,2,3,4]:
        fname = 'last_cat_{}'.format(n)
        df.loc[:, fname] = df.groupby('patient_id')['cat_score'].shift(n)

    # add gender
    valid_gender = (df.loc[:,'gender'].astype(str)
                      .str.lower().str.contains('male|female')==True)
    df.loc[:,'gender'] = df.loc[:,'gender'].astype(str).str.lower()
    df.loc[~valid_gender,'gender'] = np.nan
    df.loc[:,'gender'] = df.loc[:,'gender'].map({'male':1, 'female':0})


    # create 7 day rolling features
    agg_symp_7d = (df.set_index('date').groupby('patient_id')
                  .rolling('7D')['symptom_score'].agg(['mean','max','min']))
    
    agg_ex_7d = (df.set_index('date').groupby('patient_id')
                  .rolling('7D')['exacerbation'].agg(['sum']))

    agg_cat_7d = (df.set_index('date').groupby('patient_id')
                 .rolling('7D')['cat_score'].agg(['mean','max','min']))
    
    # create 30 day rolling features
    agg_symp_30d = (df.set_index('date').groupby('patient_id')
                  .rolling('30D')['symptom_score'].agg(['mean']))

    agg_cat_30d = (df.set_index('date').groupby('patient_id')
                 .rolling('30D')['cat_score'].agg(['mean']))
    
    agg_ex_30d = (df.set_index('date').groupby('patient_id')
                  .rolling('30D')['exacerbation'].agg(['sum']))
    
    df.set_index(['patient_id','date'], inplace=True)
    for agg in ['mean','max','min']:
        df['symp_7D_{}'.format(agg)] = agg_symp_7d[agg]
        df['cat_7D_{}'.format(agg)] = agg_cat_7d[agg]
    df['ex_count_7d'] = agg_ex_7d['sum']
    df['ex_count_30d'] = agg_ex_30d['sum']
    df['symp_30D_{}'.format('mean')] = agg_symp_30d['mean']
    df['cat_30D_{}'.format('mean')] = agg_cat_30d['mean']
    
    df['symp_ratio'] = df['symp_7D_mean'] / df['symp_30D_mean']
    df['cat_ratio'] = df['cat_7D_mean'] / df['cat_30D_mean']

    df.reset_index(inplace=True)

    return df

def add_pollution():
    pass