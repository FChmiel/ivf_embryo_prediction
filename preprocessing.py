"""
Functions used to load, process and clean the HFEA dataset in 
ivf_embryo_prediction

ivf_embryo_prediction, Machine-learnt models for predicting chance
of suitable embryo for D5 transfer or freezing.

© University of Southampton, IT Innovation Centre, 2020

Copyright in this software belongs to University of Southampt
University Road, Southampton, SO17 1BJ, UK.

This software may not be used, sold, licensed, transferred, copied
or reproduced in whole or in part in any manner or form or in or
on any media by any person other than in accordance with the terms
of the Licence Agreement supplied with the software, or otherwise
without the prior written consent of the copyright owners.

This software is distributed WITHOUT ANY WARRANTY, without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE, except where stated in the Licence Agreement supplied with
the software.

Created Date :          24-06-2020

Created for Project :   Fertility predict

Author: Francis P. Chmiel

Email: F.P.Chmiel@soton.ac.uk
"""
import pandas as pd
import numpy as np

def create_cohort(df):
    """
    Creates the patient cohort for modelling. The exclusion criteria is outlined
    in the publication (See https://github.com/FChmiel/ivf_embryo_prediction) 
    supporting this work.
    
    Parameters:
    -----------
    df : pd.DataFrame, 
        The HFEA anonymised register data.
        
    Returns:
    --------
    df : pd.DataFrame, 
        The HFEA anonymised register data, edited in-place with
        excluded patients removed.
    """
    
    # exclude DI cycles
    DI_cycles = df['Type of treatment - IVF or DI']=='DI'
    df = df[~DI_cycles]
    
    # exclude FET cycles (keep only fresh)
    fresh_cycles = df['Fresh Cycle']==1
    df = df[fresh_cycles]
    
    # exclude donated embryo cycles
    donated_cycles = df['Donated embryo']==1
    df = df[~donated_cycles]
    
    # exclude cycles for storing embryos / donation (keep only treatment now)
    treatment_cycles = (df['Main Reason for Producing Embroys Storing Eggs']    
                        =='Treatment Now')
    df = df[treatment_cycles]
    
    # exclude surrogate cycles
    surrogate_cycles = df['Patient acting as Surrogate']==1
    df = df[~surrogate_cycles]
    
    # exclude PGD
    PGD_cycles = df['PGD']==1
    df = df[~PGD_cycles]
    
    # exclude PGS
    PGS_cycles = df['PGS']==1
    df = df[~PGS_cycles]
    
    # exclude cucles with eggs from non-patient source
    patient_cycles = df['Egg Source']=='Patient'
    df = df[patient_cycles]
    
    # exclude frozen cycles 
    frozen_cycles = df['Frozen Cycle']==1
    df = df[~frozen_cycles]
    
    # exclude eggs thawed in cycle
    eggs_thawed_cycles = df['Eggs Thawed']>0
    df = df[~eggs_thawed_cycles]
    
    # exclude embryos thawed in cycle
    embryos_thawed_cycles = df['Total Embryos Thawed']>0
    df = df[~embryos_thawed_cycles]
    return df

def format_columns(df,
                   str_cols=['Main Reason for Producing Embroys Storing Eggs'],
                   num_cols=['Fresh Eggs Stored',
                             'Fresh Eggs Collected',
                             'Total Embryos Thawed', 
                             'Total Embryos Created',
                             'Embryos Stored For Use By Patient']):
    """
    Performs some data cleaning. Including stripping white space from some str 
    columns and converting '> 50' to 50 in numerical columns (this will have 
    some tiny effect on statistics, but there are VERY few instances of '> 50')!
    
    Parameters:
    -----------
    df : pd.DataFrame, 
        The HFEA IVF cycles database.
        
    str_cols : List[str],
        A list of the str columns to strip leading/trailing whitespace from.
        
    num_col : List[str],
        A list of the numerical columns to convert '> 50' to 50 within.
        
    Returns:
    --------
    df : pd.DataFrame, 
        The HFEA database edited in-place.
    """

    # re-name poorly formatted column
    df.rename(columns={'Cause  of Infertility - Tubal disease':
                       'Cause of Infertility - Tubal disease'},
              inplace=True)

    # strip whitespace from string columns
    for col in str_cols:
        df[col] = df[col].str.strip()
        
    # convert >50 to 50.
    for col in num_cols:
        try:
            mask = df[col].str.contains('>')==True
            df.loc[mask, col] = 50
            df[col] = df[col].astype(float)
        except:
            pass
            # already a numerical column
        
    # format patient age = 18-34 to be consitent with others
    mask = df['Patient Age at Treatment']=='18 - 34'
    df.loc[mask, 'Patient Age at Treatment'] = '18-34'
    return df

def create_target(df):
    """
    Creates the target for the HFEA database. The target (1) is whether the
    patient had a day 5 embryo transfer or an embryo frozen. If neither occured
    during the cycle it is labelled as 0.
    
    Parameters:
    ---------
    df : pd.DataFrame, 
        The HFEA IVF cycles database.
        
    Returns:
    --------
    df : pd.DataFrame, 
        The HFEA IVF dataset edited in-place. 
    """
    
    df['target'] = np.nan
    
    # create masks for generating target
    no_embryos_stored = df['Embryos Stored For Use By Patient']==0
    # you can just use ~ here because NaMs are treated as False
    embryos_stored = df['Embryos Stored For Use By Patient']>0 
    no_transfer =  df['Embryos Transfered']==0
    early_transfer = df['Date of Embryo Transfer']<5
    late_transfer = df['Date of Embryo Transfer']>4
    no_embryos_created = df['Total Embryos Created']==0
    embryos_created = df['Total Embryos Created']>0
    
    # create negative class
    df.loc[early_transfer & no_embryos_stored, 'target'] = 0
    df.loc[no_embryos_created, 'target'] = 0
    df.loc[embryos_created & no_transfer & no_embryos_stored, 'target'] = 0
    
    # create positive class
    df.loc[embryos_stored, 'target'] = 1
    df.loc[late_transfer, 'target'] = 1
    return df

def drop_anomalous_cycles(df):
    """
    Removes anomalous cycles in-place. These include cycles where the Day of 
    embryo transfer equals 999.
    
    Parameters:
    -----------
    df : pd.DataFrame, 
        The HFEA IVF cycles database.
    
    Returns:
    --------
    df : pd.DataFrame, 
        The HFEA IVF dataset edited in-place. 
    """
    
    # drop those with ET as 999 and no frozen embryos
    anom_transfer = df['Date of Embryo Transfer']==999
    no_embryos_stored = df['Embryos Stored For Use By Patient']==0
    to_drop = anom_transfer & no_embryos_stored
    df = df[~to_drop]
    
    # Drop those with LB=1 but no embryos transferred
    no_transfer = df['Embryos Transfered']==0
    live_birth = df['Live Birth Occurrence']==1
    to_drop = no_transfer & live_birth
    df = df[~to_drop]
    
    # Drop treatment types which are not IVF or ICSI
    allowed_types = df['Specific treatment type'].isin(['IVF', 'ICSI'])
    df = df[allowed_types]
    return df

def clean_treatment_type(df, allowed_types=['IVF', 'ICSI']):
    """
    Cleans the treatment type column by grouping all types which are neither
    IVF or ICSI into an 'other' column.
    
    Parameters:
    -----------
    df : pd.DataFrame, 
        The HFEA IVF cycles database.
    
    allowed_types : List[str]
        A list of treatment types not to be grouped.
    
    Returns:
    --------
    df : pd.DataFrame, 
        The HFEA IVF dataset edited in-place. 
    """
    allowed_cycles = df['Specific treatment type'].isin(allowed_types)
    df.loc[~allowed_cycles, 'Specific treatment type'] = 'Other/Unknown'
    return df

def encode_columns(df):
    """
    Encodes str columns with nominal / ordinal encoding, defined in the 
    functions by maps.
    
    Parameters:
    -----------
    df : pd.DataFrame, 
        The HFEA IVF cycles database.
    
    Returns:
    --------
    df : pd.DataFrame, 
        The HFEA IVF dataset edited in-place. 
    """
    # encode the age group column
    age_map = {'18-34':0, '35-37':1, '38-39':2, '40-42':3, '43-44':4, '45-50':5}
    df['Patient Age at Treatment'] = df['Patient Age at Treatment'].map(age_map)
    
    # encode the treatment type column
    treatment_map = {'IVF':0, 'ICSI':1}
    df['Specific treatment type'] = (df['Specific treatment type']
                                     .map(treatment_map))
    return df
