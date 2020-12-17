import pandas as pd
import numpy as np

import argparse


def clean(VITALS, NUM_UNIQUE_CCS, SUBSET_SIZE, LABEL, ALL_DATA, SAVE_CLEANED):
    """ Preprocessing function for eisenhower data (with vitals or without vitals datasets)
    
    The raw data is `data.csv` and `data_vitals.csv` in the /data directory
    
    Arguments
    ---------
    VITALS: boolean
        A boolean for whether or not to remove all vital-based columns before returning a cleaned dataframe.
    NUM_UNIQUE_CCS : int
        Integer amount of max number of unique Chief Complaints to care about when encoding.
    SUBSET_SIZE : int
        Integer subset length of the total 220,000 to clean
    LABEL : str
        Label for saving as output
    ALL_DATA : boolean
        A boolean for whether or not to work with all of the patients data (~220,000 total)
    SAVE_CLEANED : boolean
        Boolean for whether or not to save the data to `models/$LABEL/data_cleaned_$LABEL.csv`
    
    Returns
    -------
    object(pandas.DataFrame)
        A dataframe that holds the cleaned and processed data.
    
    """
    df1 = pd.read_csv('./data/data_vitals.csv')
    
    if ALL_DATA:
        df = df1.copy()
    else:
        df = df1.sample(frac=SUBSET_SIZE/len(df1))
        
    df.is_copy = None

    df.duplicated().sum()
    df.drop_duplicates(inplace=True)

    # changing those who left "Against Medical Advice" and Expired to be considered admitted
    df.loc[(df.Disposition == 'AMA') | (df.Disposition == 'Expired') | (df.Disposition == 'Send to Specialty Department'), 'Admit'] = 1

    # -------------
    # ---lumps long tail of uncommon chief complaints as 'other' as well as combining the separate columns into one (comma separated strings)---

    ccs = ['Chief_Complaint', 'Chief_Complaint_2']
    # get all values in all chief complaints (including NaNs)
    vals = []
    [vals.extend(df[i].values) for i in ccs]
    values = pd.Series(vals)

    # change the values whose value counts are less than the cutoff to 'OTHER' (assumes that the count of NaNs is greater than the cutoff to work!!!!)
    values_to_keep = list(values.value_counts(normalize=True, dropna=False).index[:NUM_UNIQUE_CCS])
    for cc in ccs:
        df.loc[[i not in values_to_keep for i in df[cc]], cc] =  'OTHER'

    df['CCs'] = df[[i for i in ccs]].apply(lambda x: ','.join(x[x.notnull()]), axis=1)
    # drop the chief_complaint_{1:10} separate columns
    df.drop(ccs, axis='columns', inplace=True)


    print("{}% of patients |now| have 'OTHER' as a chief complaint".format(round(len(df[df['CCs'].str.contains("OTHER")])/len(df)*100)))
    print("There are now {} unique chief complaints".format(len(values_to_keep)))

    #-------------------------------------------------------------------------------------------------------------------------------------------
    # ------ dropping unecessary column, renaming columns, fixing NaNs, fixing dtypes-----------------------------------------------------------

    df.rename(columns={"Temp": "temp", "Pulse": "HR", "Resp": "RR", "SpO2": "O2", "Admit": "admit_binary", "Year": "year", "Month":"month", 
                       "Patient_Sex":"sex", "Age":"age", "Disposition":"disposition",
                       "Age_Group":"age_group", "Ambulance":"ambulance", 
                       "ESI_Level":"ESI_level", "Arrival_Mode(re-coded_into_Ambulance_binary)":"arrival_mode", "Disposition(recoded_into_admit_binary)":"outcome"}, inplace=True)

    # ------- fixing NaNs --------------

    fix_age_group_nans = [(18, 'Pediatric'), (65, 'Adult'), (80, 'Geriatric_65-80')]
    for close_age, group in fix_age_group_nans:
        df.loc[[(np.isclose(i, close_age, atol=1) and i < close_age) for i in df.age], 'age_group'] = group

    # CONSIDER DROPPING when disposition is LWBS before Triage, NaN, 
    df.drop(df[df.disposition.isnull()].index, axis=0, inplace=True)
    df.drop(['disposition'], axis=1, inplace=True)

    # ESI_level NaNs replaced with median value (3)
    # df.loc[df[df.ESI_level.isnull()].index, 'ESI_level'] = df.ESI_level.median() 

    # dropping any NaNs that remain (only taking patients that have every column filled)
    df.dropna(how='any', inplace=True)



    # --- fix dtypes ----
    to_string_cols = ['sex', 'age_group', 'CCs']
    df[to_string_cols] = df[to_string_cols].convert_dtypes()


    DROP_COLUMNS = ['UniqueID']
    df.drop(DROP_COLUMNS, axis=1, inplace=True) 
    # -- BP separation -- 
    df = pd.concat([df.drop(['BP'], axis=1), df['BP'].str.split(pat='/', n=0, expand=True)], 1)
    df.rename(columns={0:'BP_sys', 1:'BP_dia'}, inplace=True)
    df[['BP_sys', 'BP_dia']] = df[['BP_sys', 'BP_dia']].astype('float64')

    # thresholds
    df['temp_thresh'] = ((df['temp'] >= 104) & (df['age_group'] != 'Pediatric')).astype('int')
    df['O2_thresh'] = (df['O2'] < 85).astype('int')
    df['BP_sys_thresh'] = ((df['BP_sys'] < 80) & (df['age_group'] != 'Pediatric')).astype('int')
    df['RR_thresh'] = ((df['RR'] > 40) & (df['age_group'] != 'Pediatric')).astype('int')

    #----------
    # ------ cleaning (feature extraction, one hot encoding of categoricals, thresholding)

    # add number of CCs as a feature
    df['CC_num'] = df['CCs'].apply(lambda x: x.count(",") + 1)
    df.loc[df['CCs'] == '', 'CC_num'] = 0

    # one hot encode categoricals
    categorical = ['age_group', 'sex', 'month', 'year']

    for cat in categorical:
        df = pd.concat([df.drop(cat, axis=1), pd.get_dummies(df[cat], prefix=cat)], 1)

    # encode CCs
    df = pd.concat([df.drop(['CCs'], axis=1), df['CCs'].str.get_dummies(sep=",").add_prefix('CC_')], 1)

    if not VITALS:
        # drop vital columns if we are training a non vital model
        vital_cols = ['temp_thresh', 'RR_thresh', 'BP_sys_thresh', 'temp_thresh', 'O2_thresh', 'BP_sys', 'BP_dia', 'O2', 'RR', 'HR', 'temp']
        df.drop(vital_cols, axis=1, inplace=True)
    
    # drop unnecessary binary onehot encoded
        # arrival_mode only significant if more than binary! 
        # SEX HAS MORE THAN 2 UNIQUE!! so don't drop
    # binary_categoricals = ['sex_Female'] 
    # df = df.drop(binary_categoricals, axis=1)
    #------------------ || SAVING || ------------------
    if SAVE_CLEANED:
        df.to_csv('./models/{}/data_cleaned_{}.csv'.format(LABEL, LABEL), index=False)
        df[0:1].to_csv('./models/{}/sample_cleaned_patient_{}.csv'.format(LABEL, LABEL), index=False)
    return df
            
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--NUM_UNIQUE_CCS", "-nccs", type=int, help="Number of unique CCs to be had when cleaning the data, only relevant if -c=True")
    parser.add_argument("--SUBSET_SIZE", "-ss", type=int, help="How much of the 220,000 patients do you want to work with?")
    parser.add_argument("--VITALS", "-v", type=str2bool, nargs='?', const=True, default=False, help="boolean for whether to work with vitals or not")
    parser.add_argument("--LABEL", "-l", type=str, help="str what to label the saved file with, only relevant if -c=True")
    parser.add_argument("--ALL_DATA", "-ad", type=str2bool, nargs='?', const=True, default=False, help="clean all patients of specificied data, only relevant if clean is true")
    args = parser.parse_args()
    print("ARGUMENTS PASSED: {}".format(args))

    NUM_UNIQUE_CCS = args.NUM_UNIQUE_CCS
    SUBSET_SIZE = args.SUBSET_SIZE
    VITALS = args.VITALS
    LABEL = args.LABEL
    ALL_DATA = args.ALL_DATA
    
    df = clean(VITALS, NUM_UNIQUE_CCS, SUBSET_SIZE, LABEL, ALL_DATA, SAVE_CLEANED=True)
