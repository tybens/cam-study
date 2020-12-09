# tybens 11/5/2020
import pickle
import pandas as pd
import os
import numpy as np
import argparse  # python command line flags

from itertools import combinations

from app.utils import superlearner_eval
from prepModel import prep
from cleaning import str2bool

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dest_dir = os.path.join(script_dir, 'models', LABEL, 'expanded')
    try:
        os.makedirs(dest_dir)
    except OSError:
        pass # already exists

    # load optimized models that will be expanded (to encompass all combination of features) 
    filename = './models/{}/OBSL_base_ccMeta.sav'.format(LABEL)
    MasterModelList = pickle.load(open(filename, 'rb')) 

    data_matched = pd.read_csv('./models/{}/allDataMatchedto_{}.csv'.format(LABEL, LABEL))

    # columns that might be NaNs
    relevant_cols = ['temp', 'HR', 'RR', 'O2', 'BP', 'ambulance', 'age'] if VITALS else  ['ambulance', 'age']
    # to craft a identifier dataframe:
    unique_id = list()
    uid = 0
    names = list()

    # for each possible number of missing NaNs
    for i in range(1, len(relevant_cols)+1):
        combs = combinations(relevant_cols, i)

        # for each combination of this number of missing NaNs
        for comb in combs:
            # split up BP into sys and dia
            cols = [z for z in comb]
            if 'BP' in comb:
                cols.remove('BP')
                cols.extend(['BP_sys', 'BP_dia'])
            if 'age' in comb:
                cols.extend(['age_group_Adult', 'age_group_Geriatric_65-80',
       'age_group_Geriatric_80+', 'age_group_Pediatric'])
            # drop the columns that are NaNs
            data_dropped = data_matched.drop(cols, 1)
            name = ','.join(cols)

            # fit the data with dropped cols to new models
            mymodels, mymeta_model = superlearner_eval(data_dropped,
                                                       MasterModelList[0],
                                                       MasterModelList[1],
                                                       model_name=name,
                                                       full_fit=True)

            # save the models
            filename = './models/{}/expanded/MasterModel{}.sav'.format(LABEL, uid)
            MasterModel = [mymodels, mymeta_model]
            pickle.dump(MasterModel, open(filename, 'wb'))
           
            # calculates and saves all_scores for the given fitted model and unique id
            prep(VITALS=VITALS, LABEL=LABEL, EXPAND=True, DATA_MATCH=data_dropped, IDX=uid)
            
            # save info to identify models
            unique_id.append(uid)
            uid+=1
            names.append(name)

    df_id = pd.DataFrame(np.array([unique_id, names]).T, columns=['id', 'name'])
    filename = './models/{}/expanded/df_id.sav'.format(LABEL)
    pickle.dump(df_id, open(filename, 'wb'))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--LABEL", "-l", type=str, help="str what to label the saved file with")
    args = parser.parse_args()

    LABEL = args.LABEL
    VITALS = False if 'n' in LABEL else True
    
    main()
    
        
    
