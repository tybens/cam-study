# tybens 11/5/2020
import pickle
import pandas as pd
import os
import numpy as np
import argparse  # python command line flags

from itertools import combinations
from sklearn.model_selection import train_test_split

from prepModel import prep
from utils.cleaning import str2bool
from utils.SuperLearner import SuperLearner

def main():
    ALLSCORES = []
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dest_dir = os.path.join(script_dir, 'models', LABEL, 'expanded')
    try:
        os.makedirs(dest_dir)
    except OSError:
        pass # already exists
    
     # initial prep: calibrate the metamodel, clean and save all the data matched to the data the model was trained on, save as a list of models (MasterModelList)
    prep(VITALS=VITALS, LABEL=LABEL, OPTIMIZED=OPTIMIZED, RAND_STATE=RAND_STATE)
    # -- these files are created during the prep() call on line 26 --
    # load optimized models that will be expanded (to encompass all combination of features) 
    filename = './models/{}/MasterModel{}.sav'.format(LABEL, OPTIMIZED)
    superLearner = pickle.load(open(filename, 'rb')) 
   
    # load ALL data that is cleaned to match the features of what the models were trained on 
    if VITALS:
        filename = 'data_vitals_cleaned.csv'
    else:
        filename = 'data_cleaned.csv'
    data_matched = pd.read_csv('./data/'+filename)

    # columns that might be NaNs
    relevant_cols = ['temp', 'HR', 'RR', 'O2', 'BP', 'ambulance', 'age'] if VITALS else  ['ambulance', 'age']
    # to craft an identifier dataframe:
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
            
            X = data_dropped.drop('admit_binary', axis=1)
            y = data_dropped['admit_binary']
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2, shuffle=True, random_state=RAND_STATE)
            
            # fit the data with dropped cols to new models
            superLearner.fit(X_train, y_train)
            scores = superLearner.scores(X_test, y_test)

            ALLSCORES.append(scores)
            # save the models
            filename = './models/{}/expanded/MasterModel{}.sav'.format(LABEL, uid)
            pickle.dump(MasterModel, open(filename, 'wb'))
           
            # calculates and saves all_scores for the given fitted model and unique id
            
            all_scores = superLearner.predict_proba(cleaned_large_data)[:, 1]
            filename_all = './production_data/{}/expanded/all_scores_{}.csv'.format(LABEL, uid)
            all_scores.tofile(filename_all,sep=',',format='%10.5f')
    
            # save info to identify models
            unique_id.append(uid)
            uid+=1
            names.append(name)

    df_id = pd.DataFrame(np.array([unique_id, names]).T, columns=['id', 'name'])
    filename = './models/{}/expanded/df_id.sav'.format(LABEL)
    pickle.dump(df_id, open(filename, 'wb'))
    
    return ALLSCORES

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--LABEL", "-l", type=str, help="str label of which study to load the models from")
    parser.add_argument("--OPTIMIZED", "-o", type=str, help="str how the model was optimized 'OF' for f-score, 'OP' for PRAUC, 'OR' for AUROC")
    parser.add_argument("--RAND_STATE", "-rs", type=int, help="This must be the same as it was for the modelSearch.py. Change the random_state through with np and sklearn work")
    
    args = parser.parse_args()

    RAND_STATE = args.RAND_STATE
    LABEL = args.LABEL
    OPTIMIZED = args.OPTIMIZED
    VITALS = False if 'n' in LABEL else True
    
    ALLSCORES = main()
    
    # save all scores
    columns = ['Model', 'AUROC', 'AUPRC', 'accuracy', 'f1', 'recall']
    pd.DataFrame(ALLSCORES, columns=columns).to_csv('./models/{}/ALLEXPANDEDSCORES.csv'.format(LABEL), header=False, index=False, sep=',')
    
        
    
