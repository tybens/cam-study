# tybens 11/5/2020
import pickle
import pandas as pd
import os
import numpy as np
import argparse  # python command line flags
import multiprocessing as mp # multiprocessing!
import math

from itertools import combinations
from sklearn.model_selection import train_test_split

from utils.cleaning import str2bool
from utils.SuperLearner import SuperLearner


def prep(X, y, X_train, X_test, y_train, y_test, VITALS, LABEL, RAND_STATE, OPTIMIZED=None):  
    """ Prepping the model to be ready to be used by productionApp
    
    Parameters
    ----------
    VITALS : bool
        boolean for whether vitals are being worked with or not
    LABEL : str
        str what to label the saved file with and how to identify models to prep.
    RAND_STATE : int
        random state for train_test_split, must be the same as modelSearch was done
    OPTIMIZED : bool, optional
        Default is None. The score metric ('OB', 'OF', 'OP') on which the model was optimized. 
    """
    filename_superlearner = './models/{}/SuperLearner{}SL.sav'.format(LABEL, OPTIMIZED)
    superLearner = pickle.load(open(filename_superlearner, 'rb'))

    # fit and score on large dataset
    superLearner.fit(X_train, y_train)
    scores = superLearner.scores(X_test, y_test)
    print(scores)
    
    # save model as superlearner object
    filename = './models/{}/MasterModel.sav'.format(LABEL)
    pickle.dump(superLearner, open(filename, 'wb'))

    # --SAVING (this is for production of the figure!! not really the study)
    all_scores = superLearner.predict_proba(X)[:, 1]
    filename_all = './production_data/{}/all_scores.csv'.format(LABEL)
    all_scores.tofile(filename_all,sep=',',format='%10.5f')
    
    # only need this once!
    actual_scores = y
    filename_actual = './production_data/{}/actual_scores_{}.csv'.format(LABEL, LABEL)
    actual_scores.to_csv(filename_actual, index=False)
    
    
def main():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dest_dir = os.path.join(script_dir, 'models', LABEL, 'expanded')
    try:
        os.makedirs(dest_dir)
    except OSError:
        pass # already exists
   
    # load ALL data that is cleaned to match the features of what the models were trained on 
    if VITALS:
        filename = 'data_vitals_cleaned.csv'
    else:
        filename = 'data_cleaned.csv'
    data_matched = pd.read_csv('./data/'+filename)
    X = data_matched.drop('admit_binary', axis=1)
    y = data_matched['admit_binary']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2, shuffle=True, random_state=RAND_STATE)
    
    # only using proportion of total sample size
    train = pd.concat([X_train, y_train], axis=1)
    sampled_train = train.sample(frac=PROP).reset_index(drop=True)
    y_train = sampled_train['admit_binary']
    X_train = sampled_train.drop('admit_binary', axis=1)
    
    # initial prep: clean and save all the data matched to the data the model was trained on, save as MasterModel
    prep(X, y, X_train, X_test, y_train, y_test, VITALS=VITALS, LABEL=LABEL, OPTIMIZED=OPTIMIZED, RAND_STATE=RAND_STATE)
    
    # -- these files are created during the prep() call on line 26 --
    # load optimized models that will be expanded (to encompass all combination of features) 
    filename = './models/{}/MasterModel.sav'.format(LABEL)
    superLearner = pickle.load(open(filename, 'rb')) 
    
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

            name = ','.join(cols)
            
            # save info to identify models
            unique_id.append(uid)
            uid+=1
            names.append(name)

    # identifier dataframe
    df_id = pd.DataFrame(np.array([unique_id, names]).T, columns=['id', 'name'])
    filename = './models/{}/expanded/df_id.sav'.format(LABEL)
    pickle.dump(df_id, open(filename, 'wb')) 
    
    # split the jobs between 7 cpus
    num_cpus = 7
    total_expands = len(names)
    
    list_of_names = [zip(unique_id[i:i+int(math.ceil(total_expands/num_cpus))],names[i:i+int(math.ceil(total_expands/num_cpus))])  for i in range(0, total_expands, int(math.ceil(total_expands/num_cpus)))]
    
    # multithreading saving scores
    score_queue = mp.Queue()
    # initialize threads
    workers = [ mp.Process(target=dropColsAndFitAndSave, args=(names, X_train, X_test, y_train, y_test, superLearner, score_queue,) ) for names in list_of_names]
    # SPIN UP A COUPLE THREADS heh
    [work.start() for work in workers]
    [work.join() for work in workers]
    
    ALLSCORES = []
    for _ in range(len(workers)):
        ALLSCORES.extend(score_queue.get())
    
    return ALLSCORES


def dropColsAndFitAndSave(names, X_train, X_test, y_train, y_test, superLearner, score_queue):
    """ To be used as the multithreaded job for speeding up expanding models """
    
    for (uid, name) in names:
        
        cols = name.split(',')
        X_train, X_test = X_train.drop(cols, 1), X_test.drop(cols, 1)
        # fit the data with dropped cols to new models
        superLearner.model_name = name
        superLearner.fit(X_train, y_train)
        scores = superLearner.scores(X_test, y_test)

        score_queue.put(scores)

        # save the models
        filename = './models/{}/expanded/MasterModel{}.sav'.format(LABEL, uid)
        pickle.dump(superLearner, open(filename, 'wb'))

        # calculates and saves all_scores for the given fitted model and unique id
        X_dropped_cols = pd.concat([X_train, X_test], axis=0)
        all_scores = superLearner.predict_proba(X_dropped_cols)[:, 1]
        filename_all = './production_data/{}/expanded/all_scores_{}.csv'.format(LABEL, uid)
        all_scores.tofile(filename_all,sep=',',format='%10.5f')
    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--LABEL", "-l", type=str, help="str label of which study to load the models from")
    parser.add_argument("--OPTIMIZED", "-o", type=str, help="str how the model was optimized 'OF' for f-score, 'OP' for PRAUC, 'OR' for AUROC")
    parser.add_argument("--RAND_STATE", "-rs", type=int, help="This must be the same as it was for the modelSearch.py. Change the random_state through with np and sklearn work")
    parser.add_argument("--PROPORTION", "-p", type=float, help="The proportion of the training data to be used during fitting, float between 0 and 1")

    args = parser.parse_args()

    RAND_STATE = args.RAND_STATE
    LABEL = args.LABEL
    OPTIMIZED = args.OPTIMIZED
    PROP = args.PROPORTION
    VITALS = False if 'n' in LABEL else True
    
    ALLSCORES = main()
    
    # save all scores
    columns = ['Model', 'AUROC', 'AUPRC']
    pd.DataFrame(ALLSCORES, columns=columns).to_csv('./models/{}/ALLEXPANDEDSCORES.csv'.format(LABEL), header=False, index=False, sep=',')
    
        
    
