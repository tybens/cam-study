# tybens 11/04/20
# python standard library imports
import pickle
import pandas as pd
import numpy as np
import argparse  # python command line flags
import warnings

# third party imports
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# local imports
from utils.cleaning import str2bool, clean_to_match
from utils import calibrateMeta
from utils.SuperLearner import SuperLearner


def prep(VITALS, LABEL, RAND_STATE OPTIMIZED=None):  
    """ Prepping the model to be ready to be used by productionApp
    
    Parameters
    ----------
    VITALS : bool
        boolean for whether vitals are being worked with or not
    LABEL : str
        str what to label the saved file with and how to identify models to prep.
    RAND_STATE : int
        random state for trian_test_split, must be the same as modelSearch was done
    OPTIMIZED : bool, optional
        Default is None. The score metric ('OB', 'OF', 'OP') on which the model was optimized. 
    """
    
    try:
        filename_models = './models/{}/models_{}SL_{}.sav'.format(LABEL, OPTIMIZED, LABEL)
        filename_meta = './models/{}/metamodel_{}SL_{}.sav'.format(LABEL, OPTIMIZED, LABEL)
        mymodels = pickle.load(open(filename_models, 'rb'))
        mymeta_model = pickle.load(open(filename_meta, 'rb'))
    except:
        filename_superlearner = './models/{}/SuperLearner{}SL.sav'.format(LABEL, OPTIMIZED)
        superLearner = pickle.load(open(filename_meta, 'rb'))
            
        
    # load ALL data (cleaned of itself, but not to match what the model was trained on)
    if VITALS:
        filename = 'data_vitals_cleaned.csv'
    else:
        filename = 'data_cleaned.csv'
    cleaned_large_data = pd.read_csv(open('./data/'+filename, 'rb'))
    
    # fit the optimized model to the large cleaned data
    X = cleaned_large_data.drop('admit_binary', axis=1)
    y = cleaned_large_data['admit_binary']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2, shuffle=True, random_state=RAND_STATE)

    # fit and score on large dataset
    superLearner.fit(X_train, y_train)
    score = superLearner.score(X_test, y_test)
    print("{}SL score after refitting on large data: ".format(OPTIMIZED))
    print(score)

    # --- Calibrate Meta (as CalibratedClassifier):
    mymodels, mymeta_model = superLearner.baseModels, superLearner.metaModel
    ccMeta = calibrateMeta(df=cleaned_large_data, mymodels=mymodels, mymeta_model=mymeta_model, plot=False)
    superLearner.metaModel = ccMeta

    # save model as superlearner object
    filename = './models/{}/MasterModel{}.sav'.format(LABEL, OPTIMIZED)
    pickle.dump(superLearner, open(filename, 'wb'))

    # --SAVING (this is for production of the figure!! not really the study)
    all_scores = superLearner.predict_proba(cleaned_large_data)[:, 1]
    filename_all = './production_data/{}/all_scores_{}.csv'.format(LABEL, OPTIMIZED)
    all_scores.tofile(filename_all,sep=',',format='%10.5f')
    
    actual_scores = cleaned_large_data.admit_binary
    filename_actual = './production_data/{}/actual_scores_{}.csv'.format(LABEL, LABEL)
    actual_scores.to_csv(filename_actual, index=False)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--LABEL", "-l", type=str, help="str what to label the saved file with")
    parser.add_argument("--OPTIMIZED", "-o", type=str, help="str how the model was optimized 'OF' for f-score, 'OP' for PRAUC, 'OR' for AUROC")
    parser.add_argument("--RAND_STATE", "-rs", type=int, help="This must be the same as it was for the modelSearch.py. Change the random_state through with np and sklearn work")
    args = parser.parse_args()
    
    RAND_STATE = args.RAND_STATE
    LABEL = args.LABEL
    VITALS = False if 'n' in LABEL else True
    OPTIMIZED = args.OPTIMIZED
    
    prep(VITALS=VITALS, LABEL=LABEL, RAND_STATE=RAND_STATE, OPTIMIZED=OPTIMIZED)
    
