# tybens 11/04/20
# python standard library imports
import pickle
import pandas as pd
import numpy as np
import argparse  # python command line flags
import warnings

# third party imports
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.utils import compute_class_weight
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# local imports
from cleaning import str2bool
from app.utils import calibrateMeta
from app.utils import model_eval, clean_to_match, differences, superlearner_eval, superlearner_predict, percentileFromScore # evaluations
from app.utils import predPercentageHist, precision_recallPlot, roc_Plot, plot_precision_recall_vs_threshold # personalized figures
from app.utils.patient import Patient  # Patient class


def prep(VITALS, LABEL, OPTIMIZED=None, EXPANDING=False, DATA_MATCH=None, IDX=None, LOAD=False, BETA=3):  
    """ Prepping the model to be ready to be used by productionApp
    
    Parameters
    ----------
    VITALS : bool
        boolean for whether vitals are being worked with or not
    LABEL : str
        str what to label the saved file with and how to identify models to prep.
    OPTIMIZED : bool, optional
        Default is None. The score metric ('OB', 'OF', 'OP') on which the model was optimized. Only relevant when EXPANDING = False. 
    EXPANDING : bool, optional
        Default is False. Whether or not the prepping is being done on an EXPANDINGed model.
    DATA_MATCH : pd.DataFrame, optional
        Default is None. When EXPANDING = True, pass in a dataframe with n_features equal to the model's n_features
    IDX : int, optional
        Deafult is None. A unique ID to be used when EXPANDING = True to identify the expanded models.
    BETA : float, optional
        Default is 3. fbeta_score, beta > 1 favors recall < 1 favors precision
    """
    if not EXPANDING:
        # --- load models and data
        filename_models = './models/{}/models_{}SL_{}.sav'.format(LABEL, OPTIMIZED, LABEL)
        filename_meta = './models/{}/metamodel_{}SL_{}.sav'.format(LABEL, OPTIMIZED, LABEL)
        mymodels = pickle.load(open(filename_models, 'rb'))
        mymeta_model = pickle.load(open(filename_meta, 'rb'))
    else:
        filename_master = './models/{}/expanded/MasterModel{}.sav'.format(LABEL, IDX)
        MasterModelList = pickle.load(open(filename_master, 'rb'))
        
    if VITALS:
        data_clean_match = pd.read_csv('./models/{}/data_vitals_cleaned_{}.csv'.format(LABEL, LABEL))
        raw_data = pd.read_csv('./data/data_vitals.csv')
    else:
        data_clean_match = pd.read_csv('./models/{}/data_cleaned_{}.csv'.format(LABEL, LABEL))
        raw_data = pd.read_csv('./data/data.csv')

        
    if not EXPANDING:
        # ---- evalute originally optimized model:
        if VITALS:
            filename = 'data_vitals_cleaned.csv'
        else:
            filename = 'data_cleaned.csv'

        cleaned_large_data = pd.read_csv(open('./data/'+filename, 'rb'))
        data_matched = clean_to_match(data_clean_match, cleaned_large_data)
        data_matched.to_csv('./models/{}/allDataMatchedto_{}.csv'.format(LABEL, LABEL), index=False)
        SCORES = list() # empty list of scores for to append each eval function

        print("{}SL score:".format(OPTIMIZED))

        # --- Calibrate Meta (as CalibratedClassifier):
        ccMeta = calibrateMeta(df=data_matched, mymodels=mymodels, mymeta_model=mymeta_model, plot=False)
        MasterModelList = [mymodels, ccMeta]

        filename = './models/{}/MasterModel{}.sav'.format(LABEL, OPTIMIZED)
        pickle.dump(MasterModelList, open(filename, 'wb'))
        IDX = OPTIMIZED
    else:
        # MasterModelList is already loaded from expanded
        data_matched = DATA_MATCH

    # --SAVING
    all_scores = superlearner_predict(data_matched, MasterModelList[0], MasterModelList[1], predict_proba=True)[:, 1]
    filename_all = './production_data/{}/all_scores_{}.csv'.format(LABEL, IDX)
    all_scores.tofile(filename_all,sep=',',format='%10.5f')

    if not EXPANDING:
        actual_scores = data_matched.admit_binary
        filename_actual = './production_data/{}/actual_scores_{}.csv'.format(LABEL, LABEL)
        actual_scores.to_csv(filename_actual, index=False)

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--BETA", "-b", type=float, default=3, help="Default is 3. fbeta_score, beta > 1 favors recall < 1 favors precision")
    parser.add_argument("--VITALS", "-v", type=str2bool, nargs='?', const=True, default=False, help="boolean for whether to work with vitals or not")
    parser.add_argument("--LABEL", "-l", type=str, help="str what to label the saved file with")
    parser.add_argument("--OPTIMIZED", "-o", type=str, help="str how the model was optimized 'OB' for fbeta, 'OF' for f-score, 'OP' for PRAUC")
    parser.add_argument("--LOAD", "-lo", type=str2bool, nargs='?', const=True, default=False, help="boolean for whether to load the data or to calculate and save all relevant data. True only functions if this script has already been run with -lo=f before")

    args = parser.parse_args()
    
    BETA = args.BETA
    VITALS = args.VITALS
    LABEL = args.LABEL
    OPTIMIZED = args.OPTIMIZED
    LOAD = args.LOAD
    
    prep(VITALS=VITALS, LABEL=LABEL, OPTIMIZED=OPTIMIZED, EXPANDING=False, DATA_MATCH=None, IDX=None, LOAD=False, BETA=3)
