# tybens 11/04/20
# python standard library imports
import numpy as np
import pandas as pd
import pickle
import sys
import os
import argparse  # python command line flags
import multiprocessing as mp # multiprocessing!
import warnings
import copy

# related thirdparty imports
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score, f1_score, fbeta_score, r2_score, recall_score, precision_score, roc_auc_score, average_precision_score
from sklearn.utils import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV
from mlens.ensemble import SuperLearner, Subsemble
import optuna # hyperparam ~unsupervised tuning 
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING) # comment this output optuna progress
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# local relative imports:
from utils.cleaning import clean, str2bool, clean_to_match # clean for processing data, str2bool for command line operability
from utils import superlearnerFitAndEval


def main(LABEL, subsets):
    ALLSCORES = []
    # Load in the SL model (basemodels, metamodel)
    
    filename_models = './models/{}/models_{}SL_{}.sav'.format(LABEL, OPTIMIZED, LABEL)
    filename_meta = './models/{}/metamodel_{}SL_{}.sav'.format(LABEL, OPTIMIZED, LABEL)
    filename_allData = 'data_vitals_cleaned.csv'
    
    baseModels = pickle.load(open(filename_models, 'rb'))
    metaModel = pickle.load(open(filename_meta, 'rb'))
    cleaned_large_data = pd.read_csv(open('./data/'+filename_allData, 'rb'))
    print("loaded models and cleaned_large_data")
    
    for subsetLength in subsets:
        print("performing on subset {}...".format(subsetLength))
        df = clean(VITALS=True, NUM_UNIQUE_CCS=1000, SUBSET_SIZE=subsetLength, LABEL=LABEL, ALL_DATA=False, SAVE_CLEANED=False)
        
        df_all_matched = clean_to_match(df, cleaned_large_data)
        
        # really small test size bc we want it to be true to subsetLength
        X = df.drop('admit_binary', axis=1)
        y = df['admit_binary']
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.001,random_state=0, shuffle=False) 
        
        scores, _, _, basemodelScores = superlearnerFitAndEval(X_train, X_test, y_train, y_test, 
                                                        baseModels, metaModel, df_all_data=df_all_matched, model_name=str(subsetLength), 
                                                        full_fit=True, optimized=True)
        
        print(basemodelScores) # basemodels also scored on all data
        
        # save all scors
        ALLSCORES.append(scores)
        
    columns = ['Model', 'AUROC', 'AUPRC', 'accuracy', 'f1', 'fbeta', 'recall', 'subset size']
    pd.DataFrame(ALLSCORES, columns=columns).to_csv('./models/{}/testSubsetSize.csv'.format(LABEL), header=False, index=False, sep=',')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--LABEL", "-l", type=str, help="str label from which modelSearch to load models from")
    parser.add_argument("--OPTIMIZED", "-o", type=str, help="str how the model was optimized 'OB' for fbeta, 'OF' for f-score, 'OP' for AUPRC, 'OC' for AUROC\nThis signifies which model will be loaded in")
    args = parser.parse_args()
    print("ARGUMENTS PASSED: {}".format(args))
    
    LABEL = args.LABEL
    OPTIMIZED = args.OPTIMIZED
    
    subsets = [10000, 20000, 50000, 70000, 100000, 200000]
    
    main(LABEL, subsets)