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
    
    filename_superlearner = './models/{}/SuperLearner{}SL'.format(LABEL, OPTIMIZED)
    filename_allData = './data/data_vitals_cleaned.csv'
    
    superLearner = pickle.load(open(filename_superlearner, 'rb'))
    cleaned_large_data = pd.read_csv(open(filename_allData, 'rb'))
    print("loaded models and cleaned_large_data")
    
    X = cleaned_large_data.drop('admit_binary', axis=1)
    y = cleaned_large_data['admit_binary']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, shuffle=True, random_state=RAND_STATE) 
    
    for subsetProportion in subsets:
        print("performing on subset {}...".format(subsetProportion))
        
        
        scores, _, _, basemodelScores = superlearnerFitAndEval(X_train, X_test, y_train, y_test, 
                                                        baseModels, metaModel, model_name=str(subsetProportion), 
                                                        full_fit=True, optimized=True)
        
        print(basemodelScores) # basemodels also scored on all data
        
        # save all scors
        ALLSCORES.append(scores)
        
    columns = ['Model', 'AUROC', 'AUPRC', 'accuracy', 'f1', 'fbeta', 'recall']
    pd.DataFrame(ALLSCORES, columns=columns).to_csv('./models/{}/testSubsetSize.csv'.format(LABEL), header=False, index=False, sep=',')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--LABEL", "-l", type=str, help="str label from which modelSearch to load models from")
    parser.add_argument("--OPTIMIZED", "-o", type=str, help="str how the model was optimized 'OB' for fbeta, 'OF' for f-score, 'OP' for AUPRC, 'OC' for AUROC\nThis signifies which model will be loaded in")
    parser.add_argument("--RAND_STATE", "-rs", type=int, help="Change the random_state through with np and sklearn work")

    args = parser.parse_args()
    print("ARGUMENTS PASSED: {}".format(args))
    
    LABEL = args.LABEL
    OPTIMIZED = args.OPTIMIZED
    RAND_STATE = args.RAND_STATE
    
    subsets = [0.15, 0.25, 0.4, 0.5, 0.7, 0.9]
    
    main(LABEL, subsets)