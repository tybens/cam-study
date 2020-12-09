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
from cleaning import clean, str2bool


def confMat():
    print(confusion_matrix(y_test, preds))
    

def evalModel(m):
    global y_test, preds
    print(m + ' accuracy: ', accuracy_score(y_test, preds))
    print(m + ' f1-score: ', f1_score(y_test, preds))
    print(m + ' fbeta-score: ', fbeta_score(y_test, preds, beta=BETA))
    print(m + ' recall: ', recall_score(y_test, preds))




class Optimizer:
    def __init__(self, metric, trials=50):
        self.metric = metric
        self.trials = trials
        self.sampler = TPESampler(seed=0)
        
    def objective(self, trial):
        model = create_model(trial)
        model.fit(X_train, y_train)
        
        if self.metric == 'acc':
            preds = model.predict(X_test)
            return accuracy_score(y_test, preds)
        elif self.metric == 'recall':
            preds = model.predict(X_test)
            return recall_score(y_test, preds)
        elif self.metric == 'f1':
            preds = model.predict(X_test)
            return f1_score(y_test, preds)
        elif self.metric == 'beta':
            preds = model.predict(X_test)
            return fbeta_score(y_test, preds, beta=BETA)
        else:
            if hasattr(model, 'decision_function'):
                y_score = model.decision_function(X_test)
            elif hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:, 1]
            return average_precision_score(y_test, y_score)
                
            
            
    def optimize(self):
        study = optuna.create_study(direction="maximize", sampler=self.sampler)
        study.optimize(self.objective, n_trials=self.trials)
        return study.best_params
    

print('---------------------------------------------------')

def RF(out_queue):
    retdict = {}
    global create_model, y_test, preds
    
    if 'RF' in TOTEST:
        m = 'RF'
        rf = RandomForestClassifier(random_state=RAND_STATE)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        evalModel(m)
        confMat()
    def create_model(trial):
        max_depth = trial.suggest_int("max_depth", 2, 32)
        n_estimators = trial.suggest_int("n_estimators", 2, 500)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        weights = weight_dict
        model = RandomForestClassifier(min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, max_depth=max_depth, random_state=RAND_STATE, class_weight=weights)
        return model

    if 'OARF' in TOTEST:
        m = 'OARF'
        optimizer = Optimizer('acc')
        rf_acc_params = optimizer.optimize()
        rf_acc_params['random_state'] = RAND_STATE
        rf_acc = RandomForestClassifier(**rf_acc_params)
        rf_acc.fit(X_train, y_train)
        preds = rf_acc.predict(X_test)
        print('Optimized on accuracy')
        evalModel(m)
        confMat()
        retdict.update({m: RandomForestClassifier(**rf_acc_params)})

    if 'OFRF' in TOTEST:
        m = 'OFRF'
        optimizer = Optimizer('f1')
        rf_f1_params = optimizer.optimize()
        rf_f1_params['random_state'] = RAND_STATE
        rf_f1 = RandomForestClassifier(**rf_f1_params)
        rf_f1.fit(X_train, y_train)
        preds = rf_f1.predict(X_test)
        print('Optimized on F1 score')
        evalModel(m)
        confMat()
        retdict.update({m: RandomForestClassifier(**rf_f1_params)})

    if 'OBRF' in TOTEST:
        m = 'OBRF'
        optimizer = Optimizer('beta')
        rf_beta_params = optimizer.optimize()
        rf_beta_params['random_state'] = RAND_STATE
        rf_beta = RandomForestClassifier(**rf_beta_params)
        rf_beta.fit(X_train, y_train)
        preds = rf_beta.predict(X_test)
        print('Optimized on fbeta')
        evalModel(m)
        confMat()
        retdict.update({m: RandomForestClassifier(**rf_beta_params)})

    if 'ORRF' in TOTEST:
        m = 'ORRF'
        optimizer = Optimizer('recall')
        rf_rec_params = optimizer.optimize()
        rf_rec_params['random_state'] = RAND_STATE
        rf_rec = RandomForestClassifier(**rf_rec_params)
        rf_rec.fit(X_train, y_train)
        preds = rf_rec.predict(X_test)
        print('Optimized on recall')
        evalModel(m)
        confMat()
        retdict.update({m: RandomForestClassifier(**rf_rec_params)})

    if 'OPRF' in TOTEST:
        m = 'OPRF'
        optimizer = Optimizer('pr')
        rf_pr_params = optimizer.optimize()
        rf_pr_params['random_state'] = RAND_STATE
        rf_pr = RandomForestClassifier(**rf_pr_params)
        rf_pr.fit(X_train, y_train)
        preds = rf_pr.predict(X_test)
        print('optimized on PR AUC')
        evalModel(m)
        confMat()
        retdict.update({m: RandomForestClassifier(**rf_pr_params)})
    print('---------------------------------------------------')
    out_queue.put(retdict)

def XGB(out_queue):
    retdict = {}
    global create_model, y_test, preds
    if 'XGB' in TOTEST:
        m = 'XGB'
        xgb = XGBClassifier(random_state=RAND_STATE)
        xgb.fit(X_train, y_train)
        preds = xgb.predict(X_test)
        evalModel(m)
        confMat()

    def create_model(trial):
        max_depth = trial.suggest_int("max_depth", 2, 30)
        n_estimators = trial.suggest_int("n_estimators", 1, 500)
        learning_rate = trial.suggest_uniform('learning_rate', 0.0000001, 1)
        min_child_weight = trial.suggest_uniform('min_child_weight', 0.4, 0.8)
        reg_lambda = trial.suggest_int('reg_lambda', 6, 10)
        gamma = trial.suggest_uniform('gamma', 0.0000001, 1)
        model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, min_child_weight=min_child_weight, reg_lambda=reg_lambda, max_depth=max_depth, gamma=gamma, random_state=RAND_STATE, scale_pos_weight=scale_pos_weight)
        return model

    if 'OAXGB' in TOTEST:
        m = 'OAXGB'
        optimizer = Optimizer('acc')
        xgb_acc_params = optimizer.optimize()
        xgb_acc_params['random_state'] = RAND_STATE
        xgb_acc = XGBClassifier(**xgb_acc_params)
        xgb_acc.fit(X_train, y_train)
        preds = xgb_acc.predict(X_test)
        print('Optimized on accuracy')
        evalModel(m)
        confMat()
        retdict.update({m: XGBClassifier(**xgb_acc_params)})

    if 'OFXGB' in TOTEST:
        m = 'OFXGB'
        optimizer = Optimizer('f1')
        xgb_f1_params = optimizer.optimize()
        xgb_f1_params['random_state'] = RAND_STATE
        xgb_f1 = XGBClassifier(**xgb_f1_params)
        xgb_f1.fit(X_train, y_train)
        preds = xgb_f1.predict(X_test)
        print('Optimized on F1 score')
        evalModel(m)
        confMat()
        retdict.update({m: XGBClassifier(**xgb_f1_params)})

    if 'OBXGB' in TOTEST:
        m = 'OBXGB'
        optimizer = Optimizer('beta')
        xgb_beta_params = optimizer.optimize()
        xgb_beta_params['random_state'] = RAND_STATE
        xgb_beta = XGBClassifier(**xgb_beta_params)
        xgb_beta.fit(X_train, y_train)
        preds = xgb_beta.predict(X_test)
        print('Optimized on fbeta')
        evalModel(m)
        confMat()
        retdict.update({m: XGBClassifier(**xgb_beta_params)})

    if 'ORXGB' in TOTEST:
        m = 'ORXGB'
        optimizer = Optimizer('recall')
        xgb_rec_params = optimizer.optimize()
        xgb_rec_params['random_state'] = RAND_STATE
        xgb_rec = XGBClassifier(**xgb_rec_params)
        xgb_rec.fit(X_train, y_train)
        preds = xgb_rec.predict(X_test)
        print('Optimized on recall')
        evalModel(m)
        confMat()
        retdict.update({m: XGBClassifier(**xgb_rec_params)})

    if 'OPXGB' in TOTEST:
        m = 'OPXGB'
        optimizer = Optimizer('pr')
        xgb_pr_params = optimizer.optimize()
        xgb_pr_params['random_state'] = RAND_STATE
        xgb_pr = XGBClassifier(**xgb_pr_params)
        xgb_pr.fit(X_train, y_train)
        preds = xgb_pr.predict(X_test)
        print('optimized on PR AUC')
        evalModel(m)
        confMat()
        retdict.update({m: XGBClassifier(**xgb_pr_params)})
        print('---------------------------------------------------')
    out_queue.put(retdict)

def LGBM(out_queue):
    retdict = {}
    global create_model, y_test, preds
    if 'LGBM' in TOTEST:
        m = 'LGBM'
        lgb = LGBMClassifier(random_state=RAND_STATE)
        lgb.fit(X_train, y_train)
        preds = lgb.predict(X_test)
        evalModel(m)
        confMat()


    def create_model(trial):
        max_depth = trial.suggest_int("max_depth", 2, 30)
        n_estimators = trial.suggest_int("n_estimators", 1, 500)
        learning_rate = trial.suggest_uniform('learning_rate', 0.0000001, 1)
        num_leaves = trial.suggest_int("num_leaves", 2, 5000)
        min_child_samples = trial.suggest_int('min_child_samples', 3, 200)
        model = LGBMClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, num_leaves=num_leaves, min_child_samples=min_child_samples,
                               random_state=RAND_STATE, is_unbalance=False)
        return model

    if 'OALGBM' in TOTEST:
        m = 'OALGBM'
        optimizer = Optimizer('acc')
        lgb_acc_params = optimizer.optimize()
        lgb_acc_params['random_state'] = RAND_STATE
        lgb_acc = LGBMClassifier(**lgb_acc_params)
        lgb_acc.fit(X_train, y_train)
        preds = lgb_acc.predict(X_test)
        print('Optimized on accuracy')
        evalModel(m)
        confMat()
        retdict.update({m:LGBMClassifier(**lgb_acc_params)})

    if 'OFLGBM' in TOTEST:
        m = 'OFLGBM'
        optimizer = Optimizer('f1')
        lgb_f1_params = optimizer.optimize()
        lgb_f1_params['random_state'] = RAND_STATE
        lgb_f1 = LGBMClassifier(**lgb_f1_params)
        lgb_f1.fit(X_train, y_train)
        preds = lgb_f1.predict(X_test)
        print('Optimized on F1-score')
        evalModel(m)
        confMat()
        retdict.update({m:LGBMClassifier(**lgb_f1_params)})

    if 'OBLGBM' in TOTEST:
        m = 'OBLGBM'
        optimizer = Optimizer('beta')
        lgb_beta_params = optimizer.optimize()
        lgb_beta_params['random_state'] = RAND_STATE
        lgb_beta = LGBMClassifier(**lgb_beta_params)
        lgb_beta.fit(X_train, y_train)
        preds = lgb_beta.predict(X_test)
        print('Optimized on fbeta')
        evalModel(m)
        confMat()
        retdict.update({m:LGBMClassifier(**lgb_beta_params)})

    if 'ORLGBM' in TOTEST:
        m = 'ORLGBM'
        optimizer = Optimizer('recall')
        lgb_rec_params = optimizer.optimize()
        lgb_rec_params['random_state'] = RAND_STATE
        lgb_rec = LGBMClassifier(**lgb_rec_params)
        lgb_rec.fit(X_train, y_train)
        preds = lgb_rec.predict(X_test)
        print('Optimized on recall')
        evalModel(m)
        confMat()
        retdict.update({m:LGBMClassifier(**lgb_rec_params)})

    if 'OPLGBM' in TOTEST:
        m = 'OPLGBM'
        optimizer = Optimizer('pr')
        lgb_pr_params = optimizer.optimize()
        lgb_pr_params['random_state'] = RAND_STATE
        lgb_pr = LGBMClassifier(**lgb_pr_params)
        lgb_pr.fit(X_train, y_train)
        preds = lgb_pr.predict(X_test)
        print('optimized on PR AUC')
        evalModel(m)
        confMat()
        retdict.update({m:LGBMClassifier(**lgb_pr_params)})
    print('---------------------------------------------------')
    out_queue.put(retdict)

def LR(out_queue):
    retdict = {}
    global create_model, y_test, preds
    if 'LR' in TOTEST:
        m = 'LR'
        lr = LogisticRegression(random_state=RAND_STATE)
        lr.fit(X_train, y_train)
        preds = lr.predict(X_test)
        evalModel(m)
        confMat()
    print('---------------------------------------------------')


def DT(out_queue):
    retdict = {}
    global create_model, y_test, preds
    if 'DT' in TOTEST:
        m = 'DT'
        dt = DecisionTreeClassifier(random_state=RAND_STATE)
        dt.fit(X_train, y_train)
        preds = dt.predict(X_test)
        evalModel(m)
        confMat()

    def create_model(trial):
        max_depth = trial.suggest_int("max_depth", 2, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 16)
        min_weight_fraction_leaf = trial.suggest_uniform('min_weight_fraction_leaf', 0.0, 0.5)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        model = DecisionTreeClassifier(min_samples_split=min_samples_split, min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth, 
                                       min_samples_leaf=min_samples_leaf, random_state=RAND_STATE, class_weight=weight_dict)
        return model

    if 'OADT' in TOTEST:
        m = 'OADT'
        optimizer = Optimizer('acc')
        dt_acc_params = optimizer.optimize()
        dt_acc_params['random_state'] = RAND_STATE
        dt_acc = DecisionTreeClassifier(**dt_acc_params)
        dt_acc.fit(X_train, y_train)
        preds = dt_acc.predict(X_test)
        print('Optimized on accuracy')
        evalModel(m)
        confMat()
        retdict.update({m:DecisionTreeClassifier(**dt_acc_params)})

    if 'OFDT' in TOTEST:
        m = 'OFDT'
        optimizer = Optimizer('f1')
        dt_f1_params = optimizer.optimize()
        dt_f1_params['random_state'] = RAND_STATE
        dt_f1 = DecisionTreeClassifier(**dt_f1_params)
        dt_f1.fit(X_train, y_train)
        preds = dt_f1.predict(X_test)
        print('Optimized on F1-score')
        evalModel(m)
        confMat()
        retdict.update({m:DecisionTreeClassifier(**dt_f1_params)})

    if 'OBDT' in TOTEST:
        m = 'OBDT'
        optimizer = Optimizer('beta')
        dt_beta_params = optimizer.optimize()
        dt_beta_params['random_state'] = RAND_STATE
        dt_beta = DecisionTreeClassifier(**dt_beta_params)
        dt_beta.fit(X_train, y_train)
        preds = dt_beta.predict(X_test)
        print('Optimized on fbeta')
        evalModel(m)
        confMat()
        retdict.update({m:DecisionTreeClassifier(**dt_beta_params)})

    if 'ORDT' in TOTEST:
        m = 'ORDT'
        optimizer = Optimizer('recall')
        dt_rec_params = optimizer.optimize()
        dt_rec_params['random_state'] = RAND_STATE
        dt_rec = DecisionTreeClassifier(**dt_rec_params)
        dt_rec.fit(X_train, y_train)
        preds = dt_rec.predict(X_test)
        print('Optimized on recall')
        evalModel(m)
        confMat()
        retdict.update({m:DecisionTreeClassifier(**dt_rec_params)})

    if 'OPDT' in TOTEST:
        m = 'OPDT'
        optimizer = Optimizer('pr')
        dt_pr_params = optimizer.optimize()
        dt_pr_params['random_state'] = RAND_STATE
        dt_pr = DecisionTreeClassifier(**dt_pr_params)
        dt_pr.fit(X_train, y_train)
        preds = dt_pr.predict(X_test)
        print('optimized on PR AUC')
        evalModel(m)
        confMat()
        retdict.update({m:DecisionTreeClassifier(**dt_pr_params)})
        print('---------------------------------------------------')
    out_queue.put(retdict)

def BC(out_queue):
    retdict = {}
    global create_model, y_test, preds
    if 'BC' in TOTEST:
        m = 'BC'
        bc = BaggingClassifier(random_state=RAND_STATE)
        bc.fit(X_train, y_train)
        preds = bc.predict(X_test)
        evalModel(m)
        confMat()

    def create_model(trial):
        n_estimators = trial.suggest_int('n_estimators', 2, 500)
        max_samples = trial.suggest_int('max_samples', 1, 100)
        model = BaggingClassifier(n_estimators=n_estimators, max_samples=max_samples, random_state=RAND_STATE)
        return model

    if 'OABC' in TOTEST:
        m = 'OABC'
        optimizer = Optimizer('acc')
        bc_acc_params = optimizer.optimize()
        bc_acc_params['random_state'] = RAND_STATE
        bc_acc = BaggingClassifier(**bc_acc_params)
        bc_acc.fit(X_train, y_train)
        preds = bc_acc.predict(X_test)
        print('Optimized on accuracy')
        evalModel(m)
        confMat()
        retdict.update({m:BaggingClassifier(**bc_acc_params)})

    if 'OFBC' in TOTEST:
        m = 'OFBC'
        optimizer = Optimizer('f1')
        bc_f1_params = optimizer.optimize()
        bc_f1_params['random_state'] = RAND_STATE
        bc_f1 = BaggingClassifier(**bc_f1_params)
        bc_f1.fit(X_train, y_train)
        preds = bc_f1.predict(X_test)
        print('Optimized on F1-score')
        evalModel(m)
        confMat()
        retdict.update({m:BaggingClassifier(**bc_f1_params)})

    if 'OBBC' in TOTEST:
        m = 'OBBC'
        optimizer = Optimizer('beta')
        bc_beta_params = optimizer.optimize()
        bc_beta_params['random_state'] = RAND_STATE
        bc_beta = BaggingClassifier(**bc_beta_params)
        bc_beta.fit(X_train, y_train)
        preds = bc_beta.predict(X_test)
        print('Optimized on fbeta')
        evalModel(m)
        confMat()
        retdict.update({m:BaggingClassifier(**bc_beta_params)})

    if 'ORBC' in TOTEST:
        m = 'ORBC'
        optimizer = Optimizer('recall')
        bc_rec_params = optimizer.optimize()
        bc_rec_params['random_state'] = RAND_STATE
        bc_rec = BaggingClassifier(**bc_rec_params)
        bc_rec.fit(X_train, y_train)
        preds = bc_rec.predict(X_test)
        print('Optimized on recall')
        evalModel(m)
        confMat()
        retdict.update({m:BaggingClassifier(**bc_rec_params)})

    if 'OPBC' in TOTEST:
        m = 'OPBC'
        optimizer = Optimizer('pr')
        bc_pr_params = optimizer.optimize()
        bc_pr_params['random_state'] = RAND_STATE
        bc_pr = BaggingClassifier(**bc_pr_params)
        bc_pr.fit(X_train, y_train)
        preds = bc_pr.predict(X_test)
        print('optimized on PR AUC')
        evalModel(m)
        confMat()
        retdict.update({m:BaggingClassifier(**bc_pr_params)})
        print('---------------------------------------------------')
    out_queue.put(retdict)

def KNN(out_queue):
    retdict = {}
    global create_model, y_test, preds
    if 'KNN' in TOTEST:
        m = 'KNN'
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        evalModel(m)
        confMat()

    sampler = TPESampler(seed=0)
    def create_model(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 2, 25)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        return model

    if 'OAKNN' in TOTEST:
        m = 'OAKNN'
        optimizer = Optimizer('acc')
        knn_acc_params = optimizer.optimize()
        knn_acc = KNeighborsClassifier(**knn_acc_params)
        knn_acc.fit(X_train, y_train)
        preds = knn_acc.predict(X_test)
        print('Optimized on accuracy')
        evalModel(m)
        confMat()
        retdict.update({m:KNeighborsClassifier(**knn_acc_params)})

    if 'OFKNN' in TOTEST:
        m = 'OFKNN'
        optimizer = Optimizer('f1')
        knn_f1_params = optimizer.optimize()
        knn_f1 = KNeighborsClassifier(**knn_f1_params)
        knn_f1.fit(X_train, y_train)
        preds = knn_f1.predict(X_test)
        print('Optimized on F1-score')
        evalModel(m)
        confMat()
        retdict.update({m:KNeighborsClassifier(**knn_f1_params)})

    if 'OBKNN' in TOTEST:
        m = 'OBKNN'
        optimizer = Optimizer('beta')
        knn_beta_params = optimizer.optimize()
        knn_beta = KNeighborsClassifier(**knn_beta_params)
        knn_beta.fit(X_train, y_train)
        preds = knn_beta.predict(X_test)
        print('Optimized on accuracy')
        evalModel(m)
        confMat()
        retdict.update({m:KNeighborsClassifier(**knn_beta_params)})

    if 'ORKNN' in TOTEST:
        m = 'ORKNN'
        optimizer = Optimizer('recall')
        knn_rec_params = optimizer.optimize()
        knn_rec = KNeighborsClassifier(**knn_rec_params)
        knn_rec.fit(X_train, y_train)
        preds = knn_rec.predict(X_test)
        print('Optimized on recall')
        evalModel(m)
        confMat()
        retdict.update({m:KNeighborsClassifier(**knn_rec_params)})

    if 'OPKNN' in TOTEST:
        m = 'OPKNN'
        optimizer = Optimizer('pr')
        knn_pr_params = optimizer.optimize()
        knn_pr = KNeighborsClassifier(**knn_pr_params)
        knn_pr.fit(X_train, y_train)
        preds = knn_pr.predict(X_test)
        print('optimized on PR AUC')
        evalModel(m)
        confMat()
        retdict.update({m:KNeighborsClassifier(**knn_pr_params)})

    print('---------------------------------------------------')
    out_queue.put(retdict)

def ABC(out_queue):
    retdict = {}
    global create_model, y_test, preds
    if 'ABC' in TOTEST:
        m = 'ABC'
        abc = AdaBoostClassifier(random_state=RAND_STATE)
        abc.fit(X_train, y_train)
        preds = abc.predict(X_test)
        evalModel(m)
        confMat()

    def create_model(trial):
        n_estimators = trial.suggest_int("n_estimators", 2, 500)
        learning_rate = trial.suggest_uniform('learning_rate', 0.0005, 1.0)
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=RAND_STATE)
        return model

    if 'OAABC' in TOTEST:
        m = 'OAABC'
        optimizer = Optimizer('acc')
        abc_acc_params = optimizer.optimize()
        abc_acc_params['random_state'] = RAND_STATE
        abc_acc = AdaBoostClassifier(**abc_acc_params)
        abc_acc.fit(X_train, y_train)
        preds = abc_acc.predict(X_test)
        print('Optimized on accuracy')
        evalModel(m)
        confMat()
        retdict.update({m:AdaBoostClassifier(**abc_acc_params)})

    if 'OFABC' in TOTEST:
        m = 'OFABC'
        optimizer = Optimizer('f1')
        abc_f1_params = optimizer.optimize()
        abc_f1_params['random_state'] = RAND_STATE
        abc_f1 = AdaBoostClassifier(**abc_f1_params)
        abc_f1.fit(X_train, y_train)
        preds = abc_f1.predict(X_test)
        print('Optimized on F1-score')
        evalModel(m)
        confMat()
        retdict.update({m:AdaBoostClassifier(**abc_f1_params)})

    if 'OBABC' in TOTEST:
        m = 'OBABC'
        optimizer = Optimizer('beta')
        abc_beta_params = optimizer.optimize()
        abc_beta_params['random_state'] = RAND_STATE
        abc_beta = AdaBoostClassifier(**abc_beta_params)
        abc_beta.fit(X_train, y_train)
        preds = abc_beta.predict(X_test)
        print('Optimized on accuracy')
        evalModel(m)
        confMat()
        retdict.update({m:AdaBoostClassifier(**abc_beta_params)})

    if 'ORABC' in TOTEST:
        m = 'ORABC'
        optimizer = Optimizer('recall')
        abc_rec_params = optimizer.optimize()
        abc_rec_params['random_state'] = RAND_STATE
        abc_rec = AdaBoostClassifier(**abc_rec_params)
        abc_rec.fit(X_train, y_train)
        preds = abc_rec.predict(X_test)
        print('Optimized on recall')
        evalModel(m)
        confMat()
        retdict.update({m: AdaBoostClassifier(**abc_rec_params)})

    if 'OPABC' in TOTEST:
        m = 'OPABC'
        optimizer = Optimizer('pr')
        abc_pr_params = optimizer.optimize()
        abc_pr_params['random_state'] = RAND_STATE
        abc_pr = AdaBoostClassifier(**abc_pr_params)
        abc_pr.fit(X_train, y_train)
        preds = abc_pr.predict(X_test)
        print('optimized on PR AUC')
        evalModel(m)
        confMat()
        retdict.update({m: AdaBoostClassifier(**abc_pr_params)})

    print('---------------------------------------------------')
    out_queue.put(retdict)

def ET(out_queue):
    retdict = {}
    global create_model, y_test, preds
    if 'ET' in TOTEST:
        m = 'ET'
        et = ExtraTreesClassifier(random_state=RAND_STATE)
        et.fit(X_train, y_train)
        preds = et.predict(X_test)
        evalModel(m)
        confMat()

    def create_model(trial):
        n_estimators = trial.suggest_int("n_estimators", 2, 500)
        max_depth = trial.suggest_int("max_depth", 2, 20)
        model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=RAND_STATE, class_weight=weight_dict)
        return model

    if 'OAET' in TOTEST:
        m = 'OAET'
        optimizer = Optimizer('acc')
        et_acc_params = optimizer.optimize()
        et_acc_params['random_state'] = RAND_STATE
        et_acc = ExtraTreesClassifier(**et_acc_params)
        et_acc.fit(X_train, y_train)
        preds = et_acc.predict(X_test)
        print('Optimized on accuracy')
        evalModel(m)
        confMat()
        retdict.update({m:ExtraTreesClassifier(**et_acc_params)})

    if 'OFET' in TOTEST:
        m = 'OFET'
        optimizer = Optimizer('f1')
        et_f1_params = optimizer.optimize()
        et_f1_params['random_state'] = RAND_STATE
        et_f1 = ExtraTreesClassifier(**et_f1_params)
        et_f1.fit(X_train, y_train)
        preds = et_f1.predict(X_test)
        print('Optimized on F1-score')
        evalModel(m)
        confMat()
        retdict.update({m:ExtraTreesClassifier(**et_f1_params)})

    if 'OBET' in TOTEST:
        m = 'OBET'
        optimizer = Optimizer('beta')
        et_beta_params = optimizer.optimize()
        et_beta_params['random_state'] = RAND_STATE
        et_beta = ExtraTreesClassifier(**et_beta_params)
        et_beta.fit(X_train, y_train)
        preds = et_beta.predict(X_test)
        print('Optimized on accuracy')
        evalModel(m)
        confMat()
        retdict.update({m:ExtraTreesClassifier(**et_beta_params)})

    if 'ORET' in TOTEST:
        m = 'ORET'
        optimizer = Optimizer('recall')
        et_rec_params = optimizer.optimize()
        et_rec_params['random_state'] = RAND_STATE
        et_rec = ExtraTreesClassifier(**et_rec_params)
        et_rec.fit(X_train, y_train)
        preds = et_rec.predict(X_test)
        print('Optimized on recall')
        evalModel(m)
        confMat()
        retdict.update({m:ExtraTreesClassifier(**et_rec_params)})

    if 'OPET' in TOTEST:
        m = 'OPET'
        optimizer = Optimizer('pr')
        et_pr_params = optimizer.optimize()
        et_pr_params['random_state'] = RAND_STATE
        et_pr = ExtraTreesClassifier(**et_pr_params)
        et_pr.fit(X_train, y_train)
        preds = et_pr.predict(X_test)
        print('optimized on PR AUC')
        evalModel(m)
        confMat()
        retdict.update({m:ExtraTreesClassifier(**et_pr_params)})
    print('---------------------------------------------------')
    out_queue.put(retdict)


def SL_fit_and_save(mode, result, head):

    from numpy import hstack
    from numpy import vstack
    from numpy import asarray
    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
 


    # create a list of base-models
    def get_models():
        models = list()
        for item in result:
            models.append(mdict[item])
            
        return models

    # collect out of fold predictions form k-fold cross validation
    def get_out_of_fold_predictions(X, y, models):
        meta_X, meta_y = list(), list()
        # define split of data
        kfold = KFold(n_splits=10, shuffle=True)
        # enumerate splits
        for train_ix, test_ix in kfold.split(X):
            fold_yhats = list()
            # get data
            train_X, test_X = X.iloc[train_ix], X.iloc[test_ix]
            train_y, test_y = y.iloc[train_ix], y.iloc[test_ix]
            meta_y.extend(test_y)
            # fit and make predictions with each sub-model
            for model in models:
                model.fit(train_X, train_y)
                if hasattr(model, 'predict_proba'):
                    yhat = model.predict_proba(test_X)
                else:
                    temp = model.predict(test_X)
                    yhat = np.column_stack(([int(not z) for z in temp], temp))
                # store columns
                fold_yhats.append(yhat)
            # store fold yhats as columns
            meta_X.append(hstack(fold_yhats))
        return vstack(meta_X), asarray(meta_y)

    # fit all base models on the training dataset
    def fit_base_models(X, y, models):
        for model in models:
            model.fit(X, y)

    # fit a meta model
    def fit_meta_model(X, y):
        model = mdict[head]
        model.fit(X, y)
        return model

    # evaluate a list of models on a dataset
    def evaluate_models(X, y, models):
        for model in models:
            yhat = model.predict(X)
            beta = fbeta_score(y, yhat, beta=BETA)
            print('%s beta-score: %.4f' % (model.__class__.__name__, beta))



    print('Train', X_train.shape, y_train.shape, 'Test', X_test.shape, y_test.shape)
    # get models
    models = get_models()
    # get out of fold predictions
    meta_X, meta_y = get_out_of_fold_predictions(X_train, y_train, models)
    print('Meta ', meta_X.shape, meta_y.shape)
    # fit base models
    fit_base_models(X_train, y_train, models)
    # fit the meta model
    meta_model = fit_meta_model(meta_X, meta_y)
    # evaluate base models
    #evaluate_models(X_test, y_test, models)
    # evaluate meta model
    #preds = super_learner_predictions(X_test, models, meta_model)
    
    #evalModel(mode)
    #print('SuperLearner precision: ', precision_score(y_test, preds))
    #confMat()

    filename = './models/{}/models_{}_{}.sav'.format(LABEL, mode, LABEL)
    pickle.dump(models, open(filename, 'wb'))
    filename = './models/{}/metamodel_{}_{}.sav'.format(LABEL, mode, LABEL)
    pickle.dump(meta_model, open(filename, 'wb'))
    
def SL():
    sampler = TPESampler(seed=0)
    if 'SL' in TOTEST:
        m = 'SL'
        # ENSEMBLING ---------- (using single models in the first layer and LogisticRegressor as meta learner)
        model = SuperLearner(folds=5, random_state=666)
        model.add([bc, lgb, xgb, rf, dt, knn])
        model.add_meta(LogisticRegression())
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        print('SuperLearner MODELS: [bc, lgb, xgb, rf, dt, knn] META: lr')
        evalModel(m)
        print('SuperLearner precision: ', precision_score(y_test, preds))
        confMat()
        print('---------------------------------------------------')



    def create_model(trial):
        model_names = list()
        optimized_models = ['OARF', 'OFRF', 'ORRF', 'OBRF', 'OPRF', 'OAXGB', 'OFXGB', 'ORXGB', 'OBXGB', 'OPXGB', 'OALGBM', 'OFLGBM', 'ORLGBM', 'OBLGBM', 'OPLGBM', 'OADT', 'OFDT', 'ORDT', 'OBDT', 'OPDT', 'OAKNN', 'OFKNN', 'ORKNN', 'OBKNN', 'OPKNN', 'OABC', 'OFBC', 'ORBC', 'OBBC', 'OPBC', 'OAABC', 'OFABC', 'ORABC', 'OBABC', 'OPABC', 'OAET', 'OFET', 'ORET', 'OBET', 'OPET']
        models_list = ['RF', 'XGB', 'LGBM', 'DT', 'KNN', 'BC'] + [i for i in TOTEST if i in optimized_models] + ['LR', 'ABC', 'SGD', 'ET', 'MLP', 'GB', 'RDG', 'PCP', 'PAC']

        head_list = ['RF', 'XGB', 'LGBM', 'DT', 'KNN', 'BC', 'LR', 'ABC', 'SGD', 'ET', 'MLP', 'GB', 'RDG', 'PCP', 'PAC']
        n_models = trial.suggest_int("n_models", 2, 5)
        for i in range(n_models):
            model_item = trial.suggest_categorical('model_{}'.format(i), models_list)
            if model_item not in model_names:
                model_names.append(model_item)

        folds = trial.suggest_int("folds", 2, 6)

        model = SuperLearner(folds=folds, random_state=666)
        models = list()
        for item in model_names:
            models.append(mdict[item])
        model.add(models)
        head = trial.suggest_categorical('head', head_list)
        model.add_meta(mdict[head])

        return model




    if 'OFSL' in TOTEST:
        m = 'OFSL'
        def objective(trial):
            model = create_model(trial)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = f1_score(y_test, preds)
            return score

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=150)


        params = study.best_params

        head = params['head']
        folds = params['folds']
        del params['head'], params['n_models'], params['folds']
        result = list()
        for key, value in params.items():
            if value not in result:
                result.append(value)

        SL_fit_and_save(m, result, head)


    if 'OBSL' in TOTEST:
        m = 'OBSL'
        def objective(trial):
            model = create_model(trial)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = fbeta_score(y_test, preds, beta=BETA)
            return score

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=150)


        params = study.best_params

        head = params['head']
        folds = params['folds']
        del params['head'], params['n_models'], params['folds']
        result = list()
        for key, value in params.items():
            if value not in result:
                result.append(value)

        SL_fit_and_save(m, result, head)

    if 'ORSL' in TOTEST:
        m = 'ORSL'
        def objective(trial):
            model = create_model(trial)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = recall_score(y_test, preds)
            return score

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=150)


        params = study.best_params

        head = params['head']
        folds = params['folds']
        del params['head'], params['n_models'], params['folds']
        result = list()
        for key, value in params.items():
            if value not in result:
                result.append(value)

        SL_fit_and_save(m, result, head)


    if 'OPSL' in TOTEST:
        m = 'OPSL'
        def objective(trial):
            model = create_model(trial)
            model.fit(X_train, y_train)
            if hasattr(model, 'decision_function'):
                y_score = model.decision_function(X_test)
                return average_precision_score(y_test, y_score)
            elif hasattr(model, 'predict_proba'):
                try:
                    y_score = model.predict_proba(X_test)[:, 1]
                except IndexError:
                    y_score = model.predict_proba(X_test)
                return average_precision_score(y_test, y_score)
            else:
                print("{} didn't have predict_proba or decision_function".format(model))
                preds = model.predict(X_test)
                score = f1_score(y_test, preds)

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=150)


        params = study.best_params

        head = params['head']
        folds = params['folds']
        del params['head'], params['n_models'], params['folds']
        result = list()
        for key, value in params.items():
            if value not in result:
                result.append(value)

        SL_fit_and_save(m, result, head)
        
if __name__ == '__main__':     
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--CLEAN", "-c", type=str2bool, nargs='?', const=True, default=False, help="Boolean for cleaning the data?")
    parser.add_argument("--NUM_UNIQUE_CCS", "-nccs", type=int, help="Number of unique CCs to be had when cleaning the data, only relevant if clean is true")
    parser.add_argument("--SUBSET_SIZE", "-ss", type=int, help="How much of the 220,000 patients do you want to work with?")
    parser.add_argument("--RAND_STATE", "-rs", type=int, help="Change the random_state through with np and sklearn work")
    parser.add_argument("--BETA", "-b", type=float, help="fbeta_score, beta > 1 favors recall < 1 favors precision")
    parser.add_argument("--VITALS", "-v", type=str2bool, nargs='?', const=True, default=False, help="boolean for whether to work with vitals or not")
    parser.add_argument("--SAVE_CLEANED", "-sav", type=str2bool, nargs='?', const=True, default=False, help="boolean for whether to save the cleaned file, only relevant if clean is true")
    parser.add_argument("--LABEL", "-l", type=str, help="str what to label the saved file with, only relevant if clean is true")
    parser.add_argument("--ALL_DATA", "-ad", type=str2bool, nargs='?', const=True, default=False, help="clean all patients of specificied data, only relevant if clean is true")
    args = parser.parse_args()
    print("ARGUMENTS PASSED: {}".format(args))
    
    CLEAN = args.CLEAN
    NUM_UNIQUE_CCS = args.NUM_UNIQUE_CCS
    SUBSET_SIZE = args.SUBSET_SIZE
    RAND_STATE = args.RAND_STATE
    BETA = args.BETA
    VITALS = args.VITALS
    SAVE_CLEANED = args.SAVE_CLEANED
    LABEL = args.LABEL
    ALL_DATA = args.ALL_DATA
    
    TOTEST = ['OAXGB', 'OBXGB', 'OALGBM', 'OBLGBM', 'OPXGB', 'OPLGBM','OPRF','OPABC', 'OPKNN', 'OPET', 'OPDT','OPSL', 'OBSL']


    if CLEAN:
        df = clean(VITALS, NUM_UNIQUE_CCS, SUBSET_SIZE, LABEL, ALL_DATA, SAVE_CLEANED=True)
    else:
        if VITALS:
            filename = './models/{}/data_vitals_cleaned_{}.csv'.format(LABEL, LABEL)
            df = pd.read_csv(filename)
        else:
            filename = './models/{}/data_cleaned_{}.csv'.format(LABEL, LABEL)
            df = pd.read_csv(filename)

    print("# OF FEATURES: {}  |  # OF PATIENTS: {}".format(len(df.columns)-1, len(df)))
    print("RAND_STATE: {}    |  BETA:   {}".format(RAND_STATE, BETA))

    print("TOTEST: {}".format(TOTEST))

    weight_list = compute_class_weight('balanced', classes=[0, 1],y=df['admit_binary'])
    weight_dict = {i:weight for i, weight in enumerate(weight_list)}
    scale_pos_weight = weight_list[1]/weight_list[0]

    mdict = {
        'RF': RandomForestClassifier(random_state=RAND_STATE),
        'XGB': XGBClassifier(random_state=RAND_STATE),
        'LGBM': LGBMClassifier(random_state=RAND_STATE),
        'DT': DecisionTreeClassifier(random_state=RAND_STATE),
        'KNN': KNeighborsClassifier(),
        'BC': BaggingClassifier(random_state=RAND_STATE),
        'LR': LogisticRegression(random_state=RAND_STATE),
        'ABC': AdaBoostClassifier(random_state=RAND_STATE),
        'SGD': SGDClassifier(random_state=RAND_STATE), 
        'ET': ExtraTreesClassifier(random_state=RAND_STATE),
        'MLP': MLPClassifier(random_state=RAND_STATE),
        'GB': GradientBoostingClassifier(random_state=RAND_STATE),
        'RDG': RidgeClassifier(random_state=RAND_STATE),
        'PCP': Perceptron(random_state=RAND_STATE),
        'PAC': PassiveAggressiveClassifier(random_state=RAND_STATE)
    }

    #Defining the independent variables and dependent variables
    X = df.drop('admit_binary', axis=1)
    y = df['admit_binary']

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=RAND_STATE, shuffle=False)

    # MULTIPROCESSING:
    num_cores = int(os.getenv('SLURM_CPUS_PER_TASK')) # can't be 8, so making it 7
    
    jobs = [RF, XGB, LGBM, DT, KNN, ABC, ET] # took out BC
    
    out_queue = mp.Queue()
    workers = [ mp.Process(target=job, args=(out_queue,) ) for job in jobs ]

    [work.start() for work in workers]
    [work.join() for work in workers]

    for j in range(len(workers)):
        mdict.update(out_queue.get())

        
        
    
    # SuperLearner ensembling:
    #    mdict will have been updated with optimized parameters
    SL()
