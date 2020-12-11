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
from utils.cleaning import clean, str2bool # clean for processing data, str2bool for command line operability
from utils import superlearnerFitAndEval

class Optimizer:
    def __init__(self, metric, trials=50):
        self.metric = metric
        self.trials = trials
        self.sampler = TPESampler(seed=0)
        
    def objective(self, trial):
        model = create_model(trial)
        model.fit(X_train, y_train)
        
        if self.metric == 'OA':
            preds = model.predict(X_test)
            return accuracy_score(y_test, preds)
        elif self.metric == 'OR':
            preds = model.predict(X_test)
            return recall_score(y_test, preds)
        elif self.metric == 'OF':
            preds = model.predict(X_test)
            return f1_score(y_test, preds)
        elif self.metric == 'OB':
            preds = model.predict(X_test)
            return fbeta_score(y_test, preds, beta=BETA)
        elif self.metric == 'OP':
            if hasattr(model, 'decision_function'):
                y_score = model.decision_function(X_test)
            elif hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:, 1]
            return average_precision_score(y_test, y_score)
        elif self.metric == 'OC':
            if hasattr(model, 'decision_function'):
                y_score = model.decision_function(X_test)
            elif hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:, 1]
            return roc_auc_score(y_test, y_score)
                
            
            
    def optimize(self):
        study = optuna.create_study(direction="maximize", sampler=self.sampler)
        study.optimize(self.objective, n_trials=self.trials)
        return study.best_params
    
def confMat():
    print(confusion_matrix(y_test, preds))
    

def evalModel(m):
    print(m + ' accuracy: ', accuracy_score(y_test, preds))
    print(m + ' f1-score: ', f1_score(y_test, preds))
    print(m + ' fbeta-score: ', fbeta_score(y_test, preds, beta=BETA))
    print(m + ' recall: ', recall_score(y_test, preds))
    
def getScores(model):
    
    AUROC, AUPRC = None, None
    
    if hasattr(model, 'decision_function'):
        y_score = model.decision_function(X_test)
    elif hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        AUROC = 'only binary prediction'
        AUPRC = 'only binary prediction'
        
    if AUPRC is None:
        AUROC = roc_auc_score(y_test, y_score)
        AUPRC = average_precision_score(y_test, y_score)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    fb = fbeta_score(y_test, preds, beta=BETA)
    rec = recall_score(y_test, preds)
    
    # scores = [label of model, AUROC, AUPRC, accuracy, f1_score, fbeta score, recall]
    scores = [m, AUROC, AUPRC, acc, f1, fb, rec]
    return scores
    
def optimizeAndSaveScores(model, m):
    global score_queue, preds
    
    optimizer = Optimizer(m[:2])
    model_params = optimizer.optimize()
    if m[-3:] != 'KNN':
        model_params['random_state'] = RAND_STATE
    optimizedModel = model(**model_params)
    optimizedModel.fit(X_train, y_train)
    preds = optimizedModel.predict(X_test)
    print('Optimized on ' + m[:2])
    evalModel(m)
    confMat()
    
    score_queue.put(getScores(optimizedModel))
    return {m: model(**model_params)}
    
def baselineThenOptimize(model, label):
    global m, score_queue
    retdict = {}
    
    if label in TOTEST:
        m = label
        if m[-3:] != 'KNN':
            baselineModel = model(random_state=RAND_STATE)
        else:
            baselineModel = model()
        baselineModel.fit(X_train, y_train)
        preds = baselineModel.predict(X_test)
        evalModel(m)
        confMat()
        score_queue.put(getScores(baselineModel))
        
    for metric in metrics:
        m = metric+label
        if m in TOTEST:
            retdict.update(optimizeAndSaveScores(model, m))
 
        
    print('---------------------------------------------------')
    out_queue.put(retdict)

    
def RF(out_queue, score_queue):
    global create_model, y_test, preds
    model = RandomForestClassifier
    label = 'RF'
   
    def create_model(trial):
        max_depth = trial.suggest_int("max_depth", 2, 32)
        n_estimators = trial.suggest_int("n_estimators", 2, 500)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        weights = weight_dict
        model = RandomForestClassifier(min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, max_depth=max_depth, random_state=RAND_STATE, class_weight=weights)
        return model
    
    baselineThenOptimize(model, label)

def XGB(out_queue, score_queue):
    retdict = {}
    global create_model, y_test, preds
    model = XGBClassifier
    label = 'XGB'
    
    def create_model(trial):
        max_depth = trial.suggest_int("max_depth", 2, 30)
        n_estimators = trial.suggest_int("n_estimators", 1, 500)
        learning_rate = trial.suggest_uniform('learning_rate', 0.0000001, 1)
        min_child_weight = trial.suggest_uniform('min_child_weight', 0.4, 0.8)
        reg_lambda = trial.suggest_int('reg_lambda', 6, 10)
        gamma = trial.suggest_uniform('gamma', 0.0000001, 1)
        model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, min_child_weight=min_child_weight, reg_lambda=reg_lambda, max_depth=max_depth, gamma=gamma, random_state=RAND_STATE, scale_pos_weight=scale_pos_weight)
        return model
    
    baselineThenOptimize(model, label)
    

def LGBM(out_queue, score_queue):
    retdict = {}
    global create_model, y_test, preds
    model = LGBMClassifier
    label = 'LGBM'

    def create_model(trial):
        max_depth = trial.suggest_int("max_depth", 2, 30)
        n_estimators = trial.suggest_int("n_estimators", 1, 500)
        learning_rate = trial.suggest_uniform('learning_rate', 0.0000001, 1)
        num_leaves = trial.suggest_int("num_leaves", 2, 5000)
        min_child_samples = trial.suggest_int('min_child_samples', 3, 200)
        model = LGBMClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, num_leaves=num_leaves, min_child_samples=min_child_samples,
                               random_state=RAND_STATE, is_unbalance=False)
        return model
    
    baselineThenOptimize(model, label)

def LR(out_queue, score_queue):
    retdict = {}
    global create_model, y_test, preds
    if 'LR' in TOTEST:
        m = 'LR'
        lr = LogisticRegression(random_state=RAND_STATE)
        lr.fit(X_train, y_train)
        preds = lr.predict(X_test)
        evalModel(m, preds)
        confMat()
    print('---------------------------------------------------')


def DT(out_queue, score_queue):
    retdict = {}
    global create_model, y_test, preds
    model = DecisionTreeClassifier
    label = 'DT'
    

    def create_model(trial):
        max_depth = trial.suggest_int("max_depth", 2, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 16)
        min_weight_fraction_leaf = trial.suggest_uniform('min_weight_fraction_leaf', 0.0, 0.5)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        model = DecisionTreeClassifier(min_samples_split=min_samples_split, min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth, 
                                       min_samples_leaf=min_samples_leaf, random_state=RAND_STATE, class_weight=weight_dict)
        return model
    
    baselineThenOptimize(model, label)

    
def BC(out_queue, score_queue):
    retdict = {}
    global create_model, y_test, preds
    model = BaggingClassifier
    label = 'BC'

    def create_model(trial):
        n_estimators = trial.suggest_int('n_estimators', 2, 500)
        max_samples = trial.suggest_int('max_samples', 1, 100)
        model = BaggingClassifier(n_estimators=n_estimators, max_samples=max_samples, random_state=RAND_STATE)
        return model
    
    baselineThenOptimize(model, label)

def KNN(out_queue, score_queue):
    retdict = {}
    global create_model, y_test, preds
    model = KNeighborsClassifier
    label = 'KNN'
    
    sampler = TPESampler(seed=0)
    def create_model(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 2, 25)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        return model
    
    baselineThenOptimize(model, label)

def ABC(out_queue, score_queue):
    retdict = {}
    global create_model, y_test, preds
    model = AdaBoostClassifier
    label = 'ABC'
    
    def create_model(trial):
        n_estimators = trial.suggest_int("n_estimators", 2, 500)
        learning_rate = trial.suggest_uniform('learning_rate', 0.0005, 1.0)
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=RAND_STATE)
        return model
    
    baselineThenOptimize(model, label)

def ET(out_queue, score_queue):
    retdict = {}
    global create_model, y_test, preds
    model = ExtraTreesClassifier
    label = 'ET'
    
    def create_model(trial):
        n_estimators = trial.suggest_int("n_estimators", 2, 500)
        max_depth = trial.suggest_int("max_depth", 2, 20)
        model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=RAND_STATE, class_weight=weight_dict)
        return model
    
    baselineThenOptimize(model, label)


def SL_fit_and_save(mode, result, head):
    baseModels = list()
    for item in result:
        baseModels.append(mdict[item])
    
    metaModel = mdict[head]
    
    scores, models, meta_model = superlearnerFitAndEval(X_train, X_test, y_train, y_test, 
                                                        models, head, model_name=mode, 
                                                        full_fit=True, optimized=True)

    filename = './models/{}/models_{}_{}.sav'.format(LABEL, mode, LABEL)
    pickle.dump(models, open(filename, 'wb'))
    filename = './models/{}/metamodel_{}_{}.sav'.format(LABEL, mode, LABEL)
    pickle.dump(meta_model, open(filename, 'wb'))
    
    return scores
    
def SL():
    scores = []
    sampler = TPESampler(seed=0)
    label = 'SL'

    def create_model(trial):
        model_names = list()
        optimized_models = ['OARF', 'OFRF', 'ORRF', 'OBRF', 'OPRF', 'OCRF', 'OAXGB', 'OFXGB', 'ORXGB', 'OBXGB', 'OPXGB', 'OCXGB', 'OALGBM', 'OFLGBM', 'ORLGBM', 'OBLGBM', 'OPLGBM', 'OCLGBM', 'OADT', 'OFDT', 'ORDT', 'OBDT', 'OPDT', 'OCDT', 'OAKNN', 'OFKNN', 'ORKNN', 'OBKNN', 'OPKNN', 'OCKNN', 'OABC', 'OFBC', 'ORBC', 'OBBC', 'OPBC', 'OCBC', 'OAABC', 'OFABC', 'ORABC', 'OBABC', 'OPABC', 'OCABC', 'OAET', 'OFET', 'ORET', 'OBET', 'OPET', 'OCET']
        models_list = ['RF', 'XGB', 'LGBM', 'DT', 'KNN', 'BC'] + [i for i in TOTEST if i in optimized_models] + ['LR', 'ABC', 'SGD', 'ET', 'MLP', 'GB', 'RDG', 'PCP', 'PAC']

        head_list = ['RF', 'XGB', 'LGBM', 'DT', 'KNN', 'BC', 'LR', 'ABC', 'SGD', 'ET', 'MLP', 'GB', 'RDG', 'PCP', 'PAC']
        n_models = trial.suggest_int("n_models", 2, 5)
        for i in range(n_models):
            model_item = trial.suggest_categorical('model_{}'.format(i), models_list)
            if model_item not in model_names:
                model_names.append(model_item)

        folds = trial.suggest_int("folds", 2, 6)

        models = list()
        for item in model_names:
            models.append(mdict[item])
        head = trial.suggest_categorical('head', head_list)
        head = mdict[head]
        return models, head
    
    for metric in metrics:
        m = metric+label
        
        if m in TOTEST:
            def objective(trial):
                models, head = create_model(trial)
                score = superlearnerFitAndEval(X_train, X_test, y_train, y_test, 
                                                models, head, model_name=m, 
                                                full_fit=True, optimized=False)
                # score is correct metric based on m passed, 
                # in this case 'OF' tells superlearnerFitAndEval to return fscore
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

            scores.append(SL_fit_and_save(m, result, head))
            
    return scores
    
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
    
#     TOTEST = ['OAXGB', 'OBXGB', 'OALGBM', 'OBLGBM', 'OPXGB', 'OPLGBM','OPRF','OPABC', 'OPKNN', 'OPET', 'OPDT','OPSL', 'OBSL']
    TOTEST = ['OARF', 'OFRF', 'ORRF', 'OBRF', 'OPRF', 'OCRF', 'OAXGB', 'OFXGB', 'ORXGB', 'OBXGB', 'OPXGB', 'OCXGB', 'OALGBM', 'OFLGBM', 'ORLGBM', 'OBLGBM', 'OPLGBM', 'OCLGBM', 'OADT', 'OFDT', 'ORDT', 'OBDT', 'OPDT', 'OCDT', 'OAKNN', 'OFKNN', 'ORKNN', 'OBKNN', 'OPKNN', 'OCKNN', 'OABC', 'OFBC', 'ORBC', 'OBBC', 'OPBC', 'OCBC', 'OAABC', 'OFABC', 'ORABC', 'OBABC', 'OPABC', 'OCABC', 'OAET', 'OFET', 'ORET', 'OBET', 'OPET', 'OCET'] # ALL THE MODELS MWAHAHAHA


    if CLEAN:
        df = clean(VITALS, NUM_UNIQUE_CCS, SUBSET_SIZE, LABEL, ALL_DATA, SAVE_CLEANED=True)
    else:
        if VITALS:
            filename = './models/{}/data_cleaned_{}.csv'.format(LABEL, LABEL)
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
    num_cores = 7 #int(os.getenv('SLURM_CPUS_PER_TASK')) # can't be 8, so making it 7
    
    jobs = [RF, XGB, LGBM, DT, KNN, ABC, ET] # took out BC
    
    metrics = ['OA', 'OF', 'OB', 'OR', 'OP', 'OC']
    
    out_queue = mp.Queue()
    score_queue = mp.Queue()
    workers = [ mp.Process(target=job, args=(out_queue, score_queue,) ) for job in jobs ]

    [work.start() for work in workers]
    [work.join() for work in workers]

    scores_indep = []
    for j in range(len(workers)):
        mdict.update(out_queue.get())
        scores_indep.extend(score_queue.get())
    
        
    
    # SuperLearner ensembling:
    #    mdict will have been updated with optimized parameters
    scores_indep.extend(SL())
    
    print(scores_indep)
    
    # TODO: save scores_indep
    pd.DataFrame(scores_indep).to_csv('./models/{}/ALLSCORES.csv'.format(LABEL), header=False, index=False, sep=',')
