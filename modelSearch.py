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
from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, average_precision_score
from sklearn.utils import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV
import optuna # hyperparam ~unsupervised tuning 
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING) # comment this output optuna progress
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
from mlens.ensemble import SuperLearner

# local relative imports:
from utils.cleaning import clean, str2bool # clean for processing data, str2bool for command line operability

class Optimizer:
    """ Optimization suite as adapted from optuna's framework
    """
    def __init__(self, metric, trials=50):
        self.metric = metric
        self.trials = trials
        self.sampler = TPESampler(seed=0)
        
    def objective(self, trial):
        model = create_model(trial)
        scorer_dict = {'OA': 'accuracy',
                      'OP': 'average_precision',
                      'OF': 'f1',
                      'OR': 'roc_auc'}
        return np.mean(cross_val_score(model, X_train, y_train, scoring=scorer_dict[self.metric], cv=3))
                
            
            
    def optimize(self):
        study = optuna.create_study(direction="maximize", sampler=self.sampler)
        study.optimize(self.objective, n_trials=self.trials)
        return study.best_params, study.best_value
    
def getScores(model, X_test, y_test, crossvalscore):
    """ Function for scoring the provided model on validation set
    
    Arguments
    ---------
    model: object
        A callable model object that has .fit, .predict, and ideally .predict_proba methods
        
    Returns
    -------
    list({str, float})
        a list of scores for the optimized model passed as an argument    
    """
    AUROC, AUPRC = None, None
    
    preds = model.predict(X_test)
    
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
    rec = recall_score(y_test, preds)
    
    # scores = [label of model, AUROC, AUPRC, accuracy, f1_score, recall, crossvalscore]
    scores = [m, AUROC, AUPRC, acc, f1, rec, crossvalscore]
    print(scores)
    return scores
    
def optimizeAndSaveScores(model, m):
    """ Sets up optimization suite, optimizes the model, tests and returns score and optimized params
    
    The optimization scoring metric is obtained from the str label, m, and optuna is used to optimally tune the
    hyperparameters. 'KNN' doesn't have a 'random_state' apparently.
    
    Arguments
    ---------
    model: object
        A callable model object that has .fit, .predict, and ideally .predict_proba methods
    m: str
        String label of the model. i.e. 'OAKNN', 'optimized on accuracy KNeighborsClassifier model'
    
    Returns
    -------
    tuple(dict(), list({str, float}))
        First item is a dict returning optimized params, second item is a list of scores for
        the optimized model.
    
    """
    global preds
    print('-'*40)
    print(m)
    optimizer = Optimizer(m[:2])
    model_params, crossvalscore = optimizer.optimize()
    if m[-3:] != 'KNN':
        model_params['random_state'] = RAND_STATE
    # after optimized, fit on all train data and test on validation set (X_test, y_test)
    optimizedModel = model(**model_params)
    print(f"'{m}'" + " : " + str(optimizedModel))
    optimizedModel.fit(X_train, y_train)
    retscore = getScores(optimizedModel, X_test, y_test, crossvalscore)
    return {m: model(**model_params)}, retscore
    
def baselineThenOptimize(model, label):
    """ Train and test an unoptimized 'baseline' then optimize a model for each metric.
    
    Arguments
    ----------
    model: object
        A callable model object that has .fit, .predict, and ideally .predict_proba methods
    label: string
        a string representation of the model and scoring metric. i.e. 'OAKNN', 'optimized 
        on accuracy KNeighborsClassifier model'
    
    Returns
    -------
    tuple(list(list({str, float})), dict)
        a tuple with the first item being a two-dimensional array of scores for each
        model optimized on each metric. The second item is a dictionary with {label: modelparams}
        for each model, returning optimized params.
    """
    global m
    retdict = {}
    retscores = []
    if label in TOTEST:
        m = label
        if m[-3:] != 'KNN':
            baselineModel = model(random_state=RAND_STATE)
        else:
            baselineModel = model()
        baselineModel.fit(X_train, y_train)
        
        retscores.append(getScores(baselineModel, X_test, y_test, crossvalscore=0))
        
    for metric in metrics:
        m = metric+label
        if m in TOTEST:
            paramdict, retscore = optimizeAndSaveScores(model, m)
            retdict.update(paramdict)
            retscores.append(retscore)
 
    return retscores, retdict

    
def RF(out_queue, score_queue):
    global create_model, y_test, preds
    model = RandomForestClassifier
    label = 'RF'
   
    def create_model(trial):
        max_depth = trial.suggest_int("max_depth", 30, 40)
        n_estimators = trial.suggest_int("n_estimators", 550, 650)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        weights = weight_dict
        model = RandomForestClassifier(min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, max_depth=max_depth, random_state=RAND_STATE, class_weight=weights)
        return model
    
    retscores, retdict = baselineThenOptimize(model, label)
    
    score_queue.put(retscores)
    out_queue.put(retdict)

def XGB(out_queue, score_queue):
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
    
    retscores, retdict = baselineThenOptimize(model, label)
    
    score_queue.put(retscores)
    out_queue.put(retdict)
    

def LGBM(out_queue, score_queue):
    global create_model, y_test, preds
    model = LGBMClassifier
    label = 'LGBM'

    def create_model(trial):
        num_leaves = trial.suggest_int('num_leaves', 2, 5000) 
        max_depth = trial.suggest_int('max_depth', 2, 100) 
        n_estimators = trial.suggest_int('n_estimators', 10, 500) 
        subsample_for_bin = trial.suggest_int('subsample_for_bin', 2000, 300_000) 
        min_child_samples = trial.suggest_int('min_child_samples', 20, 500) 
        reg_alpha = trial.suggest_uniform('reg_alpha', 0.0, 1.0) 
        colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.6, 1.0) 
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-0)   
        model = LGBMClassifier(learning_rate=learning_rate, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, subsample_for_bin=subsample_for_bin, num_leaves=num_leaves, max_depth=max_depth, n_estimators=n_estimators, min_child_samples=min_child_samples, random_state=RAND_STATE, is_unbalance=True)
        
        return model
    
    retscores, retdict = baselineThenOptimize(model, label)
    
    score_queue.put(retscores)
    out_queue.put(retdict)

def LR(out_queue, score_queue):
    global create_model, y_test, preds
    if 'LR' in TOTEST:
        m = 'LR'
        lr = LogisticRegression(random_state=RAND_STATE)
        lr.fit(X_train, y_train)
        preds = lr.predict(X_test)
        


def DT(out_queue, score_queue):
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
    
    retscores, retdict = baselineThenOptimize(model, label)
    
    score_queue.put(retscores)
    out_queue.put(retdict)

    
def BC(out_queue, score_queue):
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
    global create_model, y_test, preds
    model = KNeighborsClassifier
    label = 'KNN'
    
    sampler = TPESampler(seed=0)
    def create_model(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 2, 25)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        return model
    
    retscores, retdict = baselineThenOptimize(model, label)
    
    score_queue.put(retscores)
    out_queue.put(retdict)

def ABC(out_queue, score_queue):
    global create_model, y_test, preds
    model = AdaBoostClassifier
    label = 'ABC'
    
    def create_model(trial):
        n_estimators = trial.suggest_int("n_estimators", 2, 500)
        learning_rate = trial.suggest_uniform('learning_rate', 0.0005, 1.0)
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=RAND_STATE)
        return model
    
    retscores, retdict = baselineThenOptimize(model, label)
    
    score_queue.put(retscores)
    out_queue.put(retdict)

def ET(out_queue, score_queue):
    global create_model, y_test, preds
    model = ExtraTreesClassifier
    label = 'ET'
    
    def create_model(trial):
        n_estimators = trial.suggest_int("n_estimators", 2, 500)
        max_depth = trial.suggest_int("max_depth", 2, 20)
        model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=RAND_STATE, class_weight=weight_dict)
        return model
    
    retscores, retdict = baselineThenOptimize(model, label)
    
    score_queue.put(retscores)
    out_queue.put(retdict)


def SL_fit_and_save(superLearner, crossvalscore, model_name):
    """ Function for final testing and saving of optimized super learner ensemble
    
    The final scores of the optimized model ensemble are obtained, the superlearner is pickled 
    as a mlens SuperLearner object in the models/{LABEL} directory. 
    
    Arguments
    ---------
    superLearner : object
        a SuperLearner mlens object
        
    Returns
    -------
    list({str, float})
        A list of the scores of this particular ensemble. [label of model, AUROC, AUPRC, crossvalscore]
    
    """
    
    # fit the optimized hyperparameter superlearner on the training set
    superLearner.fit(X_train, y_train)
    # score it based on validation set
    preds = superLearner.predict(X_test)[:, 1]
    roc_score = roc_auc_score(y_test, preds)
    prc_score = average_precision_score(y_test, preds)
    
    scores = [model_name, roc_score, prc_score, crossvalscore]
    scores.append(crossvalscore)
    
    # save the SuperLearner object
    filename = './models/{}/SuperLearner{}.sav'.format(LABEL, model_name)
    pickle.dump(superLearner, open(filename, 'wb'))
    
    
    return scores
    
def SL():
    """Function for optimization of super learner ensembles
    
    Optuna is used to test combinations of optimized models (only if they are explicitly contained in TOTEST).
    For each metric that is in `metrics`, a super learner ensemble is tested and optimized and saved.
    After optimization, `SL_fit_and_save` is called for final testing and saving of models and scores.
        
    Returns
    -------
    list(list({str, float}))
        A two dimensional list of the scores for each super learner ensemble. 
        Each row has this structure: [label of model, AUROC, AUPRC, accuracy, f1_score, recall]
    
    """
    scores = []
    sampler = TPESampler(seed=0)
    label = 'SL'

    def create_model(trial):
        model_names = list()
        optimized_models = ['OARF', 'OFRF', 'OPRF', 'ORRF', 'OAXGB', 'OFXGB', 'OPXGB', 'ORXGB', 'OALGBM', 'OFLGBM', 'OPLGBM', 'ORLGBM', 'OADT', 'OFDT', 'OPDT', 'ORDT', 'OAKNN', 'OFKNN', 'OPKNN', 'ORKNN', 'OAABC', 'OFABC', 'OPABC', 'ORABC', 'OAET', 'OFET', 'OPET', 'ORET']
        models_list = ['RF', 'XGB', 'LGBM', 'DT', 'KNN', 'BC'] + [i for i in TOTEST if i in optimized_models] + ['LR', 'ABC', 'SGD', 'ET', 'MLP', 'GB', 'RDG', 'PCP', 'PAC']

        head_list = ['RF', 'XGB', 'LGBM', 'DT', 'KNN', 'BC', 'LR', 'ABC', 'SGD', 'ET', 'MLP', 'GB', 'RDG', 'PCP', 'PAC']
        n_models = trial.suggest_int("n_models", 2, 10)
        for i in range(n_models):
            model_item = trial.suggest_categorical('model_{}'.format(i), models_list)
            if model_item not in model_names:
                model_names.append(model_item)

        folds = trial.suggest_int("folds", 2, 8)

        models = list()
        for item in model_names:
            models.append(mdict[item])
        head = trial.suggest_categorical('head', head_list)
        metaModel = copy.deepcopy(mdict)[head]
        
        superLearner = SuperLearner()
        superLearner.add(models)
        superLearner.add_meta(metaModel, proba=True)
        
        return superLearner
    
    for metric in metrics:
        m = metric+label
        
        if m in TOTEST:
            def objective(trial):
                superLearner = create_model(trial)
                scorer_dict = {'OA': 'accuracy',
                      'OP': 'average_precision',
                      'OF': 'f1',
                      'OR': 'roc_auc'}
                score = np.mean(cross_val_score(superLearner, X_train, y_train, scoring=scorer_dict[metric], cv=3, n_jobs=-1))
                
                return score
            # initialize study
            study = optuna.create_study(direction="maximize", sampler=sampler)
            # optimize, using defined objective with specified metric for scoring
            study.optimize(objective, n_trials=50)
        
            params = study.best_params

            head = params['head']
            folds = params['folds']
            del params['head'], params['n_models'], params['folds']
            result = list()
            for key, value in params.items():
                if value not in result:
                    result.append(value)
                    
            # make SuperLearner object from result of optimization
            baseModels = list()
            for item in result:
                baseModels.append(mdict[item])
                
            metaModel = copy.deepcopy(mdict[head])
            superLearner = SuperLearner()
            superLearner.add(baseModels)
            superLearner.add_meta(metaModel, proba=True)
            
            # crossvalidation best score:
            crossvalscore = study.best_value
            
            # fit the optimized SuperLearner and save it
            scores.extend(SL_fit_and_save(superLearner, crossvalscore, m)) # extend because SL_fit_and_save returns 2-d array
            
    return scores
    
if __name__ == '__main__':     
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--CLEAN", "-c", type=str2bool, nargs='?', const=True, default=False, help="Boolean for cleaning the data?")
    parser.add_argument("--NUM_UNIQUE_CCS", "-nccs", type=int, help="Number of unique CCs to be had when cleaning the data, only relevant if clean is true")
    parser.add_argument("--SUBSET_SIZE", "-ss", type=int, help="How much of the 220,000 patients do you want to work with?")
    parser.add_argument("--RAND_STATE", "-rs", type=int, help="Change the random_state through with np and sklearn work")
    parser.add_argument("--VITALS", "-v", type=str2bool, nargs='?', const=True, default=False, help="boolean for whether to work with vitals or not")
    parser.add_argument("--SAVE_CLEANED", "-sav", type=str2bool, nargs='?', const=True, default=False, help="boolean for whether to save the cleaned file, only relevant if clean is true")
    parser.add_argument("--LABEL", "-l", type=str, help="str what to label the saved file with, only relevant if clean is true")
    parser.add_argument("--ALL_DATA", "-ad", type=str2bool, nargs='?', const=True, default=False, help="clean all patients of specificied data, only relevant if clean is true")
    parser.add_argument("--WARM_START", "-ws", type=str2bool, nargs='?', const=True, default=False, help="whether or not to skip the independent base model optimization and just optimize super learner")
    args = parser.parse_args()
    print("ARGUMENTS PASSED: {}".format(args))
    
    CLEAN = args.CLEAN
    NUM_UNIQUE_CCS = args.NUM_UNIQUE_CCS
    SUBSET_SIZE = args.SUBSET_SIZE
    RAND_STATE = args.RAND_STATE
    VITALS = args.VITALS
    SAVE_CLEANED = args.SAVE_CLEANED
    LABEL = args.LABEL
    ALL_DATA = args.ALL_DATA
    WARM_START = args.WARM_START
    
    if CLEAN:
        df = clean(VITALS, NUM_UNIQUE_CCS, SUBSET_SIZE, LABEL, ALL_DATA, SAVE_CLEANED=True)
    else:
        filename = './data/data_cleaned.csv' if not VITALS else './data/data_vitals_cleaned.csv'
        df = pd.read_csv(filename)
    
    #Defining the independent variables and dependent variables
    X = df.drop('admit_binary', axis=1)
    y = df['admit_binary']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=RAND_STATE, shuffle=True)
    if not ALL_DATA:
        X_train = X_train[:SUBSET_SIZE]
        y_train = y_train[:SUBSET_SIZE]
    
    print("X_train.shape: ", X_train.shape, " X_test.shape: ",  X_test.shape)
    print("y_train.shape: ", y_train.shape, " y_test.shape: ", y_test.shape)

    print("# OF FEATURES: {}  |  # OF PATIENTS: {}".format(len(df.columns)-1, len(df)))
    print("RAND_STATE: {}   ".format(RAND_STATE))

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

   
    # MULTIPROCESSING:
    num_cores = 7 #int(os.getenv('SLURM_CPUS_PER_TASK')) # can't be 8, so making it 7
    
    jobs = [RF, XGB, LGBM, DT, KNN, ABC, ET] # took out BC (and LR)
    
    metrics = ['OP', 'OR'] # possible metrics: 'OA', 'OF', 'OP', 'OR'
    
    scores_indep = []
    
    if not warm_start:   
        models = ['RF', 'XGB', 'LGBM', 'DT', 'KNN', 'ABC', 'ET']
        TOTEST = [metric+model for metric in metrics for model in models]
        print("TOTEST: {}".format(TOTEST))
        out_queue = mp.Queue()
        score_queue = mp.Queue()
        workers = [ mp.Process(target=job, args=(out_queue, score_queue,) ) for job in jobs ]

        [work.start() for work in workers]
        [work.join() for work in workers]

        for j in range(len(workers)):
            mdict.update(out_queue.get())
            scores_indep.extend(score_queue.get())
    else:
        models = ['SL']
        TOTEST = [metric+model for metric in metrics for model in models]
        print("TOTEST: {}".format(TOTEST))
        
        # this is the output of 50k patient optimization. 
        warm_start_dict = {'OPABC': AdaBoostClassifier(learning_rate=0.3539350274698853, 
                                                       n_estimators=437),
                           'ORABC': AdaBoostClassifier(learning_rate=0.33959804182520703, 
                                                       n_estimators=458),
                           'ORKNN': KNeighborsClassifier(n_neighbors=25),
                           'OPKNN': KneighborsClassifier(n_neighbors=25),
                           'ORRF': RandomForestClassifier(max_depth=39, min_samples_leaf=2,
                                                          n_estimators=608, random_state=24),
                           'OPRF': RandomForestClassifier(max_depth=40, min_samples_leaf=2, 
                                                          n_estimators=597, random_state=24), 
                           'ORET': ExtraTreesClassifier(max_depth=20, n_estimators=299, 
                                                        random_state=24), 
                           'OPET':ExtraTreesClassifier(max_depth=20, n_estimators=299, 
                                                       random_state=24), 
                           'ORDT':DecisionTreeClassifier(max_depth=23, min_samples_leaf=2, min_weight_fraction_leaf=0.0036809118931240043, random_state=24), 
                           'OPDT':DecisionTreeClassifier(max_depth=23, min_samples_leaf=2, min_weight_fraction_leaf=0.0036809118931240043, random_state=24)
                          }
        mdict.update(warm_start_dict)
        
    
    # SuperLearner ensembling:
    #    mdict will have been updated with optimized parameters
    scores_indep.extend(SL())
    
    print(scores_indep)
    
    # TODO: save scores_indep
    pd.DataFrame(scores_indep).to_csv('./models/{}/ALLSCORES.csv'.format(LABEL), header=False, index=False, sep=',')
    
    print("COMPLETED!")
    print("|"*40)
