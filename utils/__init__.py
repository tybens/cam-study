# tybens 11/04/20
import pickle
import pandas as pd
import numpy as np

# data manip
from numpy import hstack, vstack, asarray
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
# evals
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, r2_score, recall_score, precision_score, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.utils import compute_class_weight


def superlearnerFitAndEval(X_train, X_test, y_train, y_test, mymodels, mymeta_model, model_name=None, beta=3, full_fit=False, optimized=False):
    """ evaluates a pair of models/metamodel on a given dataset, can fit the models if desired. 
    
    also,  scores  needs to be initalized in the scope, it will append the model name, test-sample-size, accuracy, ROC AUC, Precision-Recall AUC, fbeta, f-score.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The preemptively cleaned (to match features of the data the model was fit on) data of patients to preform
        the evaluation on
    mymodels : list(object)
        A list of model objects with a  .predict()  method,  .fit()  method (if full_fit=True)
    mymeta_model : object
        A model object that is to be used as a meta model that has a predict_proba() or  decision_function() method (if prcPlot or rocPlot=True)
    model_name : str
        Describe the model / data for formatting output of scores
    beta : float, optional
        Default is 3 (which is the number I've been using). This balances f-beta score between recall and precision towards recall
    optimized : boolean, optiona
        Default is False. prints evaluations of model if True. Also if True returns the fitted models to be pickled (saved).
        
    Returns
    -------
    list({float, str})
        A list of the various scoring procedures for the tested model
    object, optional
        If optimized = True. returns tuple of (list(scores), mymodels, mymeta_model) refit on the data.
    """
    BETA=beta
    
    # create a list of base-models
    def get_models():
        models = list()

        models.extend(mymodels)

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
                if full_fit:
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
        model = mymeta_model
        model.fit(X, y)
        return model

    # evaluate a list of models on a dataset
    def evaluate_models(X, y, models):
        for model in models:
            if hasattr(model, 'decision_function'):
                y_score = model.decision_function(X)
                rocscore = roc_auc_score(y, y_score)
                prcscore = average_precision_score(y, y_score)
                if optimized:
                    print('%s AUROC-score: %.3f' % (model.__class__.__name__, rocscore))
                    print('%s AUPRC-score: %.3f' % (model.__class__.__name__, prcscore))
            elif hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X)[:, 1]
                rocscore = roc_auc_score(y, y_score)
                prcscore = average_precision_score(y, y_score)
                if optimized:
                    print('%s AUROC-score: %.3f' % (model.__class__.__name__, rocscore))
                    print('%s AUPRC-score: %.3f' % (model.__class__.__name__, prcscore))
            else:
                y_score = model.predict(X)
                accuracy =  accuracy_score(y, y_score)
                fscore =f1_score(y, y_score)
                if optimized:
                    print('%s accuracy-score: %.3f', (model.__class__.__name__, accuracy))
                    print('%s f-score: %.3f', (model.__class__.__name__, fscore))
                
            

    # make predictions with stacked model
    def super_learner_predictions(X, models, meta_model):
        meta_X = list()
        for model in models:
            if hasattr(model, 'predict_proba'):
                yhat = model.predict_proba(X)
            else:
                temp = model.predict(X)
                yhat = np.column_stack(([int(not z) for z in temp], temp))
            meta_X.append(yhat)
        meta_X = hstack(meta_X)
        # predict
        return meta_model.predict(meta_X)
    
    # get models
    new_models = get_models()
    
    # fit if desired
    if full_fit:
        # get out of fold predictions
        meta_X, meta_y = get_out_of_fold_predictions(X_train, y_train, new_models)
        # re-fit base models only if full_fit was specified
        fit_base_models(X_train, y_train, new_models)
        # re-fit meta model
        mymeta_model = fit_meta_model(meta_X, meta_y)
        
    # evaluate base models
    evaluate_models(X_test, y_test, new_models)
    # evaluate meta model
    preds = super_learner_predictions(X_test, new_models, mymeta_model)
    
    tested_on="{}".format(X_test.shape)
    accuracy =  accuracy_score(y_test, preds)
    fscore =f1_score(y_test, preds)
    betascore = fbeta_score(y_test, preds, beta=BETA)
    recall = recall_score(y_test, preds)
    if optimized:
        print('SuperLearner accuracy: ', accuracy)
        print('SuperLearner f1-score: ', fscore)
        print('SuperLearner fbeta-score', betascore)
        print('SuperLearner recall: ', recall)
        print('SuperLearner precision: ', precision_score(y_test, preds))
    
    works = True # whether or not we can do thresholding!
    if hasattr(mymeta_model, 'decision_function'):
        y_score = superlearner_predict(X_test, new_models, mymeta_model, predict_proba=False, decision_function=True)
    elif hasattr(mymeta_model, 'predict_proba'):
        y_score = superlearner_predict(X_test, new_models, mymeta_model, predict_proba=True)
        try:
            y_score = y_score[:, 1]
        except IndexError:
            pass
    else:
        works = False
    
    if works:
        fpr, tpr, ROCthresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        roc_auc = round(roc_auc, 3)

        precision, recall, PRthresholds = precision_recall_curve(y_test, y_score)
        prc_auc = auc(recall, precision)
    else:
        roc_auc, prc_auc = 'only binary prediction', 'only binary prediction'
        
    scores = [model_name, roc_auc, prc_auc, accuracy, fscore, betascore, recall]
    scoresDict = {'OC': roc_auc, 'OP':prc_auc, 'OA': accuracy, 'OF':fscore, 'OR':recall, 'OB':betascore}
    if optimized:
        # if this is the optimized SuperLearner, return the models to be pickled
        return (scores, new_models, mymeta_model)
    else:
        m = model_name
        keys = [key for key in scoresDict]
        if m[:2] in keys:
            return scoresDict[m[:2]]
        else:
            print("something went wrong, model_name (m) didn't match scoresDict keys")
            return None


def superlearnerPredict(df, mymodels, mymeta_model, predict_proba=False, decision_function=False):
    """ use superlearner models/metamodel to predict given a cleaned and matched patient or patients (df)
    
    Parameters
    ----------
    df : pandas.DataFrame
        The preemptively cleaned (to match features of the data the model was fit on) data of patients to preform
        the evaluation on
    mymodels : list(object)
        A list of model objects with a  .predict()  method
    mymeta_model : object
        A model object that is to be used as a meta model with a predict_proba()  or  decision_function()  method if either parameter is True
    predict_proba : boolean, optional
        Default is False. If True, the meta model will call the  .predict_proba()  method and will return a two dimensional list of predictions
    decision_function : boolean, optional
        Default is False. If True, the meta model will call the .decision_function() method and will return a one dimensional list of floats
    
    Returns
    -------
    list({list(float), float, int})
        The predictions / prediction confidence for {predict_proba=True, decision_function=True, default}. 
    
    """
    try:
        X_test = df.drop(['admit_binary'], 1)
        y_test = df['admit_binary']
    except KeyError:
        X_test = df
        pass
    
    # create a list of base-models
    def get_models():
        models = list()

        models.extend(mymodels)

        return models

    # make predictions with stacked model
    def super_learner_predictions(X, models, meta_model):
        meta_X = list()
        for model in models:
            if hasattr(model, 'predict_proba'):
                yhat = model.predict_proba(X)
            else:
                temp = model.predict(X)
                yhat = np.column_stack(([int(not z) for z in temp], temp))
            meta_X.append(yhat)
        meta_X = hstack(meta_X)
        
        # predict
        if predict_proba:
            if hasattr(meta_model, 'predict_proba'):
                return meta_model.predict_proba(meta_X)
            else:
                print("can't predict_proba with this meta model, returning binary prediction")
        elif decision_function:
            if hasattr(meta_model, 'decision_function'):
                return meta_model.decision_function(meta_X)
            else:
                print("can't decision_function with this meta model, returning binary prediction")      
    
        return meta_model.predict(meta_X)

        
    # get models
    models = get_models()
    
    # predict using superlearner    
    pred = super_learner_predictions(X_test, models, mymeta_model)
    
    return pred