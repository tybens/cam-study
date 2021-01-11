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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

class SuperLearner:
    """ SuperLearner object for fitting and making predictions from basemodels and metamodels
    
    Parameters
    ----------
    baseModels : list(object)
        list of model objects that have `.fit` and a `.predict` parameter (ideally
        `.predict_proba`)
    metaModel : object
        a model object that will make predictions from the "meta data" given by the 
        predictions of basemodels. This also must have a `.fit` and a `.predict` 
        parameter (really ideally `.predict_proba`)
        
      """
    
    def __init__(self, baseModels, metaModel, model_name='SL', is_fit=False):
        self.baseModels = baseModels
        self.metaModel = metaModel
        self.model_name = model_name
        self.is_fit = is_fit
    
    def predict(self, X):
        # compute prediction probas for list of instances
        return self._super_learner_predictions(X)

    def predict_proba(self, X):
        # compute prediction probas for list of instances
        return self._super_learner_predictions(X, predict_proba=True)

    def decision_function(self, X):
        return self._super_learner_predictions(X, decision_function=True)

    def classify(self, inputs):
        # thresholded classification (threshold 0)
      
        return np.sign(self.predict(inputs))
    
    
    def fit(self, X, y):
        """ Fit the SuperLearner """
        # fit the base models on the dataset
        self.models = self._fit_base_models(X, y)
        # get "meta" predictions to feed to the metamodel
        meta_X, meta_y = self._get_out_of_fold_predictions(X, y)
        # use "meta" predictions to fit the metamodel
        self.metaModel = self._fit_meta_model(meta_X, meta_y)
        
        self.is_fit = True
        
        
    def get_params(self, deep=False):
        # return parameters
        return { 'baseModels': self.baseModels, 'metaModel': self.metaModel, 'model_name':self.model_name,'is_fit': self.is_fit}
    
    def scores(self, X_test, y_test):
        
        preds = self._super_learner_predictions(X_test)
    
        accuracy =  accuracy_score(y_test, preds)
        fscore = f1_score(y_test, preds)
        recall = recall_score(y_test, preds)
        
        # non threshold scores:
        works = True 
        if hasattr(self.metaModel, 'decision_function'):
            y_score = self.decision_function(X_test)
        elif hasattr(self.metaModel, 'predict_proba'):
            y_score = self.predict_proba(X_test)
            try:
                y_score = y_score[:, 1]
            except IndexError:
                pass
        else:
            works = False

        if works:
            # AUROC
            fpr, tpr, ROCthresholds = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            # AUPRC
            precision, recalls, PRthresholds = precision_recall_curve(y_test, y_score)
            prc_auc = auc(recalls, precision)
        else:
            # else, save that these scores aren't possible (only binary prediction)
            roc_auc, prc_auc = 'only binary prediction', 'only binary prediction'

        scores = [self.model_name, roc_auc, prc_auc, accuracy, fscore, recall]
        print(scores)
        
        return scores
    
    def basemodelScores(self, X, y):
        """ evaluate the base models individually on the data set"""
        model_name = self.model_name
        basemodelScores = []
        for model in self.baseModels:
            if hasattr(model, 'decision_function'):
                y_score = model.decision_function(X)
                rocscore = roc_auc_score(y, y_score)
                prcscore = average_precision_score(y, y_score)
                print('%s AUROC-score: %.3f' % (model.__class__.__name__, rocscore))
                print('%s AUPRC-score: %.3f' % (model.__class__.__name__, prcscore))
                basemodelScores.append([model_name+model.__class__.__name__, rocscore, prcscore])
            elif hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X)[:, 1]
                rocscore = roc_auc_score(y, y_score)
                prcscore = average_precision_score(y, y_score)
                print('%s AUROC-score: %.3f' % (model.__class__.__name__, rocscore))
                print('%s AUPRC-score: %.3f' % (model.__class__.__name__, prcscore))
                basemodelScores.append([model_name+model.__class__.__name__, rocscore, prcscore])
            else:
                y_score = model.predict(X)
                accuracy =  accuracy_score(y, y_score)
                fscore =f1_score(y, y_score)
                print('%s accuracy-score: %.3f', (model.__class__.__name__, accuracy))
                print('%s f-score: %.3f', (model.__class__.__name__, fscore))
                basemodelScores.append([model_name+model.__class__.__name__, accuracy, fscore])
        return basemodelScores
    
    def _get_out_of_fold_predictions(self, X, y):
        """ Function to use base models to get out of fold predictions
        
        these predictions are to be fed to fit the metamodel
        
        """
        models = self.baseModels
        
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
    
    def _super_learner_predictions(self, X, predict_proba=False, decision_function=False):
        """ use the metamodel to predict on the data set """
        models = self.baseModels
        metaModel = self.metaModel
        
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
            if hasattr(metaModel, 'predict_proba'):
                return metaModel.predict_proba(meta_X)
            elif hasattr(metaModel, 'decision_function'):
                #print("can't predict proba with {}, returning decision function".format(metaModel))
                return metaModel.decision_function(meta_X)
            else:
                print("can't predict_proba or decision function with {}, happened in {}, returning binary prediction".format(self.metaModel, self.model_name))
        elif decision_function:
            if hasattr(metaModel, 'decision_function'):
                return metaModel.decision_function(meta_X)
            elif hasattr(metaModel, 'predict_proba'):
                #print("can't decision function with {}, returning predict proba".format(metaModel))
                return metaModel.predict_proba(meta_X)[:, 1]
            else:
                print("can't decision_function or predict proba with {}, happened in {}, returning binary prediction".format(self.metaModel, self.model_name))      
        
        return metaModel.predict(meta_X)
    
    
    def _fit_base_models(self, X, y):
        """ fit the base models """
        models = self.baseModels
        new_models = []
        for model in models:
            model.fit(X, y)
            new_models.append(model)
        return new_models

    def _fit_meta_model(self, X, y):
        """ fit the meta model """
        model = self.metaModel
        model.fit(X, y)
        return model
