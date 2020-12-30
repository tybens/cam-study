# tybens 11/04/20
import pickle
import pandas as pd
import numpy as np

# data manip
from numpy import hstack, vstack, asarray
from sklearn.model_selection import train_test_split

# models
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


def calibrateMeta(df, mymodels, mymeta_model, plot=True):
    """ Calibrates the meta learner to output percentages instead of decision_function() confidence
    
    Parameters
    ----------
    df : pandas.DataFrame
        The preemptively cleaned (to match features of the data the model was fit on) data of patients to test calibration on
    mymodels : list(object)
        A list of model objects with a  .predict()  method,  .fit()  method (if full_fit=True)
    mymeta_model : object
        A model object that is to be used as a meta model with a  .partial_fit()  method (if partial_fit_meta=True), 
        a  predict_proba()  or  decision_function()  method (if prcPlot or rocPlot=True)
    plot : boolean, optional
        Default is False. Whether or not to plot a reliability curve.
    
    Returns
    -------
    object
        sklearn's CalibratedClassifier model as applied to mymeta_model. This will replace mymeta_model when making predictions
        in order to allow for probabilities to be made with predict_proba().
    """
    X = df.drop(['admit_binary'], axis=1)
    y = df['admit_binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1, shuffle=True)

    cc = CalibratedClassifierCV(mymeta_model, cv='prefit')
    
    def prepForMeta(X, models):
        meta_X = list()
        for model in models:
            if hasattr(model, 'predict_proba'):
                yhat = model.predict_proba(X)
            else:
                temp = model.predict(X)
                yhat = np.column_stack(([int(not z) for z in temp], temp))
            meta_X.append(yhat)
        meta_X = hstack(meta_X)
        return meta_X
    # predict
    meta_X_train = prepForMeta(X_train, mymodels)
    cc.fit(meta_X_train, y_train)
    
    meta_X_test = prepForMeta(X_test, mymodels)
    yhats = cc.predict_proba(meta_X_test)[:, 1]
    
    if plot:
        fop, mpv = calibration_curve(y_test, yhats, n_bins=10)
        # plot perfectly calibrated
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot model reliability
        plt.ylabel("fraction of positives")
        plt.xlabel("predicted fraction of positives")
        plt.plot(mpv, fop, marker='.')
        plt.show()
    
    return cc