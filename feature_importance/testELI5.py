import pickle
import pandas as pd
import numpy as np

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import eli5
from eli5.sklearn import PermutationImportance


from utils.SuperLearner import SuperLearner

    
def main(LABEL, OPTIMIZED):

    df = pd.read_csv('../data/data_vitals_cleaned.csv')
    superLearner = pickle.load(open('../models/{}/SuperLearner{}SL.sav'.format(LABEL, OPTIMIZED), 'rb'))
    X = df.drop(['admit_binary'], axis=1)
    y = df['admit_binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24, shuffle=True)

    # I have to re-initialize and refit the model because it needs to have a `.score` method that I only just now added to utils.SuperLearner
    superLearner = SuperLearner(superLearner.baseModels, superLearner.metaModel)
    superLearner.fit(X_train, y_train)
    
    perm = PermutationImportance(superLearner).fit(X_test, y_test)
    print(eli5.show_weights(perm, top=100, feature_names=np.ravel(X_test.columns)).data)
    
    
if __name__ == '__main__':
    
    LABEL = 'study50k_allmodels_vit'
    OPTIMIZED = 'OP'
    
    main(LABEL, OPTIMIZED)
