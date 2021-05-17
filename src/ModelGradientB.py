import os
import random
import numpy as np
import pandas as pd
import _pickle as cpickle
import warnings
from tqdm import tqdm
import xgboost as xgb
from skimage.io import imread, imsave
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg19 import VGG19
# from keras.applications.resnet50 import RESNET50

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import sys
import seaborn as sns

from InputDataXgb import InputDataXgb
from ConfusionMatrix import ConfusionMatrix
import defs
from sklearn.ensemble import GradientBoostingClassifier

class ModelGradientB():
    def __init__(self):
        self.input_data = InputDataXgb()


    # @ cal : calibration 적용 여부 => patch_selector iterative training 할 때 만 False 로 사용
    def train(self, train_x, val_x, train_y, val_y, model_name_tosave, pt_model_file, cal=True):
        clf = GradientBoostingClassifier()
        
        try:
            clf.fit(train_x, train_y)
        except Exception as e: # work on python 3.x
            print('error : train() : clf.fit()  except:', str(e))
            return None

        model_file = defs.model_save_path + model_name_tosave

        # calibration  (최종 모델 저장할때만 적용)
        if defs.calibration:
            if cal:
                clf = CalibratedClassifierCV(clf, cv=3, method='isotonic')
                # clf = CalibratedClassifierCV(clf, cv=3, method='sigmoid')
                clf.fit(train_x, train_y)
                cpickle.dump(clf, open(model_file,'wb'))    # CalibratedClassifierCV는 save_mode()이 없다
            else:
                clf.save_model(model_file)  # patch selector 저장할 때는 xgb coninuous learning을 위해 cpickle이 아닌 model_save()로 저장
        else:
            ## test
            ##clf.save_model(model_file)
            cpickle.dump(clf, open(model_file,'wb'))

        return clf


    def test(self, test_x, test_y, test_ids, model_name):
        model_file = defs.model_save_path + model_name
        (auc, sensitivity, specificity, ppv, npv, accuracy, auc2, df_res) = self._test_from_df(model_file, 
                                                                                        test_x,
                                                                                        test_y,
                                                                                        test_ids,
                                                                                        load_dump_data=False)

        return auc, sensitivity, specificity, ppv, npv, accuracy, df_res


    def _test_from_df(self, model_file, test_x, test_y, test_ids, load_dump_data=True):
        ## test
        ## if defs.calibration:
        ##     recur_clf = cpickle.load(open(model_file, 'rb'))
        ## else:
        ##     recur_clf = xgb.XGBClassifier()
        ##     recur_clf.load_model(model_file)  # load data
        recur_clf = cpickle.load(open(model_file, 'rb'))
            
        recur_prob = recur_clf.predict_proba(test_x)[:,1]
        df = pd.DataFrame(columns=['ids', 'recur_prob', 'pred_label', 'label', 'threshold'])
        df['label'] = test_y
        df['ids'] = test_ids
        df['recur_prob'] = recur_prob

        # confusion matrix
        confusion_matrix = ConfusionMatrix()
        conf_matrix, pred_label, thres = confusion_matrix.get_matrix2(df.label, df.recur_prob)
        df['pred_label'] = pred_label
        df['threshold'] = thres

        auc = roc_auc_score(df.label, df.recur_prob)

        return (auc, conf_matrix[0], conf_matrix[1], conf_matrix[2], conf_matrix[3], conf_matrix[4], conf_matrix[5], df)
