import os
from os import listdir
from os.path import isfile, join
import glob
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc

#
#   ConfusionMatrix Class
#       Getting confusion matrix (sensitivity, specificity, precision(ppv), npv, accuracy, auc)
#
class ConfusionMatrix(object):

    def divide_group(self, temp_thres, y_pred, thres_method):
        divide_group = []
        if thres_method == 'min':
            #optimal cutoff1 : min distance
            temp = []
            for i,row in temp_thres.iterrows():
                delta = (1.0-row[1])**2 + (row[0])**2
                temp.append(delta)
            thres = temp_thres.iloc[temp.index(min(temp)),2]

        elif thres_method == "max" :
            #optimal cutoff2 : Youden
            temp = []
            for i,row in temp_thres.iterrows():
                delta = row[1] - row[0]
                temp.append(delta)
            thres = temp_thres.iloc[temp.index(max(temp)),2]
        
        else: #elif thres_method == "half" :
            thres = np.median(y_pred)

        for pred in y_pred:
            if pred >= thres :
                divide_group.append(1)
            else :
                divide_group.append(0)
        
        return divide_group, thres

    def __cacluate_confusion_matrix(self, pred_label, true_label):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(len(pred_label)):
            temp_pred_label = pred_label[i]
            temp_true_label = true_label[i]

            if temp_pred_label == 1 :
                if temp_true_label == 1 :
                    tp += 1
                elif temp_true_label == 0 :
                    fp += 1
            elif temp_pred_label == 0 :
                if temp_true_label == 1 :
                    fn += 1
                elif temp_true_label ==0 :
                    tn += 1 

        # init
        sensitivity = 0
        specificity = 0
        ppv = 0
        npv = 0
        accuracy = 0

        if tp+fn > 0:
            sensitivity = (tp)/float(tp+fn)

        if tn+fp > 0:
            specificity = (tn)/float(tn+fp)

        if tp+fp > 0:
            ppv = (tp)/float(tp+fp)   # positive predictive value (precision)

        if fn+tn > 0:
            npv = (tn)/float(fn+tn) # negative predictive value

        if tp+tn+fp+fn > 0:
            accuracy = (tp+tn)/float(tp+tn+fp+fn)
        
        formula = [sensitivity, specificity, ppv, npv, accuracy]
        return formula

    #
    # get_matrix2(..._
    #   y_true : array,  groud_truth
    #   pred : array,  prediction value
    #   return : [sensitivity, specificity, ppv, npv, accuracy, auc]
    def get_matrix2(self, y_true, y_pred):

        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label = 1)
        temp_thres = (pd.DataFrame([fpr, tpr, thresholds])).T

        try:
            # pred 가 1 만 있는 경우 등에서는 auc_score를 구할 수 없으며 이 경우 위에서 구한 temp_thres 값이 왜곡 되므로 여기서 걸러냄 (threshold = 0.5 강제 적용)
            auc_score = roc_auc_score(y_true, y_pred)
        except Exception as e:
            temp_thres[:] = 0.5

        pred_label, thres = self.divide_group(temp_thres, y_pred, 'max')
        formula = self.__cacluate_confusion_matrix(pred_label, y_true)

        # add auc
        auc_val = auc(fpr, tpr)
        formula.append(auc_val)

        return formula, pred_label, thres


    #
    # get_matrix(...)
    #
    #   df_prob_data (in)   : A DataFrame which includes Ground truth labels & predicted labels.  
    #                         e.g :
    #                                type         p0      p1      p2  ... pn
    #                            -----------------------------------------------
    #                            ground_truth    0.0     1.0     0.0     0.0
    #                                     CNN    0.35    0.20    ...
    #                            CTsizeCTraio    ...
    #                                     ...
    #                            -----------------------------------------------
    #
    #   save_path (in)      : file path to save the result.  if "None", it doen't save
    #
    #   return              : Confusion matrix as a DataFrame of pandas
    #
    def get_matrix(self, df_prob_data, save_path=None):
        matrix = []

        # y_true : labeled data
        y_true = df_prob_data.iloc[0][1:].tolist()

        for i, row in df_prob_data.iterrows():
            if i == 0:  # skip column title
                continue

            y_pred = row[1:].tolist()
                        
            matrix = self.get_matrix2(y_true, y_pred)

        # save result data frame
        df_matrix = pd.DataFrame(matrix, columns=['sensitivity', 'specificity', 'precision(ppv)', 'npv', 'accuracy', 'auc'])

        # add 'type' column without 'ground_truth' field
        new_cols = df_prob_data['type'].drop([0]).reset_index(drop=True).tolist()
        df_matrix.insert(loc=0, column='type', value=new_cols)

        # save the result
        if save_path is not None:
            if os.path.exists(save_path) == False:
                os.makedirs(save_path)
            file_name = save_path + 'confusion_matrix_formula.csv'
            df_matrix.to_csv(file_name, sep='\t', encoding='utf-8')


        return df_matrix

    # 확률로 된 y_pred를 최적의 threshold 계산하여 1, 0으로 리턴
    def get_pred_label(self, y_true, y_pred):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label = 1)
        temp_thres = (pd.DataFrame([fpr, tpr, thresholds])).T
        pred_label, thres = self.divide_group(temp_thres, y_pred, 'max')

        return pred_label, thres

#####################################

def main():

    # implement confusion matrix class
    confusion_matrix = ConfusionMatrix()

    
    # file name for result data
    result_file = 'confusion_matrix.csv'
    # check if the result csv file exist then delete.
    if os.path.isfile(result_file):
        os.remove(result_file)

    # load all .csv files
    data_files = [f for f in listdir('./') if f.endswith(".csv")]

    
    df_concat_mat = pd.DataFrame()
    file_cnt = 0
    for data_file in data_files:
        df_prob_data = pd.read_csv(data_file)

        # get a confusion matrix
        df_matrix = confusion_matrix.get_matrix(df_prob_data)

        # concat confusion matrix
        df_concat_mat = pd.concat([df_concat_mat, df_matrix])
        file_cnt += 1

    # caculate mean of all concated matrices
    df_concat_mat = df_concat_mat.groupby(['type']).mean()    
    print("\naverage result of %d input files:\n" % file_cnt)
    print(df_concat_mat)

    # save the result to file
    
    df_concat_mat.to_csv(result_file, sep='\t', encoding='utf-8')




if __name__ == '__main__':
    main()


