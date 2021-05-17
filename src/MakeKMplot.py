import os 
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import k_fold_cross_validation
from lifelines.statistics import logrank_test
import _pickle as cpickle

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 100})
from sklearn.metrics import roc_curve, roc_auc_score
import datetime
from scipy import interp

from ConfusionMatrix import ConfusionMatrix

import defs

# 보라님 clinical_survival 프로젝트에서 가져온 클래스를 수정함
class MakeKMplot(object):
    def __init__(self, group_dividing_method=['min', 'max', 'half']):
        self.group_dividing_method = group_dividing_method
        self.conf_matrix = ConfusionMatrix()
    
    def __MakeKMplotinputdf(self, y_true, dfs, pred_label):
        event_tst = np.array(y_true).astype(int) 
        kmplot_input = pd.DataFrame([list(event_tst),list(dfs)]).T
        kmplot_input.columns = ["event","duration"]

        predicted_group = []
        for label in pred_label :
            if label == 0 :
                predicted_group.append("non_recur")
            else :
                predicted_group.append("recur")
        kmplot_input["pred_group_label"] = predicted_group

        return kmplot_input
    
    def __MakeKMplot(self, kmplot_input, group_method, outfile):
        kmf = KaplanMeierFitter()
        ax = plt.subplot(111)

        # kmplot raw input save
        kmplot_raw = kmplot_input.copy()
        kmplot_raw.loc[kmplot_raw['pred_group_label'] == 'non_recur', 'pred_group_label'] = 0
        kmplot_raw.loc[kmplot_raw['pred_group_label'] == 'recur', 'pred_group_label'] = 1
        out = outfile.split('.png')[0] + '_' + group_method + '_input.csv'
        kmplot_raw.to_csv(out)

        ix = (kmplot_input["pred_group_label"] == 'non_recur')
        T = kmplot_input["duration"]
        E = kmplot_input["event"]

        if T.empty or E.empty:
            print("warning in __MakeKMplot() : T or E is empty")
            return

        try:
            kmf.fit(T[~ix], E[~ix], label='recur pred : '+str(len(kmplot_input.loc[kmplot_input["pred_group_label"] == "recur"])))
            kmf.plot(ax=ax, ci_show=False)
            kmf.fit(T[ix], E[ix], label='non recur pred : '+str(len(kmplot_input.loc[kmplot_input["pred_group_label"] == "non_recur"])))
            kmf.plot(ax=ax, ci_show=False)
        except Exception as e: # work on python 3.x
            print('error in __MakeKMplot() :' + str(e))
            print('ix=\n', ix)
            print('duration=\n', T)
            print('event=\n', E)
            return

        p = logrank_test(T[~ix],T[ix],event_observed_A=E[~ix],event_observed_B=E[ix]).p_value

        with open('./result/km_pval'+'.txt','at') as f:
            f.writelines('\n' + str(datetime.datetime.now()) + '\n')
            to_write = " | ".join(map(str,[group_method]+[p]))
            f.write(to_write+"\n")

        plt.title('KM plot')
        outfile = outfile.split('.png')[0] + '_' + group_method + '.png'
        plt.savefig(outfile)

        plt.clf()

    # KM Plot 저장 : 각각 dividing methods (min,max,half) 별로.
    def MakeKMplot(self, y_true, dfs, y_pred, outfile):
        fpr , tpr, thresholds = roc_curve(y_true, y_pred, pos_label = 1)
        temp_thres = (pd.DataFrame([fpr,tpr,thresholds])).T

        for group_method in self.group_dividing_method:
            pred_label, _ = self.conf_matrix.divide_group(temp_thres, y_pred, group_method)
            kmplot_input = self.__MakeKMplotinputdf(y_true, dfs, pred_label)
            self.__MakeKMplot(kmplot_input, group_method, outfile)


    def draw_roc_curve(self, y_true, y_pred, outfile):
        # test
        print('draw_roc_curve() ...')
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label = 1)

        try:
            auc_score = roc_auc_score(y_true, y_pred)
        except Exception as e:
            print('cannot draw_roc_curve because : ')
            print(e)
            return

        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        plt.title('ROC curve')
        plt.savefig(outfile)
        plt.clf()
        #plt.show()

    def draw_roc_curve_avg(self, tprs, base_fpr, avg_auc, outfile):
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std

        lw = 2
        plt.plot(base_fpr, mean_tprs, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % avg_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        plt.title('ROC curve')
        plt.savefig(outfile)
        plt.clf()

        # 논문용.  k fold roc curve x100, x400, both를 합쳐서 그림 (모델 불러와서..)
    def draw_roc_curve_all(self, input_data, model, args, anno_file):

        # load roc result raw file to draw
        # tprs, base_fpr, avg_auc = cpickle.load(open('./result/x100_roc_data.cpickle', 'rb'))
        # self._plot_roc(tprs, base_fpr, avg_auc, color='mediumblue', label='(A) Architectural network ')

        # tprs, base_fpr, avg_auc = cpickle.load(open('./result/x400_roc_data.cpickle', 'rb'))
        # self._plot_roc(tprs, base_fpr, avg_auc, color='green', label='(B) Tumor Cell network ')

        # tprs, base_fpr, avg_auc = cpickle.load(open('./result/x400_x100_roc_data.cpickle', 'rb'))
        # self._plot_roc(tprs, base_fpr, avg_auc, color='darkorange', label='(C) = (A)+(B) Ensemble ')

        tprs, base_fpr, avg_auc = cpickle.load(open('./result/x400_x100_full_roc_data.cpickle', 'rb'))
        self._plot_roc(tprs, base_fpr, avg_auc, color='darkorange', label='ROC curve ')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        plt.title('ROC curve')
        plt.savefig('./result/roc_all.png')
        plt.savefig('./result/roc_all.svg')
        plt.savefig('./result/roc_all.pdf')
        plt.clf()


    def _plot_roc(self, tprs, base_fpr, avg_auc, color, label):
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std

        lw = 2
        plt.plot(base_fpr, mean_tprs, color=color,
                lw=lw, label= label + '(area = %0.2f)' % avg_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    # 논문용.  k fold roc curve x100, x400, both를 합쳐서 그림 (모델 불러와서..)
    def save_roc_raw_data(self, input_data, model, args, anno_file):
        # data set
        load_dump_data = args['do_load_dump_train_kfold']
        kfold = 5
        save_crop_path = None
        trn_x_list, val_x_list, tst_x_list, trn_y_list, val_y_list, tst_y_list, tst_id_list = input_data.get_train_val_tst_list_kfold(anno_file,
                                                                                                                                kfold=kfold,
                                                                                                                                use_x=args['use_x'],
                                                                                                                                patch_sel=args['do_train_patch_sel'],
                                                                                                                                patch_thres_idx=args['patch_thres_idx'],
                                                                                                                                load_dump_data=load_dump_data,
                                                                                                                                save_crop_path=save_crop_path)

        if len(trn_x_list) == 0:
            print('error in train_kfold() : len(trn_x_list) == 0')
            return exit(0)

        auc_list = []
        sensi_list = []
        speci_list = []
        ppv_list = []
        npv_list = []
        acc_list = []
        df_pred_ext = pd.DataFrame(columns=['ids', 'recur_prob', 'pred_label', 'label'])
        test_y_ext = np.zeros((1), dtype=np.uint8)
        test_y_ext = test_y_ext[:0]
        test_id_ext = []
        tprs = []
        base_fpr = np.linspace(0, 1, 101)

        for i in range(len(trn_x_list)):
            model_name = defs.model_name.split('.')[0] + '_k' + str(i) + '.hd5'
            # test (k-fold cross validation)
            auc, sensitivity, specificity, ppv, npv, acc, df_pred = model.test(tst_x_list[i], tst_y_list[i], tst_id_list[i], model_name)

            auc_list.append(auc)
            sensi_list.append(sensitivity)
            speci_list.append(specificity)
            ppv_list.append(ppv)
            npv_list.append(npv)
            acc_list.append(acc)

            # append df_pred & test_y for roc_curve
            df_pred_ext = df_pred_ext.append(df_pred, ignore_index=True)
            test_y_ext = np.concatenate((test_y_ext, tst_y_list[i]))
            test_id_ext.extend(tst_id_list[i])

            # for roc_curve average
            fpr, tpr, thresholds = roc_curve(tst_y_list[i], df_pred['recur_prob'], pos_label = 1)
            plt.plot(fpr, tpr, 'b', alpha=0.15)
            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)

            print('fold(i) / auc   / sensitivity / specificity / ppv / npv / accuracy')
            print(i, '     ', auc, sensitivity, specificity, ppv, npv, acc)
            # with open('./result/result.txt','at') as wFile:
            #     wFile.writelines(str(i) + ',     ' + str(auc) + '  ' + str(sensitivity) + '  ' + str(specificity) + '  ' + str(ppv) + '  ' + str(npv) + '  ' + str(acc) + '\n')

            # save pred_result
            df_pred_label = pd.DataFrame(columns=['hospital', 'path_id', 'pred_prob', 'pred_label', 'ground_truth'])
            for idx, row in df_pred.iterrows():
                hospi = row['ids'].split('_')[0]
                path_id = row['ids'].split('_')[1]
                df_pred_label = df_pred_label.append({'hospital' : hospi,
                                                    'path_id' : path_id,
                                                    'pred_prob' : row['recur_prob'],
                                                    'pred_label' : row['pred_label'],
                                                    'ground_truth' : row['label']}, ignore_index=True)
                pred_lable_file = './result/train_pred_label_k' + str(i) + '.csv'
                if os.path.isfile(pred_lable_file):
                    os.remove(pred_lable_file)
                try:
                    df_pred_label.to_csv(pred_lable_file)
                except Exception as e:
                    print(e)

        # auc result
        df_res = pd.DataFrame(auc_list)
        avg_auc = sum(auc_list)/len(auc_list)
        print(df_res)
        print('k-fold avg=', avg_auc)
        print('\n')


        # roc curve (average)
        outfile = './result/roccurve_test_avg.pdf'
        self.draw_roc_curve_avg(tprs, base_fpr, avg_auc, outfile)

        # save roc raw data
        roc_data = (tprs, base_fpr, avg_auc)
        roc_file =  './result/roc_data.cpickle'
        # test
        print(roc_file)
        cpickle.dump(roc_data ,open(roc_file,'wb'))