import os
from keras import backend as K
import argparse
import pandas as pd
import datetime
import json
import itertools
import _pickle as cpickle
from shutil import rmtree
import xgboost as xgb
import subprocess
import glob
import argparse
from scipy import interp
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 100})

from InputDataFactory import InputDataFactory
from ModelFactory import ModelFactory
from Settings import *
import defs
from MakeKMplot import *
from image_augmentation import *




def get_survivals(hospi_patho_list):
    hospitals = [s.split('_')[0] for s in hospi_patho_list]
    path_nos = [s.split('_')[-1] for s in hospi_patho_list]
    anno_df = pd.read_csv(defs.anno_files[0])

    survs = []
    for i, path_no in enumerate(path_nos):
        survs.append(anno_df[anno_df['path_no'] == path_no]['survival_days'].values[0])

    return survs




def train_kfold(input_data, model, args, anno_file):
    # data set
    load_dump_data = defs.do_load_dump_train_kfold
    kfold = 5
    save_crop_path = None
    if defs.do_save_crop or defs.do_save_crop_only:
        save_crop_path = '../cache/crops/train_kfold/'
    trn_x_list, val_x_list, tst_x_list, trn_y_list, val_y_list, tst_y_list, tst_id_list = input_data.get_train_val_tst_list_kfold(anno_file,
                                                                                                                              kfold=kfold,
                                                                                                                              use_x=args['use_x'],
                                                                                                                              patch_sel=defs.do_train_patch_sel,
                                                                                                                              patch_thres_idx=defs.patch_thres_idx,
                                                                                                                              load_dump_data=load_dump_data,
                                                                                                                              save_crop_path=save_crop_path)

    if defs.do_save_crop_only:
        return

    # test 
    print('***** test : kfold len(trn_x_list) = ', len(trn_x_list))

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

    with open('./result/result.txt','at') as wFile:
        wFile.writelines('train k-folds:\n')
        wFile.writelines('fold(i) / auc   / sensitivity / specificity / ppv / npv / accuracy' + '\n')
    for i in range(len(trn_x_list)):
        # train
        model_name = defs.model_name.split('.')[0] + '_k' + str(i) + '.hd5'
        if defs.cascading:
            pt_model_file = defs.model_save_path + model_name   # pre trained model for transfer cascading learning
        else:
            pt_model_file = None
        res = model.train(trn_x_list[i], val_x_list[i], trn_y_list[i], val_y_list[i], model_name, pt_model_file=pt_model_file)
        if res == None:
            continue

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
        with open('./result/result.txt','at') as wFile:
            wFile.writelines(str(i) + ',     ' + str(auc) + '  ' + str(sensitivity) + '  ' + str(specificity) + '  ' + str(ppv) + '  ' + str(npv) + '  ' + str(acc) + '\n')

        # save pred_result
        df_pred_label = pd.DataFrame(columns=['hospital', 'path_id', 'pred_prob', 'pred_label', 'ground_truth'])
        for idx, row in df_pred.iterrows():
            hospi = row['ids'].split('_')[0]
            path_id = row['ids'].split('_')[1]
            df_pred_label = df_pred_label.append({'hospital' : hospi,
                                                  'path_id' : path_id,
                                                  'pred_prob' : row['recur_prob'],
                                                  'pred_label' : row['pred_label'],
                                                  'ground_truth' : row['label'],
                                                  'pred_thres' : row['threshold']}, ignore_index=True)
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
    with open('./result/result.txt','at') as wFile:
        wFile.writelines('k-fold avg auc= ' + str(avg_auc) + '\n')
        wFile.writelines('sensitivity= ' + str(sum(sensi_list)/len(sensi_list)) + '\n')
        wFile.writelines('specificity= ' + str(sum(speci_list)/len(speci_list)) + '\n')
        wFile.writelines('ppv= ' + str(sum(ppv_list)/len(ppv_list)) + '\n')
        wFile.writelines('npv= ' + str(sum(npv_list)/len(npv_list)) + '\n')
        wFile.writelines('acc= ' + str(sum(acc_list)/len(acc_list)) + '\n')
        wFile.writelines('\n')

    # roc curve
    make_kmplot = MakeKMplot()
    print('roc curve ...')
    outfile = './result/roccurve_' + model_name + '_extend.png'
    make_kmplot.draw_roc_curve(test_y_ext, df_pred_ext['recur_prob'], outfile)

    # roc curve (average)
    outfile = './result/roccurve_' + model_name + '_avg.png'
    make_kmplot.draw_roc_curve_avg(tprs, base_fpr, avg_auc, outfile)
    outfile = './result/roccurve_' + model_name + '_avg.pdf'
    make_kmplot.draw_roc_curve_avg(tprs, base_fpr, avg_auc, outfile)
    outfile = './result/roccurve_' + model_name + '_avg.svg'
    make_kmplot.draw_roc_curve_avg(tprs, base_fpr, avg_auc, outfile)
    

    # km plot
    survival_days = get_survivals(test_id_ext)
    outfile = './result/kmplot_' + model_name + '_extend.png'
    make_kmplot.MakeKMplot(test_y_ext, survival_days, df_pred_ext['recur_prob'], outfile)

# test set 없이 train / val set만 만들어서 학습  (최종 모델 만들 때 사용)
def train_full(input_data, model, args, anno_file):
    # Training cohort에서 cross validation 용 test set까지 합쳐서 재 학습
    load_dump_data = defs.do_load_dump_train_full
    save_crop_path = None
    if defs.do_save_crop or defs.do_save_crop_only:
        save_crop_path = '../cache/crops/train_full/'
    train_x, val_x, train_y, val_y = input_data.get_train_val_list(anno_file,
                                                                   use_x=args['use_x'],
                                                                   patch_sel=defs.do_train_patch_sel,
                                                                   patch_thres_idx=defs.patch_thres_idx,
                                                                   load_dump_data=load_dump_data,
                                                                   save_crop_path=save_crop_path,
                                                                   print_data_info=True)

    if defs.do_save_crop_only:
        return None

    if len(train_x) == 0:
        return None

    model_name = defs.model_name
    if defs.cascading:
        pt_model_file = defs.model_save_path + model_name
    else:
        pt_model_file = None
    res = model.train(train_x, val_x, train_y, val_y, defs.model_name, pt_model_file=pt_model_file)
    if res != None:
        print('>>> train full model saved to ', defs.model_name)
    return res


# test set 없이 train / val set만 만들어서 학습  (최종 모델 만들 때 사용)
# 단, val set을 kfold 해서 여러개 실험  => 젤 좋은 것 최종 선택
# train_full() 함수나 이 함수 중 하나만 사용할 것
def train_full_kfold_val(input_data, model, args, anno_file):
    # data set
    load_dump_data = defs.do_load_dump_train_full_kfold
    kfold = 5
    save_crop_path = None
    if defs.do_save_crop or defs.do_save_crop_only:
        save_crop_path = '../cache/crops/train_full_kfold/'

    trn_x_list, val_x_list, trn_y_list, val_y_list = input_data.get_train_val_list_kfold(anno_file,
                                                                                         kfold=kfold,
                                                                                         use_x=args['use_x'],
                                                                                         patch_sel=defs.do_train_patch_sel,
                                                                                         patch_thres_idx=defs.patch_thres_idx,
                                                                                         load_dump_data=load_dump_data,
                                                                                         save_crop_path=save_crop_path)

    if defs.do_save_crop_only:
        return 

    if len(trn_x_list) == 0:
        print('error in train_full_kfold_val() : len(trn_x_list) == 0')
        exit(0)

    for i in range(len(trn_x_list)):
        # train
        model_name = defs.model_name.split('.')[0] + '_full_k' + str(i) + '.hd5'
        if defs.cascading:
            pt_model_file = defs.model_save_path + model_name
        else:
            pt_model_file = None
        res = model.train(trn_x_list[i], val_x_list[i], trn_y_list[i], val_y_list[i], model_name, pt_model_file=pt_model_file)
        if res == None:
            continue
        print('>>> train full k-model saved to ', model_name)



def ext_validation(input_data, model, args, model_name, anno_file, print_data_info):
    make_kmplot = MakeKMplot()

    for model_method in defs.ext_val_models:
        if model_method == 'full':
            model_file = defs.model_save_path + defs.model_name
            if os.path.exists(model_file) or defs.do_save_crop_only:
                pred_save_file = 'ext_val_full_pred_label.csv'

                if defs.do_save_crop or defs.do_save_crop_only:
                    save_crop_path = '../cache/crops/ext_val_full/' + model_name
                else:
                    save_crop_path = None

                ##
                auc, sensi, speci, ppv, npv, acc, _, _, _ = _ext_validation(input_data, model, args, defs.model_name, anno_file, pred_save_file, save_crop_path, print_data_info=True)
                if auc == None:
                    continue

                with open('./result/result.txt','at') as wFile:
                    wFile.writelines('\nExt. Validation (full)...' + '\n')
                    wFile.writelines('model : ' +  model_name + '\n')
                    wFile.writelines('exp(i) / auc     / sensitivity   / specificity   / ppv       / npv      / accuracy' + '\n')
                    wFile.writelines('0      ' + ' ' + str(auc) + ' ' + str(sensi) + ' ' + str(speci) + ' ' + str(ppv) + ' ' + str(npv) + ' ' + str(acc))
            else:
                print('error in ext_validation() : no model file :', model_file)
                with open('./result/result.txt','at') as wFile:
                    wFile.writelines('error in ext_validation() : no model file : ' + str(model_file) + '\n')
            
            

        elif model_method == 'full_kfold':
            with open('./result/result.txt','at') as wFile:
                wFile.writelines('\nExt. Validation (full_kfold)...' + '\n')
                wFile.writelines('exp(i) / auc     / sensitivity   / specificity   / ppv       / npv      / accuracy' + '\n')

            auc_list = []
            sensi_list = []
            speci_list = []
            ppv_list = []
            npv_list = []
            acc_list = []
            test_y = []
            survival_days = []
            for i in range(5):
                model_name = defs.model_name.split('.')[0] + '_full_k' + str(i) + '.hd5'

                # check if model file exist
                model_file = defs.model_save_path + model_name
                if os.path.exists(model_file) or defs.do_save_crop_only:
                    if i == 0:
                        print_data_info = True
                    else:
                        print_data_info = False
                    pred_save_file = 'ext_val_full_kfold_pred_label_K' + str(i) + '.csv'

                    if defs.do_save_crop or defs.do_save_crop_only:
                        save_crop_path = '../cache/crops/ext_val_full_kfold/' + model_name
                    else:
                        save_crop_path = None

                    auc, sensi, speci, ppv, npv, acc, _, _, _ = _ext_validation(input_data, model, args, model_name, anno_file, pred_save_file, save_crop_path, print_data_info)

                    if defs.do_save_crop_only:
                        break

                    if auc == None:
                        continue

                    with open('./result/result.txt','at') as wFile:
                        wFile.writelines(str(i) + '     ' + ' ' + str(auc) + ' ' + str(sensi) + ' ' + str(speci) + ' ' + str(ppv) + ' ' + str(npv) + ' ' + str(acc) + '\n')
                    
                    auc_list.append(auc)
                    sensi_list.append(sensi)
                    speci_list.append(speci)
                    ppv_list.append(ppv)
                    npv_list.append(npv)
                    acc_list.append(acc)
                else:
                    print('error in ext_validation() : no model file :', model_file)
                    with open('./result/result.txt','at') as wFile:
                        wFile.writelines('error in ext_validation() : no model file : ' + str(model_file) + '\n')

            if len(auc_list) > 0:
                avg_auc = sum(auc_list)/len(auc_list)
                avg_sensi = sum(sensi_list)/len(sensi_list)
                avg_speci = sum(speci_list)/len(speci_list)
                avg_ppv = sum(ppv_list)/len(ppv_list)
                avg_npv = sum(npv_list)/len(npv_list)
                avg_acc = sum(acc_list)/len(acc_list)
                with open('./result/result.txt','at') as wFile:
                    wFile.writelines('ext.val average:' + '\n')
                    wFile.writelines('avg auc= ' + str(avg_auc) + '\n')
                    wFile.writelines('avg sensitivity= ' + str(avg_sensi) + '\n')
                    wFile.writelines('avg specificity= ' + str(avg_speci) + '\n')
                    wFile.writelines('avg ppv= ' + str(avg_ppv) + '\n')
                    wFile.writelines('avg npv= ' + str(avg_npv) + '\n')
                    wFile.writelines('avg acc= ' + str(avg_acc) + '\n')
                    wFile.writelines('\n')

        elif model_method == 'kfold':
            with open('./result/result.txt','at') as wFile:
                wFile.writelines('\nExt. Validation (kfold)...' + '\n')
                wFile.writelines('exp(i) / auc     / sensitivity   / specificity   / ppv       / npv      / accuracy' + '\n')

            auc_list = []
            sensi_list = []
            speci_list = []
            ppv_list = []
            npv_list = []
            acc_list = []
            test_y = []
            survival_days = []
            for i in range(5):
                model_name = defs.model_name.split('.')[0] + '_k' + str(i) + '.hd5'

                # check if model file exist
                model_file = defs.model_save_path + model_name
                if os.path.exists(model_file) or defs.do_save_crop_only:
                    if i == 0:
                        print_data_info = True
                    else:
                        print_data_info = False
                    pred_save_file = 'ext_val_kfold_pred_label_K' + str(i) + '.csv'

                    if defs.do_save_crop or defs.do_save_crop_only:
                        save_crop_path = '../cache/crops/ext_val_kfold/' + model_name
                    else:
                        save_crop_path = None

                    # test
                    print('save_crop_path = ', save_crop_path)
                    auc, sensi, speci, ppv, npv, acc, test_y, survival_days, recur_probs = _ext_validation(input_data, model, args, model_name, anno_file, pred_save_file, save_crop_path, print_data_info)
                    if i == 0:
                        sum_recur_probs = recur_probs
                    else:
                        sum_recur_probs = np.add(sum_recur_probs, recur_probs)

                    if defs.do_save_crop_only:
                        break

                    if auc == None:
                        continue

                    with open('./result/result.txt','at') as wFile:
                        wFile.writelines(str(i) + '     ' + ' ' + str(auc) + ' ' + str(sensi) + ' ' + str(speci) + ' ' + str(ppv) + ' ' + str(npv) + ' ' + str(acc) + '\n')

                    auc_list.append(auc)
                    sensi_list.append(sensi)
                    speci_list.append(speci)
                    ppv_list.append(ppv)
                    npv_list.append(npv)
                    acc_list.append(acc)
                else:
                    print('error in ext_validation() : no model file :', model_file)
                    with open('./result/result.txt','at') as wFile:
                        wFile.writelines('error in ext_validation() : no model file : ' + str(model_file) + '\n')

            # # km plot
            # avg_recur_probs = sum_recur_probs / 5
            # outfile = './result/kmplot_' + model_name + '_avg.png'
            # make_kmplot.MakeKMplot(test_y, survival_days, avg_recur_probs, outfile)

            # # roc curve
            # print('roc curve ...')
            # outfile = './result/roccurve_' + model_name + '_avg.png'
            # make_kmplot.draw_roc_curve(test_y, avg_recur_probs, outfile)
            

            if len(auc_list) > 0:
                avg_auc = sum(auc_list)/len(auc_list)
                avg_sensi = sum(sensi_list)/len(sensi_list)
                avg_speci = sum(speci_list)/len(speci_list)
                avg_ppv = sum(ppv_list)/len(ppv_list)
                avg_npv = sum(npv_list)/len(npv_list)
                avg_acc = sum(acc_list)/len(acc_list)
                with open('./result/result.txt','at') as wFile:
                    wFile.writelines('ext.val average:' + '\n')
                    wFile.writelines('avg auc= ' + str(avg_auc) + '\n')
                    wFile.writelines('avg sensitivity= ' + str(avg_sensi) + '\n')
                    wFile.writelines('avg specificity= ' + str(avg_speci) + '\n')
                    wFile.writelines('avg ppv= ' + str(avg_ppv) + '\n')
                    wFile.writelines('avg npv= ' + str(avg_npv) + '\n')
                    wFile.writelines('avg acc= ' + str(avg_acc) + '\n')
                    wFile.writelines('\n')
        else:
            print('error in ext_validation() : model_method = ', model_method)


def _ext_validation(input_data, model, args, model_name, anno_file, pred_save_file, save_crop_path, print_data_info):
    make_kmplot = MakeKMplot()
    anno_df = pd.read_csv(anno_file)

    load_dump_data = defs.do_load_dump_exst_val
    
    test_x, test_y, test_ids = input_data.get_test_list(anno_file,
                                                        use_x=args['use_x'], 
                                                        patch_sel=False, 
                                                        patch_thres_idx=defs.patch_thres_idx, 
                                                        load_dump_data=load_dump_data, 
                                                        save_crop_path=save_crop_path,
                                                        ext_val=True,
                                                        print_data_info=print_data_info)

    if defs.do_save_crop_only:
        return None, None, None, None, None, None

    auc, sensi, speci, ppv, npv, acc, df_pred = model.test(test_x, test_y, test_ids, model_name)

    # km plot
    print('km plot...')
    survival_days = get_survivals(test_ids)
    outfile = './result/kmplot_' + model_name + '.png'
    make_kmplot.MakeKMplot(test_y, survival_days, df_pred['recur_prob'], outfile)

    # roc curve
    print('roc curve ...')
    outfile = './result/roccurve_' + model_name + '.png'
    make_kmplot.draw_roc_curve(test_y, df_pred['recur_prob'], outfile)
    outfile = './result/roccurve_' + model_name + '.pdf'
    make_kmplot.draw_roc_curve(test_y, df_pred['recur_prob'], outfile)
    outfile = './result/roccurve_' + model_name + '.svg'
    make_kmplot.draw_roc_curve(test_y, df_pred['recur_prob'], outfile)

    # save pred_result
    print('pred_label saving...')
    df_pred_label = pd.DataFrame(columns=['hospital', 'path_id', 'pred_prob', 'pred_label', 'ground_truth'])
    for idx, row in df_pred.iterrows():
        hospi = row['ids'].split('_')[0]
        path_id = row['ids'].split('_')[1]
        df_pred_label = df_pred_label.append({'hospital' : hospi,
                                              'path_id' : path_id,
                                              'pred_threshold' : row['threshold'],
                                              'pred_prob' : row['recur_prob'],
                                              'pred_label' : row['pred_label'],
                                              'ground_truth' : row['label'],
                                              'pred_thres' : row['threshold']}, ignore_index=True)
        pred_lable_file = './result/' + pred_save_file
        if os.path.isfile(pred_lable_file):
            os.remove(pred_lable_file)
        try:
            df_pred_label.to_csv(pred_lable_file)
        except Exception as e:
            print(e)

    return auc, sensi, speci, ppv, npv, acc, test_y, survival_days, df_pred['recur_prob']

def make_patch_selector(input_data, model, args, anno_file):
    load_dump_data = defs.do_load_dump_patch_sel

    for patch_sel_iter in range(0, defs.patch_sel_iter):
        print('patch select iter %d:' % patch_sel_iter)

        # 첫번째는 patch_sel = False (patch 선택없이 모든 patch 사용)
        if patch_sel_iter == 0:
            patch_sel = False
        else:
            patch_sel = True

        # 마지막에만 calibration 적용  (아니면 transfer learning 에러나서)
        if patch_sel_iter == defs.patch_sel_iter:
            cal = True
        else:
            cal = False

        __make_patch_selector(input_data, model, args, patch_sel, patch_sel_iter, load_dump_data=load_dump_data, cal=cal, anno_file=anno_file)

def __make_patch_selector(input_data, model, args, patch_sel, patch_sel_iter, load_dump_data, cal, anno_file):
    for mag_idx, magnificient in enumerate(args['use_x']):
        print('  >> ', magnificient)
        if magnificient == 'x400':
            model_name = defs.x400_patch_sel_name
        elif magnificient == 'x200':
            model_name = defs.x200_patch_sel_name
        elif magnificient == 'x100':
            model_name = defs.x100_patch_sel_name
        else:
            print('error in make_patch_selector() : use_x is not correct !!')
            exit(0)

        # model loading for transfer learning
        model_file = None
        if patch_sel:
            model_file = os.path.join(defs.model_save_path, model_name)
            try:
                #patch_sel_model = cpickle.load(open(model_file, 'rb'))
                patch_sel_model = xgb.XGBClassifier(seed=1, objective='binary:logistic', n_estimators=1000)
                patch_sel_model.load_model(model_file)  # load data
            except Exception as e: # work on python 3.x
                print('error in __make_patch_selector() :' + str(e))
                exit(0)

        use_x = []
        use_x.append(magnificient)
        train_x, val_x, train_y, val_y = input_data.get_train_val_list(anno_file, use_x=use_x, patch_sel=patch_sel, patch_thres_idx=patch_sel_iter, load_dump_data=load_dump_data, print_data_info=False)
        # rename the model file name (iteration 에서 덮어쓰여지지 않고 저장하도록)
        if model_file:
            new_name = model_file.split('.hd5')[0] + '_' + str(patch_sel_iter) + '.hd5'
            os.rename(model_file, new_name)
            model_file = new_name

        res = model.train(train_x, val_x, train_y, val_y, model_name_tosave=model_name, pt_model_file=model_file, cal=cal)
        if res != None:
            print('>>> patch selector saved : ', model_name)

# defs.py 변수 setting by config.json
def init_defs(args):
    defs.anno_files = args['anno_files']
    defs.aug_anno_files = args['aug_anno_files']
    defs.pretrain = args['pretrain']
    defs.pretrain_file = args['pretrain_file']
    defs.calibration = args['calibration']
    defs.cascading = args['cascading']
    defs.crop_div_x400 = args['crop_div_x400']
    defs.crop_div_x200 = args['crop_div_x200']
    defs.crop_div_x100 = args['crop_div_x100']

    defs.patch_sel_iter = args['patch_sel_iter']
    defs.patch_thres_idx = args['patch_thres_idx']
    defs.patch_sel_method = args['patch_sel_method']

    defs.thres_h_x400 = args['thres_h_x400']
    defs.thres_l_x400 = args['thres_l_x400']
    defs.thres_h_x200 = args['thres_h_x200']
    defs.thres_l_x200 = args['thres_l_x200']
    defs.thres_h_x100 = args['thres_h_x100']
    defs.thres_l_x100 = args['thres_l_x100']

    if len(defs.anno_files) > 1:
        defs.do_remove_all_model = False
    else:
        defs.do_remove_all_model = args['do_remove_all_model']
    defs.do_save_crop = args['do_save_crop']
    defs.do_train_patch_sel = args['do_train_patch_sel']
    defs.do_train_kfold = args['do_train_kfold']
    defs.do_train_full = args['do_train_full']
    defs.do_train_full_kfold = args['do_train_full_kfold']
    defs.ext_val_models = args['ext_val_models']
    defs.do_train_set_combination = args['do_train_set_combination']
    defs.do_data_augmentation = args['do_data_augmentation']
    defs.do_load_dump_patch_sel = args['do_load_dump_patch_sel']
    defs.do_load_dump_train_kfold = args['do_load_dump_train_kfold']
    defs.do_load_dump_train_full = args['do_load_dump_train_full']
    defs.do_load_dump_train_full_kfold = args['do_load_dump_train_full_kfold']
    defs.do_load_dump_exst_val = args['do_load_dump_exst_val']
    defs.do_save_crop_only = args['do_save_crop_only']

def main(args):
    gpu = args['gpu']
    if gpu != 'all':
        os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']

    # 사용할 input data 클래스 결정
    input_data = InputDataFactory.get_input_data()
    input_data.set_params(args)

    # 사용할 model 클래스 결정
    model = ModelFactory.get_model()

    # result folder
    if not os.path.isdir('result'):
        os.mkdir('result')
    
    git_commit = subprocess.check_output(["git", "describe", "--always"]).strip()
    with open('./result/result.txt','at') as wFile:
        wFile.writelines('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>' + str(datetime.datetime.now()) + '\n')
        wFile.writelines('git : ' + str(git_commit) + '\n')
        json.dump(args, wFile, indent=4)
        wFile.writelines('\n')

    # make patch selector models
    if defs.do_train_patch_sel:
        print('>>>>> make_patch_selecor')

        anno_file = defs.anno_files[0]

        # remove model file if exit
        model_file = os.path.join(defs.model_save_path, defs.x400_patch_sel_name)
        if os.path.isfile(model_file):
            os.remove(model_file)
        model_file = os.path.join(defs.model_save_path, defs.x100_patch_sel_name)
        if os.path.isfile(model_file):
            os.remove(model_file)

        make_patch_selector(input_data, model, args, anno_file)

    K.clear_session()

    # #######################################
    # # test roc curve for k-fold in the training set  (한그래프에 x400, x100, x400_100 다 표현)
    # # 사용법 : 
    # #   1. save_roc_raw_data(..) 호출하여 roc_data.cpickle 생성.  (draw_roc_curve_all(...) 함수는 주석처리)
    # #   2. x400_roc_data.cpickle로 이름 변경
    # #   3. 위의 과정을 x100, x400_x100에대해 반복 (최종결과물 3개 : x400_roc_data.cpickle, x100_roc_data.cpickle, x400_x100_roc_data.cpickle)
    # #   4. save_roc_raw_data()를 주석처리하고 draw_roc_curve_all() 호출
    # make_kmplot = MakeKMplot()
    # for i, anno_file in enumerate(defs.anno_files):
    #     # save roc result raw file  (1. 해당하는 model이 model_save에 있어야 함)
    #     make_kmplot.save_roc_raw_data(input_data, model, args, anno_file)

    #     # draw roc curve (위의 save roc raw data가 있어야 함)
    #     #make_kmplot.draw_roc_curve_all(input_data, model, args, anno_file)


    #######################################################
    # training (k-fold)
    if defs.do_train_kfold:
        print('\n>>>>> train kfold')        
        for i, anno_file in enumerate(defs.anno_files):
            print('>>> anno_file : ', anno_file)
            with open('./result/result.txt','at') as wFile:
                wFile.writelines('>>> anno_file : ' + anno_file + '\n')
            train_kfold(input_data, model, args, anno_file)
    
    K.clear_session()


    #######################################################
    # training (full) : 최종 모델
    if defs.do_train_full:
        print('\n>>>>> train full once')
        for i, anno_file in enumerate(defs.anno_files):
            print('>>> anno_file : ', anno_file)
            with open('./result/result.txt','at') as wFile:
                wFile.writelines('>>> anno_file : ' + anno_file + '\n')
            res = train_full(input_data, model, args, anno_file)
        
    elif defs.do_train_full_kfold:
        print('\n>>>>> train full kfold')
        for i, anno_file in enumerate(defs.anno_files):
            print('>>> anno_file : ', anno_file)
            with open('./result/result.txt','at') as wFile:
                wFile.writelines('>>> anno_file : ' + anno_file + '\n')
            train_full_kfold_val(input_data, model, args, anno_file)
    
    K.clear_session()
    
    #######################################################
    # External Validationmodel_file = defs.model_save_path + defs.model_name
    print('\n>>>>> ext. validation')
    anno_file = defs.anno_files[0]
    ext_validation(input_data, model, args, defs.model_name, anno_file, print_data_info=True)


    with open('./result/result.txt','at') as wFile:
        wFile.writelines('\n======================================== ' + str(datetime.datetime.now()) + ' ===\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.json', type=str)
    args = parser.parse_args()

    config_file = args.config
    # config.json file 확인 & 없으면 샘플 config.json파일 출력 후 종료
    if os.path.exists(config_file) == False:
        print('Error : no config file!')
        exit(0)

    # load input json
    with open(config_file) as f:
        args = json.load(f)
        print(json.dumps(args, indent = 4))

    # init defs.py
    init_defs(args)

    # init (model_save folder 삭제)
    if defs.do_remove_all_model:
        ans = input("Removing all models (y/n):")
        if ans.lower() == 'y': 
            model_file = os.path.join(defs.model_save_path, 'xgb_*')
            fileList = glob.glob(model_file, recursive=True)
            for filePath in fileList:
                try:
                    os.remove(filePath)
                except OSError:
                    print("Error while deleting file")

    # init(crop save folder 삭제&생성)
    if os.path.exists(defs.crop_path):
        rmtree(defs.crop_path)
        os.makedirs(defs.crop_path)


    # imbalanced image augmentation
    if defs.do_data_augmentation == True:
        remove_aug_images(args['img_root_path'])
        image_augmentation(args)

    # run
    main(args)
    



    # # test
    # for i in range(5):
    #     for j in range(5):            
    #         model_name = 'resnet50_' + str(i) + '_' + str(j) + '.h5'
    #         args['pretrain_file'] = model_name

    #         if not(i == 0 and j == 0):
    #             args['do_load_dump_train_kfold'] = True
    #             args['do_load_dump_train_full'] = True
    #             args['do_load_dump_train_full_kfold'] = True
    #             args['do_load_dump_exst_val'] = True

    #         main(args)

    
    # if defs.do_train_set_combination:
    #     # 여러 train_hospi에 대해 자동 테스트
    #     lst = args['train_hospi_name']
    #     combs = []
    #     for i in range(1, len(lst)+1):
    #         els = [list(x) for x in itertools.combinations(lst, i)]
    #         combs.extend(els)

    #     for hospi_name in combs:
    #         K.clear_session()
    #         args['train_hospi_name'] = hospi_name
    #         main(args)
    # else:
    #     main(args)
