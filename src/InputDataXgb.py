import os
import random
import numpy as np
import pandas as pd
import _pickle as cpickle
import warnings
import sys
from tqdm import tqdm
import xgboost as xgb
from skimage.io import imread, imsave
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras import backend as K
from keras.models import Model, load_model
#from keras.applications.vgg16 import VGG16
#from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50, preprocess_input

from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

from skimage.transform import resize
from shutil import rmtree
import xgboost as xgb
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


from ImageFactory import ImageFactory
import defs

class InputDataXgb():
    def __init__(self):
        self.img_factory = ImageFactory()

    def set_params(self, args):
        self.img_root_path = args['img_root_path']
        self.train_hospi_name = args['train_hospi_name']
        self.test_hospi_name = args['test_hospi_name']
        self.color_norm = args['color_norm']


    # kfold validation을 위한 set list
    def get_train_val_tst_list_kfold(self, anno_file, kfold, use_x, patch_sel, patch_thres_idx, load_dump_data=False, save_crop_path=None):
        print('InputDataXgb.get_train_val_list_kold()...')
        anno_df = pd.read_csv(anno_file)

        print_data_info = False

        if load_dump_data:
            trn_x_list, val_x_list, tst_x_list, trn_y_list, val_y_list, tst_y_list, tst_id_list = cpickle.load(open('train_val_tst_list_data_dump.cpickle','rb'))
            return trn_x_list, val_x_list, tst_x_list, trn_y_list, val_y_list, tst_y_list, tst_id_list

        anno_df = anno_df[anno_df['hospital'].isin(self.train_hospi_name)]
        hospi_name = anno_df.hospital.values
        path_ids = anno_df.path_no.values
        labels = anno_df.label.values

        if not (len(hospi_name) == len(path_ids) and len(hospi_name) == len(labels)):
            print('error in get_train_val_tst_list_kfold() : incorrect data length')
            exit(0)

        # train을 위한 feature를 뽑기전에 "hospi_name+path_id" 과 label 로 x, y data set을 만들고 split 함 
        # 이렇게 하는 이유는 train_val set 과 test set의 전처리 과정이 다르기 떄문  (train set만 good patch selection 과정 추가)
        data_x = []
        for i, hospi in enumerate(hospi_name):
            data_x.append(hospi + ',' + path_ids[i])
        data_x = np.asarray(data_x)
        labels = np.asarray(labels)
        path_ids = np.asarray(path_ids)

        data_x, labels, path_ids = shuffle(data_x, labels, path_ids, random_state = 1024)

        # 1. 전체 데이터를 train set과 test set으로 분리(5-fold)
        # 2. 5개(5-fold) train set에 대해서 다시 trn set과 val set으로 분리
        trn_x_list = []
        val_x_list = []
        tst_x_list = []
        trn_y_list = []
        val_y_list = []
        tst_y_list = []
        tst_id_list = []
        skfold = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=1024)
        i = 0
        for trn_val_idx, tst_idx in skfold.split(data_x, labels):
            trn_val_x = data_x[trn_val_idx]
            trn_val_y = labels[trn_val_idx]
            tst_x = data_x[tst_idx]
            tst_y = labels[tst_idx]

            try:
                trn_x, val_x, trn_y, val_y = train_test_split(trn_val_x, trn_val_y, test_size=0.1, random_state=1024, stratify=trn_val_y)
            except:
                print('warning : get_train_val_tst_list_kfold() : train_test_split exception!')
                return [], [], [], [], [], [], []

            # get trn_x,y (change to features)
            print('\n<get trn set>')
            if i == 0:
                print_data_info = True
            else:
                print_data_info = False

            if print_data_info:
                with open('./result/result.txt','at') as wFile:
                    wFile.writelines('<get K#' + str(i) + ' trn set>' + '\n')
            
            crop_path = None
            if save_crop_path != None:
                crop_path = save_crop_path + defs.model_name.split('.')[0] + '_k' + str(i) + '.hd5' # save_crop_path에 model name 첨부 이유는 patch pred 를 구하기 위함 (visualization 용), 단 ext.val과는 달리 cheating 결과임
            trn_x, trn_y, trn_id = self.make_feature_label_list(self.img_root_path,
                                                                trn_x,
                                                                trn_y,
                                                                self.train_hospi_name,
                                                                use_x,
                                                                self.color_norm,
                                                                patch_sel=patch_sel,
                                                                patch_thres_idx=patch_thres_idx,
                                                                save_crop_path=crop_path,
                                                                print_data_info=print_data_info,
                                                                aug=defs.do_data_augmentation)
            if len(trn_x) == 0:
                print('error in get_train_val_tst_list_kfold() : len(trn_x) = 0 ')
                return [], [], [], [], [], [], []

            # get val_x,y (features)
            print('\n<get val set>')
            if print_data_info:
                with open('./result/result.txt','at') as wFile:
                    wFile.writelines('<get K#' + str(i) + ' val set>' + '\n')
            val_x, val_y, val_id = self.make_feature_label_list(self.img_root_path,
                                                                val_x,
                                                                val_y,
                                                                self.train_hospi_name,
                                                                use_x,
                                                                self.color_norm,
                                                                patch_sel=False,
                                                                patch_thres_idx=patch_thres_idx,
                                                                save_crop_path=crop_path,
                                                                print_data_info=print_data_info,
                                                                aug=defs.do_data_augmentation)
            if len(val_x) == 0:
                print('error in get_train_val_tst_list_kfold() : len(val_x) = 0 ')
                return [], [], [], [], [], [], []

            # get tst_x,y (features)
            print('\n<get tst set>')
            if print_data_info:
                with open('./result/result.txt','at') as wFile:
                    wFile.writelines('<get K#' + str(i) + ' tst set>' + '\n')
            tst_x, tst_y, tst_id = self.make_feature_label_list(self.img_root_path,
                                                                tst_x,
                                                                tst_y,
                                                                self.train_hospi_name,
                                                                use_x,
                                                                self.color_norm,
                                                                patch_sel=False,
                                                                patch_thres_idx=patch_thres_idx,
                                                                save_crop_path=crop_path,
                                                                print_data_info=print_data_info,
                                                                aug=False)

            if defs.do_save_crop_only:
                return None, None, None, None, None, None, None

            trn_x_list.append(trn_x)
            trn_y_list.append(trn_y)
            val_x_list.append(val_x)
            val_y_list.append(val_y)
            tst_x_list.append(tst_x)
            tst_y_list.append(tst_y)
            tst_id_list.append(tst_id)

            print('>>> get_train_val_tst_list_kfold(#%d) finished : trn=%d, val=%d, tst=%d' % (i, len(trn_x), len(val_x), len(tst_x)))
            i += 1

        train_data = (trn_x_list, val_x_list, tst_x_list, trn_y_list, val_y_list, tst_y_list, tst_id_list)
        cpickle.dump(train_data,open('train_val_tst_list_data_dump.cpickle','wb'))

        return trn_x_list, val_x_list, tst_x_list, trn_y_list, val_y_list, tst_y_list, tst_id_list


    # 최종 모델 train용  (test set 없음)
    def get_train_val_list(self, anno_file, use_x, patch_sel, patch_thres_idx, load_dump_data=False, save_crop_path=None, print_data_info=False):
        print('InputDataXgb.get_train_val_list()...')
        anno_df = pd.read_csv(anno_file)

        if load_dump_data:
            train_x, val_x, train_y, val_y = cpickle.load(open('train_val_data_dump.cpickle','rb'))
            return train_x, val_x, train_y, val_y

        anno_df = anno_df[anno_df['hospital'].isin(self.train_hospi_name)]
        hospi_name = anno_df.hospital.values
        path_ids = anno_df.path_no.values
        labels = anno_df.label.values

        if not (len(hospi_name) == len(path_ids) and len(hospi_name) == len(labels)):
            print('error in get_train_val_list() : incorrect data length')
            exit(0)

        # train을 위한 feature를 뽑기전에 "hospi_name+path_id" 과 label 로 x, y data set을 만들고 split 함 
        # 이렇게 하는 이유는 train set과 val & test set의 전처리 과정이 다르기 떄문  (train set만 good patch selection 과정 추가)
        data_x = []
        for i, hospi in enumerate(hospi_name):
            data_x.append(hospi + ',' + path_ids[i])
        data_x, labels, path_ids = shuffle(data_x, labels, path_ids, random_state = 1024)

        try:
            trn_x, val_x, trn_y, val_y = train_test_split(data_x, labels, test_size=0.1, random_state=1024, stratify=labels)
        except:
            print('warning : get_train_val_list() : train_test_split exception!')
            return [], [], [], []

        # get trn_x,y list (change to features)
        print('\n<get trn set>')
        if print_data_info:
            with open('./result/result.txt','at') as wFile:
                wFile.writelines('<get full trn set>' + '\n')

        crop_path = None
        if save_crop_path != None:
            crop_path = save_crop_path + defs.model_name # save_crop_path에 model name 첨부 이유는 patch pred 를 구하기 위함 (visualization 용), 단 ext.val과는 달리 cheating 결과임
        trn_x, trn_y, trn_id = self.make_feature_label_list(self.img_root_path,
                                                            trn_x,
                                                            trn_y,
                                                            self.train_hospi_name,
                                                            use_x,
                                                            self.color_norm,
                                                            patch_sel=patch_sel,
                                                            patch_thres_idx=patch_thres_idx,
                                                            save_crop_path=save_crop_path,
                                                            print_data_info=print_data_info,
                                                            aug=defs.do_data_augmentation)
        if len(trn_x) == 0:
                return [], [], [], []

        # get val_x,y (features)
        print('\n<get val set>')
        if print_data_info:
            with open('./result/result.txt','at') as wFile:
                wFile.writelines('<get full val set>' + '\n')
        val_x, val_y, val_id = self.make_feature_label_list(self.img_root_path,
                                                            val_x,
                                                            val_y,
                                                            self.train_hospi_name,
                                                            use_x,
                                                            self.color_norm,
                                                            patch_sel=False,
                                                            patch_thres_idx=patch_thres_idx,
                                                            save_crop_path=save_crop_path,
                                                            print_data_info=print_data_info,
                                                            aug=defs.do_data_augmentation)

        train_data = (trn_x, val_x, trn_y, val_y)
        cpickle.dump(train_data,open('train_val_data_dump.cpickle','wb'))

        print('>>> get_train_val_list() finished : trn=%d, val=%d' % (len(trn_x), len(val_x)))
        return trn_x, val_x, trn_y, val_y


    # 최종 full train set 구하는거지만 val set을 5가지 fold로 해서 젤 좋은 걸 최종적으로 선택하기 위한 함수
    # get_train_val_list() 함수나 이함수 둘중 하나만 사용 할 것
    # test set 없음
    def get_train_val_list_kfold(self, anno_file, kfold, use_x, patch_sel, patch_thres_idx, load_dump_data=False, save_crop_path=None):
        print('InputDataXgb.get_train_val_list_kold()...')
        anno_df = pd.read_csv(anno_file)

        print_data_info = False

        if load_dump_data:
            trn_x_list, val_x_list, trn_y_list, val_y_list = cpickle.load(open('train_val_list_data_dump.cpickle','rb'))
            return trn_x_list, val_x_list, trn_y_list, val_y_list

        anno_df = anno_df[anno_df['hospital'].isin(self.train_hospi_name)]
        hospi_name = anno_df.hospital.values
        path_ids = anno_df.path_no.values
        labels = anno_df.label.values

        if not (len(hospi_name) == len(path_ids) and len(hospi_name) == len(labels)):
            print('error in get_train_val_list_kfold() : incorrect data length')
            exit(0)

        # train을 위한 feature를 뽑기전에 "hospi_name+path_id" 과 label 로 x, y data set을 만들고 split 함 
        # 이렇게 하는 이유는 train_val set 과 test set의 전처리 과정이 다르기 떄문  (train set만 good patch selection 과정 추가)
        data_x = []
        for i, hospi in enumerate(hospi_name):
            data_x.append(hospi + ',' + path_ids[i])
        data_x = np.asarray(data_x)
        labels = np.asarray(labels)
        path_ids = np.asarray(path_ids)

        data_x, labels, path_ids = shuffle(data_x, labels, path_ids, random_state = 1024)

        # 1. 전체 데이터를 train set과 test set으로 분리(5-fold)
        # 2. 5개(5-fold) train set에 대해서 다시 trn set과 val set으로 분리
        trn_x_list = []
        val_x_list = []
        trn_y_list = []
        val_y_list = []
        skfold = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=1024)
        i = 0
        for trn_idx, val_idx in skfold.split(data_x, labels):
            trn_x = data_x[trn_idx]
            trn_y = labels[trn_idx]
            val_x = data_x[val_idx]
            val_y = labels[val_idx]

            # get trn_x,y (change to features)
            print('\n<get trn set>')
            if i == 0:
                print_data_info = True
            else:
                print_data_info = False

            if print_data_info:
                with open('./result/result.txt','at') as wFile:
                    wFile.writelines('<get full trn set> : K#' + str(i) + '\n')

            crop_path = None
            if save_crop_path != None:
                crop_path = save_crop_path + defs.model_name.split('.')[0] + '_full_k' + str(i) + '.hd5' # save_crop_path에 model name 첨부 이유는 patch pred 를 구하기 위함 (visualization 용), 단 ext.val과는 달리 cheating 결과임
            trn_x, trn_y, trn_id = self.make_feature_label_list(self.img_root_path,
                                                                trn_x,
                                                                trn_y,
                                                                self.train_hospi_name,
                                                                use_x,
                                                                self.color_norm,
                                                                patch_sel=patch_sel,
                                                                patch_thres_idx=patch_thres_idx,
                                                                save_crop_path=crop_path,
                                                                print_data_info=print_data_info,
                                                                aug=defs.do_data_augmentation)
            if len(trn_x) == 0:
                print('error in get_train_val_list_kfold() : len(trn_x) = 0 ')
                return [], [], [], [], [], [], []

            # get val_x,y (features)
            print('\n<get val set>')
            if print_data_info:
                with open('./result/result.txt','at') as wFile:
                    wFile.writelines('<get val set>' + '\n')
            val_x, val_y, val_id = self.make_feature_label_list(self.img_root_path,
                                                                val_x,
                                                                val_y,
                                                                self.train_hospi_name,
                                                                use_x,
                                                                self.color_norm,
                                                                patch_sel=False,
                                                                patch_thres_idx=patch_thres_idx,
                                                                save_crop_path=crop_path,
                                                                print_data_info=print_data_info,
                                                                aug=defs.do_data_augmentation)

            if defs.do_save_crop_only:
                return None, None, None, None

            if len(val_x) == 0:
                print('error in get_train_val_list_kfold() : len(val_x) = 0 ')
                return [], [], [], [], [], [], []

            trn_x_list.append(trn_x)
            trn_y_list.append(trn_y)
            val_x_list.append(val_x)
            val_y_list.append(val_y)

            print('>>> get_train_val_list_kfold(#%d) finished : trn=%d, val=%d' % (i, len(trn_x), len(val_x)))
            i += 1

        train_data = (trn_x_list, val_x_list, trn_y_list, val_y_list)
        cpickle.dump(train_data,open('train_val_list_data_dump.cpickle','wb'))

        return trn_x_list, val_x_list, trn_y_list, val_y_list


    def get_test_list(self, anno_file, use_x, patch_sel, patch_thres_idx, load_dump_data=False, save_crop_path=None, ext_val=False, print_data_info=False):
        print('InputDataXgb.get_test_list()...')
        anno_df = pd.read_csv(anno_file)
        anno_df = anno_df[anno_df['hospital'].isin(self.test_hospi_name)]
        hospi_name = anno_df.hospital.values
        path_ids = anno_df.path_no.values
        labels = anno_df.label.values

        if not (len(hospi_name) == len(path_ids) and len(hospi_name) == len(labels)):
            print('error in get_test_list() : incorrect data length')
            exit(0)

        data_x = []
        for i, hospi in enumerate(hospi_name):
            data_x.append(hospi + ',' + path_ids[i])
        data_x, labels, path_ids = shuffle(data_x, labels, path_ids, random_state = 1024)

        if load_dump_data:
            test_x, test_y, test_ids = cpickle.load(open('test_data_dump.cpickle','rb'))
        else:
            test_x, test_y, test_ids = self.make_feature_label_list(self.img_root_path,
                                                                    data_x,
                                                                    labels,
                                                                    self.test_hospi_name,
                                                                    use_x,
                                                                    self.color_norm,
                                                                    patch_sel,
                                                                    patch_thres_idx,
                                                                    save_crop_path=save_crop_path,
                                                                    ext_val=ext_val,
                                                                    print_data_info=print_data_info,
                                                                    aug=False)
            test_data = (test_x, test_y, test_ids)
            outfile = 'test_data_dump.cpickle'
            if os.path.exists(outfile):
                outfile = outfile.split('.cpickle')[0] + '_1.cpickle'
            cpickle.dump(test_data,open(outfile,'wb'))

        return test_x, test_y, test_ids

    
    #
    # make_feature_label_list()
    #   @data_x : list -> [ "hospi + ',' + path_id", ... ]
    def make_feature_label_list(self, root, data_x, labels, hospital_name, use_x, color_norm, patch_sel, patch_thres_idx, save_crop_path=None, ext_val=False, print_data_info=False, aug=False):
        warnings.filterwarnings("ignore", "Possibly corrupt EXIF data", UserWarning)

        # data augmentation
        if aug:
            data_x, labels = self.add_augment_list(data_x, labels)

        # crop image save folder
        if save_crop_path != None:
            if not os.path.exists(save_crop_path):
                os.makedirs(save_crop_path)

            non_resize_crop_path = os.path.join(save_crop_path, '_non_resize_crop')
            if not os.path.exists(non_resize_crop_path):
                os.makedirs(non_resize_crop_path)

        x400_patch_clf = None
        x200_patch_clf = None
        x100_patch_clf = None
        patch_clf = None
        if patch_sel:
            for mag_idx, magnificient in enumerate(use_x):
                if magnificient == 'x400':
                    fname = os.path.join(defs.model_save_path, defs.x400_patch_sel_name)
                    try:
                        if defs.calibration:
                            x400_patch_clf = xgb.XGBClassifier(seed=1, objective='binary:logistic', n_estimators=1000)
                            x400_patch_clf.load_model(fname)  # load data
                        else:
                            x400_patch_clf = cpickle.load(open(fname, 'rb'))
                    except:
                        print('error : no patch selector : ', fname)
                        exit(0)

                elif magnificient == 'x200':
                    fname = os.path.join(defs.model_save_path, defs.x200_patch_sel_name)
                    try:
                        if defs.calibration:
                            x200_patch_clf = xgb.XGBClassifier(seed=1, objective='binary:logistic', n_estimators=1000)
                            x200_patch_clf.load_model(fname)  # load data
                        else:
                            x200_patch_clf = cpickle.load(open(fname, 'rb'))
                    except:
                        print('error : no patch selector : ', fname)
                        exit(0)

                elif magnificient == 'x100':
                    fname = os.path.join(defs.model_save_path, defs.x100_patch_sel_name)
                    try:
                        if defs.calibration:
                            x100_patch_clf = xgb.XGBClassifier(seed=1, objective='binary:logistic', n_estimators=1000)
                            x100_patch_clf.load_model(fname)  # load data
                        else:
                            x100_patch_clf = cpickle.load(open(fname, 'rb'))
                        
                    except:
                        print('error : no patch selector : ', fname)
                        exit(0)

        # # test
        # try:
        #     ext_patch_clf = cpickle.load(open('./model_save/xgb_recur_clf.hd5', 'rb'))
        # except:
        #     print('warning : xgb_recur_clf.hd5 !!')
        #     ext_patch_clf = None

        hospi_name = []
        path_ids = []
        for item in data_x:
            hospi_name.append(item.split(',')[0])
            path_ids.append(item.split(',')[1])

        base_model = ResNet50(include_top=True)
        weight_file = os.path.join(defs.model_save_path, 'tcga_MODEL_resnet50.h5')
        base_model.load_weights(weight_file,by_name=True)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

        channel = 3
        feature_list = []
        id_list = []
        y_list = []
        crop_times = 50
        ##crop_top_k = 20        
        crop_thres_high = 0
        crop_thres_low = 0
        do_pca = False
        pca = PCA(n_components=16)
        patch_size = 224

        # input data 통계 저장용 df (slide matched)
        df_input_info = pd.DataFrame(columns=['hospital','recur', 'non_recur'])
        input_infos = []
        hospi_list = list(set(hospi_name))
        for hospi in hospi_list:
            df_input_info = df_input_info.append({'hospital':hospi, 'recur':0, 'non_recur':0}, ignore_index=True)

        print('\n')
        for i in tqdm(range(0,len(labels)), file=sys.stdout): 
            # features[] : crop 한개의 feature 4096개
            # crop_features[] : 한 이미지당 존재.  crop들의 features[] 합침  => crop_features[ features[], features[], ... ]
            # mag_crop_features[] : crop_features[]를 x100, x200, ... 별로 합침  => [ crop_features[features[], features[], ...], crop_features[], ...]
            mag_crop_features = []
            for mag_idx, magnificient in enumerate(use_x):
                if magnificient == 'x400':
                    # 임시로 use_x에 따라 threshold 다르게 ???
                    crop_thres_high = defs.thres_h_x400[patch_thres_idx]
                    crop_thres_low = defs.thres_l_x400[patch_thres_idx]

                    x_file_tail = '_5.jpg'
                    patch_clf = x400_patch_clf
                    patch_size = 224
                    crop_divider = defs.crop_div_x400
                if magnificient == 'x200':
                    crop_thres_high = defs.thres_h_x200[patch_thres_idx]
                    crop_thres_low = defs.thres_l_x200[patch_thres_idx]

                    x_file_tail = '_4.jpg'
                    patch_clf = x200_patch_clf
                    patch_size = 224
                    crop_divider = defs.crop_div_x200
                if magnificient == 'x100':
                    # 임시로 use_x에 따라 threshold 다르게 ???
                    crop_thres_high = defs.thres_h_x100[patch_thres_idx]
                    crop_thres_low = defs.thres_l_x100[patch_thres_idx]

                    x_file_tail = '_3.jpg'
                    patch_clf = x100_patch_clf
                    patch_size = 500
                    crop_divider = defs.crop_div_x100
                if magnificient == 'x40':
                    x_file_tail = '_2.jpg'
                    crop_divider = 3

                if hospi_name[i]=='Seo' or hospi_name[i]=='Seo2':
                    file = root+hospi_name[i]+'/'+path_ids[i]+x_file_tail
                else:
                    if hospi_name[i]=='Inc':
                        items = path_ids[i].split('-')
                        file = root+hospi_name[i]+'/'+items[0]+'-'+items[1]+'-'+str(int(items[2]))+x_file_tail
                    else:
                        items = path_ids[i].split('-')
                        file = root+hospi_name[i]+'/'+items[0]+'-'+str(int(items[1]))+x_file_tail            

                try:
                    org_img = imread(file)[:,:,:channel]
                except:
                    # magnificient 중 하나라도 이미지가 없으면 둘다 무시
                    #print(i,magnificient,file,'read img error')
                    mag_crop_features = []
                    break

                img_height = int(org_img.shape[0])
                img_width = int(org_img.shape[1])

                crop_features = []
                cell_prob = []

                # test
                print(file, "  ", img_width, ' ', img_height)

                # resize image to the common size (가로세로 비율 유지, height기준으로 맞춤)
                if (img_width != defs.max_img_size[0]) or (img_height != defs.max_img_size[1]):
                    new_hgt = defs.max_img_size[1]
                    ratio = img_width / img_height
                    new_wdt = int(new_hgt * ratio)
                    img = resize(org_img, (new_hgt, new_wdt), anti_aliasing=True)
                    rescaled_image = 255 * img
                    # Convert to integer data type pixels.
                    img = rescaled_image.astype(np.uint8)
                    img_height = new_hgt
                    img_width = new_wdt
                else:
                    img = org_img

                # test
                print("after resize: ", img_width, ' ', img_height)
                    
                crop_positions = self.img_factory.get_crop_position(img_width, img_height, 
                                                                    crop_size=patch_size, 
                                                                    bottom_cut=100, 
                                                                    crop_divider=crop_divider, 
                                                                    random_crop_n=0)
                # test (crop image save)
                crops = []

                for crop_pos in crop_positions:
                    start_w = crop_pos[0]
                    start_h = crop_pos[1]
                    end_w = crop_pos[2]
                    end_h = crop_pos[3]

                    img_crop = img[start_h:end_h,start_w:end_w,:]

                    if(color_norm):
                        img_crop = self.img_factory.stain_normalize(img_crop)

                    # resize to 224x224
                    if patch_size != 224:
                        img_crop = resize(img_crop, (224, 224), anti_aliasing=True)
                        rescaled_image = 255 * img_crop
                        # Convert to integer data type pixels.
                        img_crop = rescaled_image.astype(np.uint8)

                    x = np.expand_dims(img_crop, axis=0)
                    x = preprocess_input(x)
                    features = model.predict(x)

                    if patch_clf != None:
                        pred = patch_clf.predict_proba(features)[:,1]
                    else:
                        pred = [0]*len(features)

                    # test (ext. val crops 들의 확률 저장을 위함. (patch_clf가 아닌 recur_clf사용)
                    # x400또는 x100 단독 수행시만 가능.)
                    if save_crop_path != None and len(use_x) == 1:
                        model_name = save_crop_path.split('/')[-1]
                        try:
                            recur_clf = cpickle.load(open(os.path.join(defs.model_save_path, model_name), 'rb'))
                            # recur_clf = xgb.XGBClassifier(seed=1, objective='binary:logistic', n_estimators=1000)
                            # recur_clf.load_model(os.path.join(defs.model_save_path, model_name))  # load data
                        except:
                            print('warning : crop pred model (for test) does not exist')

                        if recur_clf != None:
                            pred = recur_clf.predict_proba(features)[:,1]

                    features = features.flatten()
                    crop_features.append(features)
                    cell_prob.append(pred)

                    # test (crop image save)
                    crops.append(img_crop)

                crop_df = pd.DataFrame(columns=['features','cell_prob'])
                crop_df['features'] = crop_features
                crop_df['cell_prob'] = cell_prob
                crop_df = crop_df.sort_values(by=['cell_prob'], ascending=False)

                # crop image save
                print('save_crop_path=', save_crop_path)
                if save_crop_path != None:
                    img_name = file.split('/')[-1].split('.')[0] + '_'
                    target_path = os.path.join(save_crop_path, hospi_name[i] + '_' + img_name)
                    self.img_factory.save_images(target_path=target_path, imgs=crops, suffix=str(labels[i]))
                    crop_df.to_csv(target_path + str(labels[i]) + '.csv')

                    # test save non-resized crop image for visualization 임광일선생님
                    target_path = os.path.join(non_resize_crop_path, hospi_name[i] + '_' + img_name)
                    self.save_non_resized_crops(org_img, patch_size, target_path, crop_divider, labels[i])

                total_crop_cnt = len(crop_df.index)
                
                if patch_sel:
                    if patch_clf != None:
                        crop_df = self.patch_select(crop_df, labels[i], crop_thres_high, crop_thres_low, method=defs.patch_sel_method)
                    else:
                        print('error in make_feature_label_list() : patch_clf is None!')
                        exit(0)

                # # test  (ext.val)
                # if ext_val:
                #     # prediction이 아주 크거나 아주 작은 것만 택함
                #     crop_n = len(crop_df.index)
                #     low_end = int(1/4 * crop_n)
                #     high_start = int(3/4 * crop_n)
                #     crop_df_l = crop_df[:low_end]
                #     crop_df_h = crop_df[high_start:]
                #     # crop_df_h = crop_df[crop_df.cell_prob > crop_thres_high]
                #     # crop_df_l = crop_df[crop_df.cell_prob < crop_thres_low]
                    
                #     crop_df = crop_df_h.append(crop_df_l, ignore_index=True)

                # test
                if i == 0 and mag_idx == 0:
                    erase = ' '
                else:
                    erase = '\x1b[1A\x1b[2K'
                print(erase + '\r' + magnificient + ' imgsize(%d, %d),  tot_crops = %d,  sel_crops = %d' % (img_width, img_height, total_crop_cnt, len(crop_df.index)))

                # 조건에 맞는 crop이 하나도 없는 경우 이 이미지 건너 뜀
                if len(crop_df.index) == 0:
                    #print('>> crop num is 0')
                    mag_crop_features = []
                    break

                mag_crop_features.append(crop_df['features'].tolist())

            # if some magnificant image's appropriate crop doesn't exit, skip this image
            if len(mag_crop_features) == 0:
                continue

            # input date 통계 업데이트
            if labels[i] == 1:
                cur_cnt = df_input_info[df_input_info.hospital == hospi_name[i]].iloc[0]['recur']
                df_input_info.loc[df_input_info.hospital == hospi_name[i], 'recur'] = cur_cnt + 1
            else:
                cur_cnt = df_input_info[df_input_info.hospital == hospi_name[i]].iloc[0]['non_recur']
                df_input_info.loc[df_input_info.hospital == hospi_name[i], 'non_recur'] = cur_cnt + 1
                
            mag_avg_features = []
            for crop_features in mag_crop_features:
                avg_feature = np.mean(crop_features,axis=0)
                mag_avg_features = np.append(mag_avg_features, avg_feature)

            feature_list.append(mag_avg_features)
            y_list.append(labels[i])
            id_list.append(hospi_name[i] + '_' + path_ids[i])

        print('final_len : ', len(y_list))

        tot_recur = df_input_info['recur'].sum()
        tot_non_recur = df_input_info['non_recur'].sum()
        df_input_info = df_input_info.append({'hospital':'sum', 'recur':tot_recur, 'non_recur':tot_non_recur}, ignore_index=True)
        print(df_input_info)
        if print_data_info:
            with open('./result/result.txt','at') as wFile:
                wFile.writelines('-------------------------------' + '\n')
                df_input_info.to_string(wFile)
                wFile.writelines('\n-------------------------------' + '\n')

        return np.asarray(feature_list), np.asarray(y_list), np.asarray(id_list)

    # visualization (gradCAM) 할 때 resized crop 이미지는 해상도가 떨어져서 분석하기 힘들기 때문에 사용
    def save_non_resized_crops(self, img, patch_size, target_path, crop_divider, label):
        img_height = int(img.shape[0])
        img_width = int(img.shape[1])

        resize_hgt = defs.max_img_size[1]
        resize_ratio = img_height / resize_hgt
        org_crop_size = int(resize_ratio * patch_size + 0.5)

        crop_positions = self.img_factory.get_crop_position(img_width, img_height, 
                                                            crop_size=org_crop_size, 
                                                            bottom_cut=100, 
                                                            crop_divider=crop_divider, 
                                                            random_crop_n=0)
        # test (crop image save)
        crops = []

        for crop_pos in crop_positions:
            start_w = crop_pos[0]
            start_h = crop_pos[1]
            end_w = crop_pos[2]
            end_h = crop_pos[3]

            img_crop = img[start_h:end_h,start_w:end_w,:]
            crops.append(img_crop)

        self.img_factory.save_images(target_path=target_path, imgs=crops, rewrite=True, suffix=str(label))

    #
    # add_augment_list()
    #   clinical_2_aug.csv 파일로부터 augmented id list를 읽어와서 data_x 에 추가
    #   @data_x : list -> [ "hospi + ',' + path_id", ... ]
    #
    def add_augment_list(self, data_x, labels):
        df = pd.read_csv(defs.aug_anno_files[0])
        if type(data_x) == np.ndarray:
            data_x = data_x.tolist()
        if type(labels) == np.ndarray:
            labels = labels.tolist()

        for index, row in df.iterrows():
            # data_x에 있는 이미지가 augmented된 것만 추가
            # prefix 'a' or 'aa' 제거
            path_no = 'S' + row['path_no'].split('S')[-1]
            aug_data_x_wo_prefix = row['hospital'] + ',' + path_no
            if aug_data_x_wo_prefix in data_x:
                data_x.append(row['hospital'] + ',' + row['path_no'])
                labels.append(row['label'])

        return data_x, labels


    # Global Average Pooling
    def global_avg_pool(self, input):
        pass



    
    
    # patch_select()
    # @crop_df  : must be sorted by probability
    # @method   :   "label_prob"        : label=1이면 prob high thres prob 보다 큰 patch 선택
    #               "label_percent"     : label=1이면 prob high thres percentage에 해당하는 patch 선택
    #               "label_cnt"         : label=1이면 prob high thres count 만큼 patch 선택
    #               "bipolar_prob"      : High, Low threshold 만족하는 양 극단 모두 선택
    #               "bipolar_percent"
    #               "bipolar_cnt"
    def patch_select(self, crop_df, label, thr_h, thr_l, method='label_percent'):
        if method == 'label_prob':
            crop_df = self._patch_sel_label_prob(crop_df, label, thr_h, thr_l)
        elif method == 'label_percent':
            crop_df = self._patch_sel_label_percent(crop_df, label, thr_h, thr_l)
        elif method == 'label_cnt':
            crop_df = self._patch_sel_label_cnt(crop_df, label, thr_h, thr_l)
        # elif method == 'bipolar_prob':
        #     crop_df = self._patch_sel_bipolar_prob(crop_df, label, thr_h, thr_l)
        # elif method == 'bipolar_percent':
        #     crop_df = self._patch_sel_bipolar_percent(crop_df, label, thr_h, thr_l)
        # elif method == 'bipolar_cnt':
        #     crop_df = self._patch_sel_bipolar_cnt(crop_df, label, thr_h, thr_l)

        return crop_df


    def _patch_sel_label_prob(self, crop_df, label, thr_h, thr_l):
        if label == 1:
            crop_df = crop_df[crop_df.cell_prob > thr_h]
        else:
            crop_df = crop_df[crop_df.cell_prob < thr_l]
        return crop_df

    def _patch_sel_label_percent(self, crop_df, label, thr_h, thr_l):
        total_crop_cnt = len(crop_df.index)
        if label == 1:
            start_idx = int((1-thr_h) * total_crop_cnt)
            crop_df = crop_df[start_idx:]
        else:
            end_idx = int(thr_l * total_crop_cnt)
            crop_df = crop_df[:end_idx]
        return crop_df

    def _patch_sel_label_cnt(self, crop_df, label, thr_h, thr_l):
        total_crop_cnt = len(crop_df.index)
        if label == 1:
            start_idx = int(total_crop_cnt - thr_h)
            crop_df = crop_df[start_idx:]
        else:
            end_idx = thr_l
            crop_df = crop_df[:end_idx]
        return crop_df
