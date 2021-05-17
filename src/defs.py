model_save_path = './model_save/'
cache_path = '../cache/'
crop_path = cache_path + 'crops/'
model_name = 'xgb_recur_clf.hd5'
x400_patch_sel_name = 'xgb_x400_recur_clf.bin'
x200_patch_sel_name = 'xgb_x200_recur_clf.bin'
x100_patch_sel_name = 'xgb_x100_recur_clf.bin'

# 병원마다 이미지 크기가 달라서 cell 크기를 동일하게 맞추기 위해 기준 이미지 사이즈를 정하고 모든 이미지를 이 크기로 resize 함
max_img_size = (1360, 1024)#(1600, 1200)


#######################################
# config.json에서 읽어오는 부분
anno_files = []
aug_anno_files = []

calibration = False
cascading = False

pretrain = ""
pretrain_file = ""

# crop divider
crop_div_x400 = 1
crop_div_x200 = 1
crop_div_x100 = 1

# patch selector를 만들 때 cascading방식으로 반복해서 만드는 횟수  (1 이상이어야 함)
patch_sel_iter = 0
# patch selector가 완성된후 실제 patch selection을 하기 위한 threshold index  (아래 배열에 대한 index)
patch_thres_idx = 0

# patch selection methods:
# "label_prob"        : label=1이면 high thres prob 보다 큰 patch 선택
# "label_percent"     : label=1이면 high thres percentage에 해당하는 갯수의 상위 prob patch 선택
# "label_cnt"         : label=1이면 prob high thres count 만큼 patch 선택
# "bipolar_prob"      : High, Low threshold 만족하는 양 극단 모두 선택
# "bipolar_percent"
# "bipolar_cnt"
patch_sel_method = ''

# patch selection thresold
thres_h_x400 = []
thres_l_x400 = []
thres_h_x200 = []
thres_l_x200 = []
thres_h_x100 = []
thres_l_x100 = []

## 
do_remove_all_model = False
do_save_crop = False
do_train_patch_sel = False
do_train_kfold = False
do_train_full = False
ext_val_models = []
do_train_set_combination = False
do_data_augmentation = False

do_load_dump_patch_sel = False
do_load_dump_train_kfold = False
do_load_dump_train_full = False
do_load_dump_exst_val = False
do_save_crop_only = False