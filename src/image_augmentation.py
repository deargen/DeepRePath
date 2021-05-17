import os
import random
import numpy as np
import pandas as pd
import _pickle as cpickle
import warnings
import sys
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.transform import resize
from shutil import rmtree
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import argparse
import json
from pathlib import Path
    
from ImageFactory import ImageFactory
import defs



def image_augmentation(args):
    print('image_augmentation() ...')
    anno_file = args['anno_files'][0]
    aug_anno_file = args['aug_anno_files'][0]
    root = args['img_root_path']
    use_x = args['use_x']

    anno_df = pd.read_csv(anno_file)
    aug_anno_df = pd.DataFrame(columns=['hospital', 'path_no', 'label', 'survival_days'])
    hospi_name = anno_df.hospital.values
    path_ids = anno_df.path_no.values
    labels = anno_df.label.values
    survival_days = anno_df.survival_days.values

    img_factory = ImageFactory()

    # class별 augmentation할 갯수 구함
    major_class, minor_class, major_aug_cnt, minor_aug_cnt = get_aug_class_cnt(anno_file)

    for i in tqdm(range(0,len(labels)), file=sys.stdout):
        for mag_idx, magnificient in enumerate(use_x):
            if magnificient == 'x400':
                x_file_tail = '_5.jpg'
            if magnificient == 'x200':
                x_file_tail = '_4.jpg'
            if magnificient == 'x100':
                x_file_tail = '_3.jpg'
            if magnificient == 'x40':
                x_file_tail = '_2.jpg'

            img_file = get_img_file_name(root, hospi_name[i], path_ids[i], x_file_tail)

            try:
                org_img = imread(img_file)[:,:,:3]
            except:
                # magnificient 중 하나라도 이미지가 없으면 둘다 무시
                break
            
            ##
            # augment image only for the recur (prefix "a") 
            if labels[i] == major_class:
                aug_cnt = major_aug_cnt
            else:
                aug_cnt = minor_aug_cnt
            aug_anno_df = augment_img(org_img, aug_cnt, aug_anno_df, root, use_x, hospi_name[i], path_ids[i], labels[i], survival_days[i], x_file_tail, mag_idx)

    # save aug_cli data
    aug_anno_df.to_csv(aug_anno_file, index=False)


# Image augmentation & save
# @aug_cnt      : augmenting img cnt excluding org_img
# return    : augment anno list dataframe
def augment_img(img, aug_cnt, aug_anno_df, img_root_path, use_x, hospi_name, path_id, label, survival_days, x_file_tail, mag_idx):
    aug_imgs = _augment_img(img, aug_cnt)
    for idx, aug_img in enumerate(aug_imgs):
        aug_path_id = 'aug' + str(idx) + path_id
        aug_file_name = get_img_file_name(img_root_path, hospi_name, aug_path_id, x_file_tail)
        imsave(aug_file_name, aug_img)
        # annofile update (only for one magnification)
        if mag_idx+1 == len(use_x):
            aug_anno_df = aug_anno_df.append({'hospital' : hospi_name, 'path_no' : aug_path_id, 'label' : label, 'survival_days' : survival_days}, ignore_index=True)

    return aug_anno_df

# Image augmentation
# @cnt      : augmenting img cnt excluding org_img
# return    : image list
def _augment_img(img, cnt):
    imgs = []

    # test img augmentaion
    # datagen = ImageDataGenerator(
    #             rotation_range=40,
    #             width_shift_range=0.2,
    #             height_shift_range=0.2,
    #             shear_range=0.2,
    #             zoom_range=0.2,
    #             horizontal_flip=True,
    #             brightness_range=[0.2,1.0],
    #             fill_mode='nearest')
    datagen = ImageDataGenerator(
                horizontal_flip=True,
                vertical_flip=True,
                featurewise_center=True,
                featurewise_std_normalization=True,
                )

    x = img_to_array(img)

    if cnt > 0:
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            if i >= cnt:
                break
            batch = batch.reshape(img.shape)
            imgs.append(batch)
            i += 1

    return imgs

#
# get_aug_class_cnt()
#   class별 augaugment할 갯수 구하기 (majority와 minority class의 총 갯수를 동일하게 맞춰줌)
#   @ major_multiple    : majority class를 몇 배 augmentation 할것인지
#   return              : (major_class, minor_class, major_class_cnt, minor_class_cnt)
def get_aug_class_cnt(anno_file, major_multiple=2):
    df = pd.read_csv(anno_file)
    class_cnt = np.zeros(2)
    class_cnt[0] = len(df[(df['label']==0)])
    class_cnt[1] = len(df[(df['label']==1)])

    if class_cnt[0] > class_cnt[1]:
        major = 0
        minor = 1
    else:
        major = 1
        minor = 0

    # augmentation후 majority class의 총 갯수
    major_tot_cnt = int(class_cnt[major] * major_multiple)
    major_aug_per_img = int(major_tot_cnt / class_cnt[major] - 1)
    minor_tot_cnt = major_tot_cnt
    minor_aug_per_img = int(minor_tot_cnt / class_cnt[minor] - 1)

    print('major_multiple = ', major_multiple)
    print('major class = ', major, '  cnt = ', int(class_cnt[major]))
    print('minor class = ', minor, '  cnt = ', int(class_cnt[minor]))
    print('major_aug_per_img = ', major_aug_per_img, '  major tot cnt = ', int(class_cnt[major] + (class_cnt[major] * major_aug_per_img)))
    print('minor_aug_per_img= ', minor_aug_per_img, '  minor tot cnt = ', int(class_cnt[minor] + (class_cnt[minor] * minor_aug_per_img)))

    return major, minor, major_aug_per_img, minor_aug_per_img
    
# 각 병원별 이미지 파일 이름 구함
# @aug  : if True, add prefix "aug"
def get_img_file_name(root, hospi_name, path_id, x_file_tail):
    if hospi_name=='Seo' or hospi_name=='Seo2':
        img_file = root+hospi_name+'/'+path_id+x_file_tail
    else:
        if hospi_name=='Inc':
            items = path_id.split('-')
            img_file = root+hospi_name+'/'+items[0]+'-'+items[1]+'-'+str(int(items[2]))+x_file_tail
        else:
            items = path_id.split('-')
            img_file = root+hospi_name+'/'+items[0]+'-'+str(int(items[1]))+x_file_tail
    
    return img_file


# 없어질 함수..
# get label_0_cnt & label_1_cnt
#   return  : aug_label (augmetation할 label),  aug_cnt (augment할 갯수)
def get_labels_cnt(anno_file):
    df = pd.read_csv(anno_file)
    label_0_cnt = len(df[(df['label']==0)])
    label_1_cnt = len(df[(df['label']==1)])

    if label_0_cnt > label_1_cnt:
        aug_label = 1
        aug_cnt = int(label_0_cnt / label_1_cnt)
    elif label_1_cnt > label_0_cnt:
        aug_label = 0
        aug_cnt = int(label_1_cnt / label_0_cnt)
    else: # no augmetation
        aug_cnt = 0

    # except org image
    aug_cnt -= 1

    # test
    print('label_0_cnt = ', label_0_cnt)
    print('label_1_cnt = ', label_1_cnt)
    print('aug_label = ', aug_label)
    print('aug_cnt = ', aug_cnt)

    return aug_label, aug_cnt

#
# remove augmented image (prefix 'a' or 'aa')
#
def remove_aug_images(root):
    for path in Path(root).rglob('aug*'):
        if os.path.isfile(path.absolute()):
            print('!!! removing ', path.absolute())
            os.remove(path.absolute())



def save_matched_clinical(args):
    anno_file = args['anno_files'][0]
    root = args['img_root_path']
    use_x = args['use_x']

    anno_df = pd.read_csv(anno_file)
    matched_anno_df = anno_df.copy()
    matched_anno_df['matched'] = 0

    # test
    print(matched_anno_df)

    hospi_name = anno_df.hospital.values
    path_ids = anno_df.path_no.values

    img_factory = ImageFactory()

    for index, row in matched_anno_df.iterrows():
        for mag_idx, magnificient in enumerate(use_x):
            if magnificient == 'x400':
                x_file_tail = '_5.jpg'
            if magnificient == 'x200':
                x_file_tail = '_4.jpg'
            if magnificient == 'x100':
                x_file_tail = '_3.jpg'
            if magnificient == 'x40':
                x_file_tail = '_2.jpg'

            img_file = get_img_file_name(root, hospi_name[index], path_ids[index], x_file_tail)
            # test
            print(img_file)

            try:
                org_img = imread(img_file)[:,:,:3]
            except:
                # magnificient 중 하나라도 이미지가 없으면 둘다 무시
                break
            
            matched_anno_df.loc[index, 'matched'] = 1

    matched_anno_df = matched_anno_df[matched_anno_df['matched']==1]
    matched_file = anno_file.split('.csv')[0] + '_m.csv'
    matched_anno_df.to_csv(matched_file, index=False)



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

    # # test
    # save_matched_clinical(args)

    # remove existing augmented images
    ans = input("Removing augmented images (y/n):")
    if ans.lower() == 'y':
        remove_aug_images(args["img_root_path"])    
        
    image_augmentation(args)