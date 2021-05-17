from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import cv2
from ColorBalance import ColorBalance
from random import randint
import numpy as np
import os
from shutil import rmtree
import warnings
from os.path import join

class ImageFactory(object):

    # crop_size로 자른후 224x224로 resize
    def get_crop_imgs(self, img_name='',crop_size=224, bottom_cut=100, crop_divider=2, white_space_filtering=False, fix_crop_size=False, random_crop_n=0, random_crop_center=False, color_normalization=False, max_img_size=(1360, 1024), use_rect_annotation=False):
        crops = []
        start_points = []

        warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

        if(crop_size < 1):
            return crops
        
        #img = cv2.imread(img_name)
        img = imread(img_name)[:,:,:3]
        if (img is None):
            #print('cv2.imread error in get_crop_imgs(): ', img_name)
            return crops

        # # test
        # cv2.imshow('before', img)

        # # test
        # cv2.imshow('norm', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit(0)

        img_height = img.shape[0]
        img_width = img.shape[1]

        # resize image to the common size (가로세로 비율 유지, height기준으로 맞춤)
        if (img_width != max_img_size[0]) or (img_height != max_img_size[1]):
            new_hgt = max_img_size[1]
            ratio = img_width / img_height
            new_wdt = int(new_hgt * ratio)
            img = resize(img, (new_hgt, new_wdt), anti_aliasing=True)
            rescaled_image = 255 * img
            # Convert to integer data type pixels.
            img = rescaled_image.astype(np.uint8)
       
        # annotations된 roi 영역들 구하기
        anno_areas = self.__get_rect_anno_areas(img, use_rect_annotation, color_normalization)

        # test
        # print('get_crop_imgs()... : ', img_name)
        # self.save_images('../cache/anno_areas/', anno_areas)

        # anno_areas에서는 bottom_cut 사용 안함 (anno_areas에는 이미지 아래에 배율 표시가 없을테니까)
        if len(anno_areas) > 1:
            bottom_cut = 0

        for anno_area in anno_areas:
            img_height = anno_area.shape[0] - bottom_cut
            img_width = anno_area.shape[1]

            local_crop_size, crop_num_x, crop_num_y, sliding_x, sliding_y = self.get_crop_num(img_width, img_height, crop_size, crop_divider, fix_crop_size=fix_crop_size)
            # # test
            # print('anno_area shape:', anno_area.shape)

            if crop_num_x < 1 or crop_num_y < 1:
                continue
            
            # crop images starting point 구하기 (start_x, start_y)
            if random_crop_n > 0:   # n 개의 random crop
                for k in range(random_crop_n):
                    if random_crop_center:  # 중앙 근처에서 crop
                        x_divide_cnt = int(crop_num_x / 3)
                        y_divide_cnt = int(crop_num_y / 3)
                        range_start_x = int(local_crop_size * x_divide_cnt)
                        range_end_x = int(local_crop_size * x_divide_cnt * 2)
                        range_start_y = int(local_crop_size * y_divide_cnt)
                        range_end_y = int(local_crop_size * y_divide_cnt * 2)
                        start_x = randint(range_start_x, range_end_x)
                        start_y = randint(range_start_y, range_end_y)
                        start_points.append((start_x, start_y))
                    else:
                        start_x = randint(0, img_width - local_crop_size)
                        start_y = randint(0, img_height - local_crop_size)
                        start_points.append((start_x, start_y))
            else: # 전체 이미지에서 고르게 crop
                for y_cnt in range(crop_num_y):
                    for x_cnt in range(crop_num_x):
                        start_x = x_cnt * sliding_x
                        start_y = y_cnt * sliding_y
                        start_points.append((start_x, start_y))

            # crop image 만들기
            k = 0
            while k < len(start_points):
                (start_x, start_y) = start_points[k]
                end_x = start_x + local_crop_size 
                end_y = start_y + local_crop_size

                # error check
                excess = max(end_x - img_width, end_y - img_height)
                if excess > 0:
                    k += 1
                    continue

                crop_img = anno_area[start_y:end_y, start_x:end_x, :]

                # # test
                # print('   crop_img shape: ', crop_img.shape)
                # print('      excess = ', excess)

                if local_crop_size != 224:
                    crop_img = resize(crop_img, (224, 224), anti_aliasing=True)
                    rescaled_image = 255 * crop_img
                    # Convert to integer data type pixels.
                    crop_img = rescaled_image.astype(np.uint8)

                # white space filtering
                if(white_space_filtering == False):
                    crops.append(crop_img)
                elif(self.is_white_area(crop_img) == False):
                    crops.append(crop_img)

                # augment images #1
                crop_img = self.stain_normalize(crop_img)
                crops.append(crop_img)

                k += 1

        return crops

    # overlap 안 되게 224x224로 crop.   이미지 끝부분 남는 부분은 날림  (visualization할때 필요해서 임시로 만든 함수)
    #    ??? get_crop_imgs()와 통합할 것  (현재 get_crop_imgs()보다 old version임 !!!  업데이트 필요)
    def get_crop_imgs_non_overlap(self, img_name='',crop_size=224, bottom_cut=100, crop_divider=2, white_space_filtering=False):
        crops = []
        start_points = []

        warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
        try:
            img = imread(img_name)#[:,:,:IMG_CHANNELS]
        except Exception as e:
            #print('image read fail : '+ str(e))
            return crops

        img_height = img.shape[0]
        img_width = img.shape[1]
        
        # annotations된 roi 영역들 구하기
        anno_areas = self.__get_rect_anno_areas(img, use_rect_annotation)

        # test
        # print('get_crop_imgs()... : ', img_name)
        # self.save_images('../cache/anno_areas/', anno_areas)

        # anno_areas에서는 bottom_cut 사용 안함 (anno_areas에는 이미지 아래에 배율 표시가 없을테니까)
        if len(anno_areas) > 1:
            bottom_cut = 0

        for anno_area in anno_areas:
            img_height = anno_area.shape[0] - bottom_cut
            img_width = anno_area.shape[1]

            # sliding : crop_image를 몇픽셀씩 이동해서 만들지
            # slide_num : x축, y축별로 몇 등분할지.   (x, y축 동일한 숫자로 등분)
            slide_num_x = int(np.floor(img_width / 224))
            slide_num_y = int(np.floor(img_height / 224))
            sliding_x = 224
            sliding_y = 224

            for y_cnt in range(slide_num_y):
                for x_cnt in range(slide_num_x):
                    start_x = x_cnt * sliding_x
                    start_y = y_cnt * sliding_y
                    start_points.append((start_x, start_y))

            # crop image 만들기
            k = 0
            while k < (slide_num_x * slide_num_y):
                (start_x, start_y) = start_points[k]
                end_x = start_x + crop_size 
                end_y = start_y + crop_size

                # error check
                excess = max(end_x - img_width, end_y - img_height)
                if excess > 0:
                    k += 1
                    continue

                crop_img = anno_area[start_y:end_y, start_x:end_x, :]

                # # test
                # print('   crop_img shape: ', crop_img.shape)
                # print('      excess = ', excess)

                if crop_size != 224:
                    crop_img = resize(crop_img, (224, 224), anti_aliasing=True)
                    rescaled_image = 255 * crop_img
                    # Convert to integer data type pixels.
                    crop_img = rescaled_image.astype(np.uint8)

                # white space filtering
                if(white_space_filtering == False):
                    crops.append(crop_img)
                elif(self.is_white_area(crop_img) == False):
                    crops.append(crop_img)

                # augment images #1
                crop_img = self.stain_normalize(crop_img)
                crops.append(crop_img)

                k += 1

        return crops


    # get_crop_position
    #   crop position을 찾아줌  (crop image를 외부에서 만들 때 사용 => 메모리 문제해결)
    #   return : (start_x, start_y, end_x, end_y)
    def get_crop_position(self, img_wdt, img_hgt, crop_size=224, bottom_cut=100, crop_divider=2, fix_crop_size=False, random_crop_n=0, random_crop_center=False, use_rect_annotation=False):
        img_width = img_wdt
        img_height = img_hgt - bottom_cut

        local_crop_size, crop_num_x, crop_num_y, sliding_x, sliding_y = self.get_crop_num(img_width, img_height, crop_size, crop_divider, fix_crop_size=fix_crop_size)

        if crop_num_x < 1 or crop_num_y < 1:
            print('error in get_crop_position() : crop_num_x or crop_num_y is less than 1 (%d, %d) ' % (crop_num_x, crop_num_y))
            exit(0)
        
        

        # crop images starting point 구하기 (start_x, start_y)
        crop_points = []
        if random_crop_n > 0:   # n 개의 random crop
            for k in range(random_crop_n):
                if random_crop_center:  # 중앙 근처에서 crop
                    x_divide_cnt = int(crop_num_x / 3)
                    y_divide_cnt = int(crop_num_y / 3)
                    range_start_x = int(local_crop_size * x_divide_cnt)
                    range_end_x = int(local_crop_size * x_divide_cnt * 2)
                    range_start_y = int(local_crop_size * y_divide_cnt)
                    range_end_y = int(local_crop_size * y_divide_cnt * 2)
                    start_x = randint(range_start_x, range_end_x)
                    start_y = randint(range_start_y, range_end_y)
                else:
                    start_x = randint(0, img_width - local_crop_size)
                    start_y = randint(0, img_height - local_crop_size)
                
                end_x = start_x + local_crop_size
                end_y = start_y + local_crop_size
                excess = max(end_x - img_width, end_y - img_height)
                if excess > 0:
                    continue
                crop_points.append((start_x, start_y, end_x, end_y))
        else: # 전체 이미지에서 고르게 crop
            for y_cnt in range(crop_num_y):
                for x_cnt in range(crop_num_x):
                    start_x = x_cnt * sliding_x
                    start_y = y_cnt * sliding_y
                    end_x = start_x + local_crop_size
                    end_y = start_y + local_crop_size

                    excess = max(end_x - img_width, end_y - img_height)
                    if excess > 0:
                        continue
                    crop_points.append((start_x, start_y, end_x, end_y))

        return crop_points
                
                


    # save images
    #   @imgs : numpy array [img_cnt, row, col, colors]
    #   @rewrite : if True remove target_path and recreate it
    #   @img_names : imgs의 이름들 (확장자나 dir정보는 없어야함).  없으면 자동 생성 
    #   @suffix : 파일 이름 끝에 붙는 string
    #   return : saved image count
    def save_images(self, target_path, imgs, img_names=[], rewrite=True, suffix=''):
        if len(img_names) > 0 and len(imgs) != len(img_names):
            print('Error in save_images() : len(imgs) != len(img_names)')
            return 0

        cnt = 0

        for i, img in enumerate(imgs):
            tail = '_' + suffix + '.png'
            if len(img_names) > 0:
                file_name = target_path + img_names[i] + tail
            else:
                file_name = target_path + "{:02d}".format(i) + tail
            try:
                #cv2.imwrite(file_name, img)
                imsave(file_name, img)
            except Exception as e:
                print(e)

            cnt += 1

        return cnt

    # 이미지상의 white area 정도를 계산하여 True, False return
    def is_white_area(self, img):

        thr_mean = 200.0  #190.0
        img_mean = self.__get_pixels_mean(img)
        # test
        #print('img_mean = ', img_mean)
        if thr_mean < img_mean:
            return True
        else:
            return False

    def is_flat_area(self, img):
        thr_var = 100000.0 #2000.0
        if thr_var < self.__get_pixels_var(img):
            return True
        else:
            return False


    # get crop num
    #   이미지 crop_size로 crop할 때 몇개의 crop 수가 나오는지 구함 (divider에 의해 결정)
    #   crop image들이 대략 반이 겹치게 하기 위해서는 divider=2
    def get_crop_num(self, img_width, img_height, crop_size, divider=2, fix_crop_size=False):
        local_crop_size = crop_size

        # image size가 원하는 crop_size보다 작을 때 crop_size 조정
        if(fix_crop_size == False):
            if(local_crop_size > img_width):
                local_crop_size = img_width
            if(local_crop_size > img_height):
                local_crop_size = img_height

        if(local_crop_size < 1):
            return 0, 0, 0

        crop_num_x = int(np.ceil(img_width / (local_crop_size/divider)))
        crop_num_y = int(np.ceil(img_height / (local_crop_size/divider)))

        # sliding : crop_image를 몇픽셀씩 이동해서 만들지
        if(crop_num_x > 1):
            sliding_x = int(np.floor((img_width - local_crop_size) / (crop_num_x - 1)))
        else:
            sliding_x = 0

        if(crop_num_y > 1):
            sliding_y = int(np.floor((img_height - local_crop_size) / (crop_num_y - 1)))
        else:
            sliding_y = 0

        return local_crop_size, crop_num_x, crop_num_y, sliding_x, sliding_y

    # get image width, height
    def get_image_size(self, img_name):
        try:
            img = imread(img_name)#[:,:,:IMG_CHANNELS]
        except Exception as e:
            print('Error in get_image_size() : image read fail : '+ str(e))
            return (0, 0)

        img_height = img.shape[0]
        img_width = img.shape[1]

        return img_width, img_height


    # rectangle로 annotation된 roi 영역 list 리턴
    def __get_rect_anno_areas(self, img='', use_rect_annotation=False, color_normalization=False):
        anno_areas = []

        if len(img.shape) > 2 and img.shape[2] == 4:
            #convert the image from RGBA2RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if use_rect_annotation == False:
            anno_areas = []
            anno_areas.append(img)
            return anno_areas

        # annotation rectangle만 흰색으로 뽑아냄
        lower_color = np.array([0, 115, 0])
        upper_color = np.array([100, 155, 150])
        anno_rect = cv2.inRange(img, lower_color, upper_color)
        contours, hierarchy = cv2.findContours(anno_rect, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        i = 0
        for i, contour in enumerate(contours):
            # outter contour 제외
            if hierarchy[0][i][2] != -1:
                continue

            (x,y,w,h) = cv2.boundingRect(contour)

            # 너무작은 crop 제외
            if w < 224 or h < 224:
                continue

            x_end = x + w
            y_end = y + h
            margin = 5
            x += margin
            y += margin
            x_end -= margin
            y_end -= margin

            crop_img = img[y:y_end, x:x_end, :]

            # color normalization
            if(color_normalization):
                crop_img = self.stain_normalize(crop_img)

            anno_areas.append(crop_img)
            #cv2.rectangle(img, (x,y), (x_end,y_end), (0,255,0), 1)

        # annotation rectangle이 없는 경우 전체 이미지 사용
        if i == 0:
            anno_areas.append(img)

        return anno_areas

    # 이미지 pixel mean 계산
    def __get_pixels_mean(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flat_img = [item for sublist in gray for item in sublist]
        return np.mean(flat_img)

    def __get_pixels_var(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flat_img = [item for sublist in gray for item in sublist]
        return np.var(flat_img)

    # histogram equlization
    def stain_normalize(self, img):
        #return self.__histo_equ(img)
        return self.__clahe(img)

    def __histo_equ(self, img):
        b,g,r = cv2.split(img)

        # # test
        # cv2.imshow('b', b)
        # cv2.imshow('g', g)
        # cv2.imshow('r', r)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit(0)

        equ_b = cv2.equalizeHist(b)
        equ_g = cv2.equalizeHist(g)
        equ_r = cv2.equalizeHist(r)
        equ = cv2.merge((equ_b, equ_g, equ_r))

        return equ

    # Contrast Limited Adaptive Histogram Equalization)
    def __clahe(self, img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # LAB color space로 변환
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        return image
    
