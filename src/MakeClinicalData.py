###################################################
# 병원별 임상데이터들을 모아서 학습에 필요한 항목만 정리하고 합침

import os
import pandas as pd
import numpy as np
from skimage.io import imread
import warnings
from EnvParam import EnvParam
from Helper import Helper

class MakeClinicalData(object):
    def __init__(self, env_params='', helper=''):
        self.params = env_params
        self.helper = helper

    def run(self):
        df_abnormal, df_normal = self.helper.make_data_frame(input_type=self.params.input_type, output_type=self.params.output_type)
        df_abnormal['label'] = 1
        df_normal['label'] = 0

        # merge
        df_clinical = df_abnormal.append(df_normal, ignore_index=True)
        df_clinical = df_clinical[['hospital', 'path_no', 'label', 'survival_days']]
        print(df_clinical.head())

        result_file = os.path.join(self.params.result_save_path, 'clinical_data.csv')
        df_clinical.to_csv(result_file)
        print('len(clinical_data) = ', len(df_clinical))

        # slide matching
        df_cli_match = self.sel_slide_matching(df_clinical)
        result_file = os.path.join(self.params.result_save_path, 'clinical_data_m.csv')
        df_cli_match.to_csv(result_file)
        print('len(clinical_data_m) = ', len(df_cli_match))


    def sel_slide_matching(self, df_input):

        df_cli_match = pd.DataFrame(columns=df_input.columns)
        new_rows = []

        for idx, row in df_input.iterrows():

            hospital = row['hospital']
            patho_num = row['path_no']
            if hospital != 'Seo' and hospital != 'Seo2' and hospital != 'TMA':
                patho_item = patho_num.split('-')               
                num = str(int(patho_item[-1]))  # 예) 012345 -> 12345

                patho_num = ''
                for i in range(len(patho_item) - 1):    # s10, s-10 들을 다 처리하기 위한 for loop
                    patho_num += str(patho_item[i]) + '-'   # 예) s10-  또는 S-10-
                patho_num += num    # 예) s10-12345

            # 공백제거
            patho_num = patho_num.strip()
            raw_file_base = '../imgs_2020_04_18/' + hospital + '/' + patho_num

            existSlide = False

            raw_file_name1 = raw_file_base + '_5.jpg'
            raw_file_name2 = raw_file_base + '_5.png'
            raw_file_name3 = raw_file_base + '_5' + '_00.jpg'
            raw_file_name4 = raw_file_base + '_5' + '_00.png'

            if os.path.isfile(raw_file_name1) or os.path.isfile(raw_file_name2) or os.path.isfile(raw_file_name3) or os.path.isfile(raw_file_name4):
                raw_file_name1 = raw_file_base + '_3.jpg'
                raw_file_name2 = raw_file_base + '_3.png'
                raw_file_name3 = raw_file_base + '_3' + '_00.jpg'
                raw_file_name4 = raw_file_base + '_3' + '_00.png'
                if os.path.isfile(raw_file_name1) or os.path.isfile(raw_file_name2) or os.path.isfile(raw_file_name3) or os.path.isfile(raw_file_name4):
                    new_rows.append(row.tolist())
            else:
                continue

        df_cli_match = df_cli_match.append(pd.DataFrame(new_rows, columns=df_cli_match.columns)).reset_index()

        print(df_cli_match.head())
        return df_cli_match


if __name__ == '__main__':
    envParam = EnvParam()
    helper = Helper(recur_day_cut_off = 365*3, normal_day_cut_off = 365*3, env_params=envParam)

    make_clinical_data = MakeClinicalData(env_params=envParam, helper=helper)
    make_clinical_data.run()
