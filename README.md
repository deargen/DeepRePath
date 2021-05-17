# DeepRePath
An official lung adenocarcinoma prognosis prediction model

* **Author**: [Won Sang Shim](mailto:wsshim@deargen.me)
* **Paper**: Won Sang Shim, Kwangil Yim, Tae-Jung Kim, Ji Hyung Hong, Sang Hoon Chun, Seoree Kim, Ho Jung An, Jae Jun Kim, Mi Hyoung Moon, Seok Whan Moon, Sungsoo Park, Soon Auck Hong, Yoon Ho Ko (2021). [DeepRePath: identifying the prognostic features of early-stage lung adenocarcinoma using multi-scale pathology images and deep convolutional neural networks
]
## Required Files

1. Annotation csv file (clinical_data.csv)
    * Annotation csv file should have three columns : hospital, path_no, label
    * Sample anotation file which was used for this model is in the data/clinical_data/
2. Pathological slide images
    * Images should be in the data/images/"hospital_name"/
    * A patient should have two images. One is a x100 magnification image and the other is a x400 magnification image.
    * The naming rule of the image : "pathology_number_magnificationID.jpg(.png)"
    * The magnification ID of x100 is "_3", and the magnification ID of x400 is "_5"
    * e.g. "S10-012345_3.jpg"
    * The images which was used for this paper are not publicly available due to privacy and ethical restrictions.


## VirtualEnv

* create a conda env with the following commands

```
conda create -n my_env python==3.7
conda activate my_env
pip install tensorflow-gpu==1.15.4
conda install biopython
pip install keras==2.3.1
conda install scikit-image
conda install scikit-learn
conda install xlrd
pip install opencv-python
conda install pandas
pip install keras-swa
conda install xgboost
conda install lifelines
```


## Training & Validation

```
cd src
bash run.sh
```



