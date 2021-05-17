from Settings import *
from ModelResnet50 import ModelResnet50
from ModelXgb import ModelXgb
from ModelSvm import ModelSvm
from ModelRF import ModelRF
from ModelAdaB import ModelAdaB
from ModelGradientB import ModelGradientB

class ModelFactory:
    @staticmethod
    def get_model():
        # input data, model type 결정 (Factory pattern)
        if Settings.model_type() == ModelType.XGB:
            return ModelXgb()
        
        if Settings.model_type() == ModelType.SVM:
            return ModelSvm()

        if Settings.model_type() == ModelType.RF:
            return ModelRF()

        if Settings.model_type() == ModelType.RESNET50:
            return ModelResnet50()

        if Settings.model_type() == ModelType.ADABOOST:
            return ModelAdaB()

        if Settings.model_type() == ModelType.GRADIENTBOOST:
            return ModelGradientB()