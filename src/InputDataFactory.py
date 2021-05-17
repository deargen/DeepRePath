from Settings import *
from InputDataReset50 import InputDataResnet50
from InputDataXgb import InputDataXgb

class InputDataFactory:
    @staticmethod
    def get_input_data():
        # input data, model type 결정 (Factory pattern)
        if Settings.input_data_type() == InputDataType.XGB:
            return InputDataXgb()

        if Settings.input_data_type() == InputDataType.RESNET50:
            return InputDataResnet50()