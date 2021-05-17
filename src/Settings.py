import enum

class InputDataType(enum.Enum):
    RESNET50 = 0
    XGB = 1
    SVM = 2
    RF = 3
    ADABOOST = 4
    GRADIENTBOOST = 5

class ModelType(enum.Enum):
    RESNET50 = 0
    XGB = 1
    SVM = 2
    RF = 3
    ADABOOST = 4
    GRADIENTBOOST = 5


class Settings:
    @staticmethod
    def input_data_type():
        return InputDataType.XGB

    @staticmethod 
    def model_type():
        return ModelType.XGB


 
