import os
import numpy as np
import math
import rdkit.Chem as Chem

import matplotlib.pyplot as plt
import seaborn as sns


class ComponentSpecificParametersEnum:
    __LOW = "low"
    __HIGH = "high"
    __K = "k"
    __TRANSFORMATION = "transformation"
    __SCIKIT = "scikit"
    __CLAB_INPUT_FILE = "clab_input_file"
    __COEF_DIV = "coef_div"
    __COEF_SI = "coef_si"
    __COEF_SE = "coef_se"
    __TRANSFORMATION_TYPE = "transformation_type"
    __DESCRIPTOR_TYPE = "descriptor_type"

    @property
    def LOW(self):
        return self.__LOW

    @LOW.setter
    def LOW(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def HIGH(self):
        return self.__HIGH

    @HIGH.setter
    def HIGH(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def K(self):
        return self.__K

    @K.setter
    def K(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def TRANSFORMATION(self):
        return self.__TRANSFORMATION

    @TRANSFORMATION.setter
    def TRANSFORMATION(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def SCIKIT(self):
        return self.__SCIKIT

    @SCIKIT.setter
    def SCIKIT(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def CLAB_INPUT_FILE(self):
        return self.__CLAB_INPUT_FILE

    @CLAB_INPUT_FILE.setter
    def CLAB_INPUT_FILE(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def COEF_DIV(self):
        return self.__COEF_DIV

    @COEF_DIV.setter
    def COEF_DIV(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def COEF_SI(self):
        return self.__COEF_SI

    @COEF_SI.setter
    def COEF_SI(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def COEF_SE(self):
        return self.__COEF_SE

    @COEF_SE.setter
    def COEF_SE(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def TRANSFORMATION_TYPE(self):
        return self.__TRANSFORMATION_TYPE

    @TRANSFORMATION_TYPE.setter
    def TRANSFORMATION_TYPE(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def DESCRIPTOR_TYPE(self):
        return self.__DESCRIPTOR_TYPE

    @DESCRIPTOR_TYPE.setter
    def DESCRIPTOR_TYPE(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")


class TransformationTypeEnum:
    __DOUBLE_SIGMOID = "double_sigmoid"
    __SIGMOID = "sigmoid"
    __REVERSE_SIGMOID = "reverse_sigmoid"
    __RIGHT_STEP = "right_step"
    __STEP = "step"
    __NO_TRANSFORMATION = "no_transformation"

    @property
    def DOUBLE_SIGMOID(self):
        return self.__DOUBLE_SIGMOID

    @DOUBLE_SIGMOID.setter
    def DOUBLE_SIGMOID(self, value):
        raise ValueError("Do not assign value to a TransformationTypeEnum field")

    @property
    def SIGMOID(self):
        return self.__SIGMOID

    @SIGMOID.setter
    def SIGMOID(self, value):
        raise ValueError("Do not assign value to a TransformationTypeEnum field")

    @property
    def REVERSE_SIGMOID(self):
        return self.__REVERSE_SIGMOID

    @REVERSE_SIGMOID.setter
    def REVERSE_SIGMOID(self, value):
        raise ValueError("Do not assign value to a TransformationTypeEnum field")

    @property
    def RIGHT_STEP(self):
        return self.__RIGHT_STEP

    @RIGHT_STEP.setter
    def RIGHT_STEP(self, value):
        raise ValueError("Do not assign value to a TransformationTypeEnum field")

    @property
    def STEP(self):
        return self.__STEP

    @STEP.setter
    def STEP(self, value):
        raise ValueError("Do not assign value to a TransformationTypeEnum field")

    @property
    def NO_TRANSFORMATION(self):
        return self.__NO_TRANSFORMATION

    @NO_TRANSFORMATION.setter
    def NO_TRANSFORMATION(self, value):
        raise ValueError("Do not assign value to a TransformationTypeEnum field")


class TransformationFactory:

    def __init__(self):
        self._csp_enum = ComponentSpecificParametersEnum()
        self._transformation_function_registry = self._default_transformation_function_registry()

    def _default_transformation_function_registry(self) -> dict:
        enum = TransformationTypeEnum()
        transformation_list = {
            enum.SIGMOID: self.sigmoid_transformation,
            enum.REVERSE_SIGMOID: self.reverse_sigmoid_transformation,
            enum.DOUBLE_SIGMOID: self.double_sigmoid,
            enum.NO_TRANSFORMATION: self.no_transformation,
            enum.RIGHT_STEP: self.right_step,
            enum.STEP: self.step
        }
        return transformation_list

    def get_transformation_function(self, parameters: dict):
        transformation_type = parameters[self._csp_enum.TRANSFORMATION_TYPE]
        transformation_function = self._transformation_function_registry[transformation_type]
        return transformation_function

    def no_transformation(self, predictions: list, parameters: dict) -> np.array:
        return np.array(predictions, dtype=np.float32)

    def right_step(self, predictions, parameters) -> np.array:
        _low = parameters[self._csp_enum.LOW]

        def _right_step_formula(value, low):
            if value >= low:
                return 1
            return 0

        transformed = [_right_step_formula(value, _low) for value in predictions]
        return np.array(transformed, dtype=np.float32)

    def step(self, predictions, parameters) -> np.array:
        _low = parameters[self._csp_enum.LOW]
        _high = parameters[self._csp_enum.HIGH]

        def _right_step_formula(value, low, high):
            if low <= value <= high:
                return 1
            return 0

        transformed = [_right_step_formula(value, _low, _high) for value in predictions]
        return np.array(transformed, dtype=np.float32)

    def sigmoid_transformation(self, predictions: list, parameters: dict) -> np.array:
        _low = parameters[self._csp_enum.LOW]
        _high = parameters[self._csp_enum.HIGH]
        _k = parameters[self._csp_enum.K]

        def _exp(pred_val, low, high, k) -> float:
            return math.pow(10, (10 * k * (pred_val - (low + high) * 0.5) / (low - high)))

        transformed = [1 / (1 + _exp(pred_val, _low, _high, _k)) for pred_val in predictions]
        return np.array(transformed, dtype=np.float32)

    def reverse_sigmoid_transformation(self, predictions: list, parameters: dict) -> np.array:
        _low = parameters[self._csp_enum.LOW]
        _high = parameters[self._csp_enum.HIGH]
        _k = parameters[self._csp_enum.K]

        def _reverse_sigmoid_formula(value, low, high, k) -> float:
            try:
                return 1 / (1 + 10 ** (k * (value - (high + low) / 2) * 10 / (high - low)))
            except:
                return 0

        transformed = [_reverse_sigmoid_formula(pred_val, _low, _high, _k) for pred_val in predictions]
        return np.array(transformed, dtype=np.float32)

    def double_sigmoid(self, predictions: list, parameters: dict) -> np.array:
        _low = parameters[self._csp_enum.LOW]
        _high = parameters[self._csp_enum.HIGH]
        _coef_div = parameters[self._csp_enum.COEF_DIV]
        _coef_si = parameters[self._csp_enum.COEF_SI]
        _coef_se = parameters[self._csp_enum.COEF_SE]

        def _double_sigmoid_formula(value, low, high, coef_div=100., coef_si=150., coef_se=150.):
            try:
                A = 10 ** (coef_se * (value / coef_div))
                B = (10 ** (coef_se * (value / coef_div)) + 10 ** (coef_se * (low / coef_div)))
                C = (10 ** (coef_si * (value / coef_div)) / (
                        10 ** (coef_si * (value / coef_div)) + 10 ** (coef_si * (high / coef_div))))
                return (A / B) - C
            except:
                return 0

        transformed = [_double_sigmoid_formula(pred_val, _low, _high, _coef_div, _coef_si, _coef_se) for pred_val in
                       predictions]
        return np.array(transformed, dtype=np.float32)


def render_curve(title, x, y):
    plt.figure(figsize=(16, 10), dpi=80)
    plt.xlabel("input_score")
    plt.ylabel("transformed_score")
    plt.title(title, fontsize=18)
    sns.lineplot(x=x, y=y)
    plt.show()
