import attr
from typing import List


# A very thin base class for all
# the specification of the terms
class OptimizationTermSpec(object):
    # The type of this method
    @staticmethod
    def type_name():  # type: () -> str
        raise NotImplementedError

    @staticmethod
    def is_cost():  # type: () -> bool
        raise NotImplementedError

    # The method for serialization
    def to_dict(self):
        raise NotImplementedError

    def from_dict(self, data_map):
        raise NotImplementedError


# For the specification of constraint types:
@attr.s
class Point2PointConstraintSpec(OptimizationTermSpec):
    # 1 - tolerance <= dot(from, to) <= 1
    keypoint_name = ""  # type: str
    keypoint_idx = -1  # type: int
    target_position = [0, 0, 0]  # type: List[float]
    tolerance = 0.001  # type: float

    @staticmethod
    def type_name():
        return "point2point_constraint"

    @staticmethod
    def is_cost():  # type: () -> bool
        return False

    def to_dict(self):
        constraint_dict = dict()
        constraint_dict["keypoint_name"] = self.keypoint_name
        constraint_dict["keypoint_idx"] = self.keypoint_idx
        constraint_dict["target_position"] = self.target_position
        constraint_dict["tolerance"] = self.tolerance
        return constraint_dict

    def from_dict(self, data_map):
        self.keypoint_name = data_map["keypoint_name"]
        self.keypoint_idx = data_map["keypoint_idx"]
        self.target_position = data_map["target_position"]
        self.tolerance = data_map["tolerance"]


@attr.s
class AxisAlignmentConstraintSpec(OptimizationTermSpec):
    from_axis = [0, 0, 1]  # type: List[float]
    target_axis = [0, 0, 1]  # type: List[float]
    tolerance = 0.01  # type: float
    target_inner_product = 1

    @staticmethod
    def type_name():
        return "axis_alignment"

    @staticmethod
    def is_cost():  # type: () -> bool
        return False

    def to_dict(self):
        constraint_dict = dict()
        constraint_dict["from_axis"] = self.from_axis
        constraint_dict["target_axis"] = self.target_axis
        constraint_dict["tolerance"] = self.tolerance
        # constraint_dict['target_inner_product'] = self.target_inner_product
        return constraint_dict

    def from_dict(self, data_map):
        self.from_axis = data_map["from_axis"]
        self.target_axis = data_map["target_axis"]
        self.tolerance = data_map["tolerance"]
        # self.target_inner_product = data_map['target_inner_product']


@attr.s
class KeypointAxisAlignmentConstraintSpec(OptimizationTermSpec):
    axis_from_keypoint_idx = -1  # type: int
    axis_from_keypoint_name = ""  # type: str
    axis_to_keypoint_idx = -1  # type: int
    axis_to_keypoint_name = ""  # type: str
    target_axis = [0, 0, 1]  # type: List[float]
    tolerance = 0.01  # type: float
    target_inner_product = 1

    @staticmethod
    def type_name():
        return "keypoint_axis_alignment"

    @staticmethod
    def is_cost():  # type: () -> bool
        return False

    def to_dict(self):
        constraint_dict = dict()
        constraint_dict["axis_from_keypoint_name"] = self.axis_from_keypoint_name
        constraint_dict["axis_from_keypoint_idx"] = self.axis_from_keypoint_idx
        constraint_dict["axis_to_keypoint_name"] = self.axis_to_keypoint_name
        constraint_dict["axis_to_keypoint_idx"] = self.axis_to_keypoint_idx
        constraint_dict["target_axis"] = self.target_axis
        constraint_dict["tolerance"] = self.tolerance
        constraint_dict["target_inner_product"] = self.target_inner_product

        return constraint_dict

    def from_dict(self, constraint_dict):
        self.axis_from_keypoint_name = constraint_dict["axis_from_keypoint_name"]
        self.axis_from_keypoint_idx = constraint_dict["axis_from_keypoint_idx"]
        self.axis_to_keypoint_name = constraint_dict["axis_to_keypoint_name"]
        self.axis_to_keypoint_idx = constraint_dict["axis_to_keypoint_idx"]
        self.target_axis = constraint_dict["target_axis"]
        self.tolerance = constraint_dict["tolerance"]
        self.target_inner_product = constraint_dict["target_inner_product"]


@attr.s
class KeypointAxisOrthogonalConstraintSpec(OptimizationTermSpec):
    axis_from_keypoint_idx = -1  # type: int
    axis_from_keypoint_name = ""  # type: str
    axis_to_keypoint_idx = -1  # type: int
    axis_to_keypoint_name = ""  # type: str
    target_axis = [0, 0, 1]  # type: List[float]
    tolerance = 0.01  # type: float
    target_inner_product = 0

    @staticmethod
    def type_name():
        return "keypoint_axis_orthogonal"

    @staticmethod
    def is_cost():  # type: () -> bool
        return False

    def to_dict(self):
        constraint_dict = dict()
        constraint_dict["axis_from_keypoint_name"] = self.axis_from_keypoint_name
        constraint_dict["axis_from_keypoint_idx"] = self.axis_from_keypoint_idx
        constraint_dict["axis_to_keypoint_name"] = self.axis_to_keypoint_name
        constraint_dict["axis_to_keypoint_idx"] = self.axis_to_keypoint_idx
        constraint_dict["target_axis"] = self.target_axis
        constraint_dict["tolerance"] = self.tolerance
        constraint_dict["target_inner_product"] = self.target_inner_product
        return constraint_dict

    def from_dict(self, constraint_dict):
        self.axis_from_keypoint_name = constraint_dict["axis_from_keypoint_name"]
        self.axis_from_keypoint_idx = constraint_dict["axis_from_keypoint_idx"]
        self.axis_to_keypoint_name = constraint_dict["axis_to_keypoint_name"]
        self.axis_to_keypoint_idx = constraint_dict["axis_to_keypoint_idx"]
        self.target_axis = constraint_dict["target_axis"]
        self.tolerance = constraint_dict["tolerance"]
        self.target_inner_product = constraint_dict["target_inner_product"]


# For the specification of cost terms
@attr.s
class Point2PointCostL2Spec(OptimizationTermSpec):
    keypoint_name = ""  # type: str
    keypoint_idx = -1  # type: int
    target_position = [0, 0, 0]  # type: List[float]
    penalty_weight = 1.0  # type: float

    @staticmethod
    def type_name():
        return "point2point_cost"

    @staticmethod
    def is_cost():  # type: () -> bool
        return True

    def to_dict(self):
        constraint_dict = dict()
        constraint_dict["keypoint_name"] = self.keypoint_name
        constraint_dict["keypoint_idx"] = self.keypoint_idx
        constraint_dict["target_position"] = self.target_position
        constraint_dict["penalty_weight"] = self.penalty_weight
        return constraint_dict

    def from_dict(self, data_map):
        self.keypoint_name = data_map["keypoint_name"]
        self.keypoint_idx = data_map["keypoint_idx"]
        self.target_position = data_map["target_position"]
        self.penalty_weight = data_map["penalty_weight"]


# For the specification of point2plane cost
@attr.s
class Point2PlaneCostSpec(OptimizationTermSpec):
    keypoint_name = ""  # type: str
    keypoint_idx = -1  # type: int
    target_position = [0.0, 0.0, 0.0]  # type: List[float]
    plane_normal = [0.0, 0.0, 1.0]  # type: List[float]
    penalty_weight = 1.0  # type: float

    @staticmethod
    def type_name():  # type: () -> str
        return "point2plane_cost"

    @staticmethod
    def is_cost():  # type: () -> bool
        return True

    def to_dict(self):
        constraint_dict = dict()
        constraint_dict["keypoint_name"] = self.keypoint_name
        constraint_dict["keypoint_idx"] = self.keypoint_idx
        constraint_dict["target_position"] = self.target_position
        constraint_dict["plane_normal"] = self.plane_normal
        constraint_dict["penalty_weight"] = self.penalty_weight
        return constraint_dict

    def from_dict(self, data_map):
        self.keypoint_name = data_map["keypoint_name"]
        self.keypoint_idx = data_map["keypoint_idx"]
        self.target_position = data_map["target_position"]
        self.plane_normal = data_map["plane_normal"]
        self.penalty_weight = data_map["penalty_weight"]
