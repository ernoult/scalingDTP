from typing import ClassVar, Type

from target_prop.models.dtp_test import not_well_supported

from .dtp import DTP
from .tp import TargetProp, VanillaDTP
from .vanilla_dtp_test import TestVanillaDTP as VanillaDTPTests


@not_well_supported
class TestTargetProp(VanillaDTPTests):
    # The type of model to test. (In this case, Target Propagation)
    model_class: ClassVar[Type[DTP]] = TargetProp
