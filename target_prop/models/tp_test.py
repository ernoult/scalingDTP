from typing import ClassVar, Type

from .dtp import DTP
from .tp import TargetProp, VanillaDTP
from .vanilla_dtp_test import TestVanillaDTP as VanillaDTPTests


class TestTargetProp(VanillaDTPTests):
    # The type of model to test. (In this case, Target Propagation)
    model_class: ClassVar[Type[DTP]] = TargetProp
