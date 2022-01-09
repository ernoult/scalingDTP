from .vanilla_dtp_test import TestVanillaDTP as VanillaDTPTests
from .tp import TargetProp, VanillaDTP
from .dtp import DTP
from typing import Type, ClassVar


class TestTargetProp(VanillaDTPTests):
    # The type of model to test. (In this case, Target Propagation)
    model_class: ClassVar[Type[DTP]] = TargetProp
