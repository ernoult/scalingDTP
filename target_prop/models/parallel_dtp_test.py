from typing import ClassVar, Type

from .parallel_dtp import ParallelDTP
from .dtp_test import TestDTP as DTPTests
from .dtp import DTP


class TestParallelModel(DTPTests):
    model_class: ClassVar[Type[DTP]] = ParallelDTP
