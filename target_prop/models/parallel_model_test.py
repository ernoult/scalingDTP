from typing import ClassVar, Type

from target_prop.models import BaseModel, ParallelDTP
from .model_test import ModelTests


class TestParallelModel(ModelTests):
    model_class: ClassVar[Type[BaseModel]] = ParallelDTP
