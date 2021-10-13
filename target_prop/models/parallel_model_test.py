from typing import ClassVar, Type

from target_prop.models import BaseModel, ParallelModel
from .model_test import ModelTests


class TestParallelModel(ModelTests):
    model_class: ClassVar[Type[BaseModel]] = ParallelModel
