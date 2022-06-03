from .network_test import NetworkTests
from .simple_vgg import SimpleVGG


class TestSimpleVGG(NetworkTests[SimpleVGG]):
    net_type: type[SimpleVGG] = SimpleVGG
