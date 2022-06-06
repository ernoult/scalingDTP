from .lenet import LeNet
from .network_test import NetworkTests


class TestLeNet(NetworkTests[LeNet]):
    net_type: type[LeNet] = LeNet
