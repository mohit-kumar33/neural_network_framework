"""
neural_network_framework — a minimal neural network framework built from scratch.
"""

from neural_network_framework.tensor import Tensor
from neural_network_framework import nn
from neural_network_framework import optim
from neural_network_framework import utils
from neural_network_framework import viz
from neural_network_framework.viz import draw_graph

__version__ = "0.1.0"
__all__ = ["Tensor", "nn", "optim", "utils", "viz", "draw_graph"]
