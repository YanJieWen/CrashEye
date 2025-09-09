# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

# from .example_model import ResNet18
from .build import build_detector,build_tracker

__all__ = [
    'build_detector','build_tracker',
]

# def build_model(cfg):
#     model = ResNet18(cfg.MODEL.NUM_CLASSES)
#     return model
