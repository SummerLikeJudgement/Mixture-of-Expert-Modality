"""
ATIO -- All Trains in One
"""
from .singleTask import *

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'emoe': EMOE,
        }
    
    def getTrain(self, args):
        return self.TRAIN_MAP[args['model_name']](args) # 从args中获取'model_name'的值作为键，使用这个键从TRAIN_MAP中查找对应的类或函数
