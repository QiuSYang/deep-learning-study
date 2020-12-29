"""
# transformer config
"""
import os


class TransformerConfig(object):
    def __init__(self, **kwargs):
        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)  # 设置类属性
            except AttributeError as error:
                raise error
