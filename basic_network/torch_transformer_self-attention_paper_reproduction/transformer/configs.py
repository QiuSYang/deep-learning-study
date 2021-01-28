"""
# transformer config
"""
import os
import json


class TransformerConfig(object):
    def __init__(self, **kwargs):
        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)  # 设置类属性
            except AttributeError as error:
                raise error

    def para_to_json(self, save_path):
        """将参数保存到json文件中"""
        with open(save_path, mode='w', encoding='utf-8') as fw:
            json.dump(self.__dict__, fw, ensure_ascii=False, indent=2)
