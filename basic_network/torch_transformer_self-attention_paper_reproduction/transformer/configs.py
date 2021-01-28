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

    def to_json_string(self):
        return json.dumps(self.__dict__, indent=2)

    def save_para_to_json_file(self, json_file):
        """参数保存至json文件"""
        with open(json_file, mode='w', encoding='utf-8') as fw:
            json.dump(self.__dict__, fw, ensure_ascii=False, indent=2)
