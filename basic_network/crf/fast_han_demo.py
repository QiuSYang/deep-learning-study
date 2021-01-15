"""
# 复旦 fastHan库使用
"""
from fastHan import FastHan


if __name__ == "__main__":
    model = FastHan()
    sentence = "郭靖是金庸笔下的一名男主。"
    result = model(sentence, 'CWS')
    print(result)
