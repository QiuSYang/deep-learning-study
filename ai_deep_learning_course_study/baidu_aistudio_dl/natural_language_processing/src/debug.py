"""
# debug program
"""
import paddle
from natural_language_processing.src.word_embedding import SkipGram

if __name__ == '__main__':
    model = SkipGram(vocab_size=100, embedding_size=20)
    center_words_var = paddle.to_tensor([10])
    target_words_var = paddle.to_tensor([4])
    pred = model(center_words_var, target_words_var)
    print(pred.numpy()[0])
    pass
