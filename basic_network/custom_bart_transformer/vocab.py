from collections import defaultdict
import pickle
import torch
from torch import Tensor
from torch.autograd import Variable
from nltk import FreqDist
from convert import to_tensor, to_var

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
SPEAKER_ONE_TOKEN = "<speaker1>"
SPEAKER_TWO_TOKEN = "<speaker2>"
special_tokens = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SPEAKER_ONE_TOKEN, SPEAKER_TWO_TOKEN]

special_ids = [0, 1, 2, 3, 4, 5]
PAD_ID, UNK_ID, BOS_ID, EOS_ID, SPEAKER_ONE_ID, SPEAKER_TWO_ID = special_ids


class Vocab(object):
    def __init__(self, lang="en", max_size=None, min_freq=1):
        """Basic Vocabulary object"""
        self.lang = lang
        self.vocab_size = 0
        self.freqdist = FreqDist()

    def update(self, max_size=None, min_freq=1):
        """
        Initialize id2word & word2id based on self.freqdist
        max_size include 4 special tokens
        """

        # {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
        self.id2word = {
            PAD_ID: PAD_TOKEN, UNK_ID: UNK_TOKEN,
            BOS_ID: BOS_TOKEN, EOS_ID: EOS_TOKEN,
            SPEAKER_ONE_ID: SPEAKER_ONE_TOKEN, SPEAKER_TWO_ID: SPEAKER_TWO_TOKEN,
        }
        # {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.word2id = defaultdict(lambda: UNK_ID)  # Not in vocab => return UNK
        self.word2id.update({
            PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
            BOS_TOKEN: BOS_ID, EOS_TOKEN: EOS_ID,
            SPEAKER_ONE_TOKEN: SPEAKER_ONE_ID, SPEAKER_TWO_TOKEN: SPEAKER_TWO_ID,
        })
        # self.word2id = {
        #     PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        #     SOS_TOKEN: SOS_ID, EOS_TOKEN: EOS_ID
        # }

        vocab_size = len(special_ids)
        min_freq = max(min_freq, 1)

        # Reset frequencies of special tokens
        # [...('<eos>', 0), ('<pad>', 0), ('<sos>', 0), ('<unk>', 0)]
        freqdist = self.freqdist.copy()
        special_freqdist = {token: freqdist[token]
                            for token in special_tokens}
        freqdist.subtract(special_freqdist)

        # Sort: by frequency, then alphabetically
        # Ex) freqdist = { 'a': 4,   'b': 5,   'c': 3 }
        #  =>   sorted = [('b', 5), ('a', 4), ('c', 3)]
        sorted_frequency_counter = sorted(freqdist.items(), key=lambda k_v: k_v[0])
        sorted_frequency_counter.sort(key=lambda k_v: k_v[1], reverse=True)

        for word, freq in sorted_frequency_counter:

            if freq < min_freq or vocab_size == max_size:
                break
            self.id2word[vocab_size] = word
            self.word2id[word] = vocab_size
            vocab_size += 1

        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.id2word)

    def load(self, word2id_path=None, id2word_path=None):
        if word2id_path:
            with open(word2id_path, 'rb') as f:
                word2id = pickle.load(f)
            # Can't pickle lambda function
            self.word2id = defaultdict(lambda: UNK_ID)
            self.word2id.update(word2id)
            self.vocab_size = len(self.word2id)

        if id2word_path:
            with open(id2word_path, 'rb') as f:
                id2word = pickle.load(f)
            self.id2word = id2word

    def add_word(self, word):
        assert isinstance(word, str), 'Input should be str'
        self.freqdist.update([word])

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_dataframe(self, conversation_df):
        for conversation in conversation_df:
            for sentence in conversation:
                self.add_sentence(sentence)

    def pickle(self, word2id_path, id2word_path):
        with open(word2id_path, 'wb') as f:
            pickle.dump(dict(self.word2id), f)

        with open(id2word_path, 'wb') as f:
            pickle.dump(self.id2word, f)

    def to_list(self, list_like):
        """Convert list-like containers to list"""
        if isinstance(list_like, list):
            return list_like

        if isinstance(list_like, Variable):
            return list(to_tensor(list_like).numpy())
        elif isinstance(list_like, Tensor):
            return list(list_like.numpy())

    def id2sent(self, id_list):
        """list of id => list of tokens (Single sentence)"""
        id_list = self.to_list(id_list)
        sentence = []
        for id in id_list:
            word = self.id2word[id]
            if word not in special_tokens:
                # 在特殊token中的字符都不进行添加
                sentence.append(word)
            if word == EOS_TOKEN:
                break
        return sentence

    def sent2id(self, sentence, var=False):
        """list of tokens => list of id (Single sentence)"""
        id_list = [self.word2id[word] for word in sentence]
        if var:
            id_list = to_var(torch.LongTensor(id_list), eval=True)
        return id_list

    def decode(self, id_list):
        sentence = self.id2sent(id_list)
        if self.lang == "zh":
            return ''.join(sentence)
        return ''.join(sentence)
