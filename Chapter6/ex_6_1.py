""" 本示例所用的代码可以被用于NLP项目中用于统计词频，构建单词表
"""

from collections import Counter

class Vocab(object):

    UNK = '<unk>'

    def __init__(self, counter, max_size=None, min_freq=1,
                 specials=['<unk>', '<pad>'], specials_first=True):

        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        # 定义整数序号到单词映射
        self.itos = list()
        self.unk_index = None
        if specials_first:
            self.itos = list(specials)
            max_size = None if max_size is None else max_size + len(specials)

        # 如果输入有特殊字符，删掉这些特殊字符
        for tok in specials:
            del counter[tok]

        # 先按照字母顺序排序，再按照频率排序
        words_and_frequencies = sorted(counter.items(), \
                                       key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        # 排除小频率单词
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        if Vocab.UNK in specials: 
            unk_index = specials.index(Vocab.UNK)
            self.unk_index = unk_index if specials_first \
                else len(self.itos) + unk_index
            self.stoi = defaultdict(self._default_unk_index)
        else:
            self.stoi = defaultdict()

        if not specials_first:
            self.itos.extend(list(specials))

        # 定义单词到整数序号映射
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

def build_vocab_from_iterator(iterator):

    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    word_vocab = Vocab(counter)
    return word_vocab
