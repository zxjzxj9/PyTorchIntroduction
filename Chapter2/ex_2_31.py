""" 该代码仅为演示类的构造方法所用，并不能实际运行
   （#号及其后面内容为注释，可以忽略）
"""

class Dataset(object):
    def __getitem__(self, index):
        # index: 数据缩索引（整数，范围为0到数据数目-1）
        # ...
        # 返回数据张量

    def __len__(self):
        # 返回数据的数目
        # ...
