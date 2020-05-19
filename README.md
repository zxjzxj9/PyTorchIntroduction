# 《深入浅出PyTorch——从模型到源码》书籍源代码
**（如果对于代码有疑惑或者代码中有错误，请在GitHub仓库开新的Issue指出。）**

## 源代码目录

## 1. 第一章 深度学习概念简介

(无代码，略)

## 2. 第二章 PyTorch 深度学习框架简介

**代码2.1** [PyTorch软件包的导入和测试](./Chapter2/ex_2_1.py)

**代码2.2** [安装PyTorch的依赖关系](./Chapter2/ex_2_2.sh)

**代码2.3** [PyTorch编译命令](./Chapter2/ex_2_3.sh)

**代码2.4** [Python列表和Numpy数组转换为PyTorch张量](./Chapter2/ex_2_4.py)

**代码2.5** [指定形状生成张量](./Chapter2/ex_2_5.py)

**代码2.6-2.7** [指定形状生成张量](./Chapter2/ex_2_6.py)

**代码2.8** [PyTorch在不同设备上的张量](./Chapter2/ex_2_8.py)

**代码2.9** [PyTorch张量形状相关的一些函数](./Chapter2/ex_2_9.py)

**代码2.10** [PyTorch张量的切片和索引](./Chapter2/ex_2_10.py)

**代码2.11** [PyTorch张量的函数运算](./Chapter2/ex_2_11.py)

**代码2.12** [PyTorch张量的四则运算](./Chapter2/ex_2_12.py)

**代码2.13**  [PyTorch极值和排序的函数](./Chapter2/ex_2_13.py)

**代码2.14**  [PyTorch张量的矩阵乘法运算](./Chapter2/ex_2_14.py)

**代码2.15**  [torch.einsum函数的使用](./Chapter2/ex_2_15.py)

**代码2.16**  [张量的拼接和分割](./Chapter2/ex_2_16.py)

**代码2.17**  [张量维度扩增和压缩](./Chapter2/ex_2_17.py)

**代码2.18**  [张量的广播](./Chapter2/ex_2_18.py)

**代码2.19** [PyTorch模块类的构建方法](./Chapter2/ex_2_19.py)

**代码2.20** [PyTorch线性回归模型示例](./Chapter2/ex_2_20.py)

**代码2.21** [PyTorch线性回归模型调用方法实例](./Chapter2/ex_2_21.py)

**代码2.22** [PyTorch模块方法调用实例](./Chapter2/ex_2_22.py)

**代码2.23** [反向传播函数测试代码](./Chapter2/ex_2_23.py)

**代码2.24** [梯度函数的使用方法](./Chapter2/ex_2_24.py)

**代码2.25** [控制计算图产生的方法示例](./Chapter2/ex_2_25.py)

**代码2.26** [损失函数模块的使用方法](./Chapter2/ex_2_26.py)

**代码2.27** [简单线性回归函数和优化器](./Chapter2/ex_2_27.py)

**代码2.28** [PyTorch优化器对不同参数指定不同的学习率](./Chapter2/ex_2_28.py)

**代码2.29** [PyTorch学习率衰减类示例](./Chapter2/ex_2_29.py)

**代码2.30** [torch.utils.data.DataLoader类的签名](./Chapter2/ex_2_30.py)

**代码2.31** [torch.untils.data.Dataset类的构造方法](./Chapter2/ex_2_31.py)

**代码2.32** [简单torch.utils.data.Dataset类的实现](./Chapter2/ex_2_32.py)

**代码2.33** [torch.utils.data.IterableDataset类的构造方法](./Chapter2/ex_2_33.py)

**代码2.34** [PyTorch保存和载入模型](./Chapter2/ex_2_34.py)

**代码2.35** [PyTorch的状态字典的保存和载入](./Chapter2/ex_2_35.py)

**代码2.36** [PyTorch检查点的结构](./Chapter2/ex_2_36.py)

**代码2.37** [TensorBoard使用方法示例](./Chapter2/ex_2_37.py)

**代码2.38** [SummaryWriter提供的添加数据显示的方法](./Chapter2/ex_2_38.py)

**代码2.39** [torch.nn.DataParallel使用方式](./Chapter2/ex_2_39.py)

**代码2.40** [PyTorch分布式进程启动函数](./Chapter2/ex_2_40.py)

**代码2.41** [多进程训练模型的数据载入](./Chapter2/ex_2_41.py)

**代码2.42** [分布式数据并行模型的API](./Chapter2/ex_2_42.py)

**代码2.43** [分布式数据并行模型训练时的输出](./Chapter2/ex_2_43.py)

## 3. 第三章 PyTorch 计算机视觉模块

**代码3.1-3.3** [线性层的定义和使用方法](./Chapter3/ex_3_1.py)

**代码3.4** [ConvNd类的定义代码](./Chapter3/ex_3_4.py)

**代码3.5-3.9** [归一化层模块的定义](./Chapter3/ex_3_5.py)

**代码3.10-3.15** [池化模块的定义](./Chapter3/ex_3_10.py)

**代码3.16** [丢弃层模块的定义](./Chapter3/ex_3_16.py)

**代码3.17** [顺序模块的构造方法](./Chapter3/ex_3_17.py)

**代码3.18** [模块列表和模块字典的构造方法](./Chapter3/ex_3_18.py)

**代码3.19-3.22** [AlexNet实例及特征提取模块](./Chapter3/ex_3_19.py)

**代码3.23-3.24** [增益系数的计算和模块参数初始化](./Chapter3/ex_3_23.py)

**代码3.25** [InceptionNet的基础框架代码](./Chapter3/ex_3_25.py)

**代码3.26-3.27** [ResNet的基础框架代码](./Chapter3/ex_3_26.py)

## 4. 第四章 PyTorch 机器视觉案例

**代码4.1** [PyTorch常用的数据集包装类](./Chapter4/ex_4_1.py)

**代码4.2-4.9** [LeNet模型工程](./Chapter4/LeNet)

**代码4.10-4.12** [使用argparse库指定LeNet神经网络超参数](./Chapter4/ex_4_10.py)

**代码4.13** [imagenet.py训练代码数据载入部分](./Chapter4/ex_4_13.py)

**代码4.14** [ResNet瓶颈残差模块的代码实现](./Chapter4/ex_4_14.py)

**代码4.15-4.20** [InceptionNet子模块的实现](./Chapter4/ex_4_15.py)

**代码4.21-4.26** [SSD模型的代码实现](./Chapter4/SSD)

**代码4.27-4.30** [FCN模型的代码实现](./Chapter4/fcn.py)

**代码4.31** [U-Net模型的代码实现](./Chapter4/unet.py)

**代码4.32-4.37** [图像风格迁移代码](./Chapter4/style_transfer.py)

**代码4.38-4.39** [变分自编码器（VAE）代码实现](./Chapter4/vae.py)

**代码4.40-4.42** [生成对抗网络（GAN）代码实现](./Chapter4/gan.py)

## 5. 第五章 PyTorch 自然语言处理模块

**代码5.1** [使用sklearn的CountVectorizer来提取词频特征](./Chapter5/ex_5_1.py)

**代码5.2** [CountVectorizer类声明](./Chapter5/ex_5_2.py)

**代码5.3** [TF-IDF代码实例](./Chapter5/ex_5_3.py)

**代码5.4** [TfidfTransformer和TfidfVectorizer类的声明](./Chapter5/ex_5_4.py)

**代码5.5** [nn.Embedding类的定义](./Chapter5/ex_5_5.py)

**代码5.6** [词嵌入模块的使用示例](./Chapter5/ex_5_6.py)

**代码5.7** [从预训练的词嵌入矩阵得到词嵌入模块](./Chapter5/ex_5_7.py)

**代码5.8** [nn.EmbeddingBag类的定义](./Chapter5/ex_5_8.py)

**代码5.9-5.10** [pack_padded_sequence和pad_packed_sequence的使用](./Chapter5/ex_5_9.py)

**代码5.11** [简单RNN的参数代码](./Chapter5/ex_5_11.py)

**代码5.12** [RNN代码使用实例](./Chapter5/ex_5_12.py)

**代码5.13** [LSTM和GRU的参数定义](./Chapter5/ex_5_13.py)

**代码5.14** [LSTM和GRU模块的使用方法](./Chapter5/ex_5_14.py)

**代码5.15** [RNNCell、LSTMCell和GRUCell参数定义](./Chapter5/ex_5_15.py)

**代码5.16** [RNNCell、LSTMCell和GRUCell的使用方法](./Chapter5/ex_5_16.py)

**代码5.17** [MultiheadAttention模块参数定义](./Chapter5/ex_5_17.py)

**代码5.18** [Transformer单层编码器和解码器模块定义](./Chapter5/ex_5_18.py)

**代码5.19** [Transformer编码器、解码器和Transformer模型](./Chapter5/ex_5_19.py)

## 6. 第六章 PyTorch 自然语言处理案例

**代码6.1** [使用collections.Counter类构建单词表](./Chapter6/ex_6_1.py)

**代码6.2-6.3** [CBOW模型及其训练过程](./Chapter6/word2vec.py)

**代码6.4** [PyTorch余弦相似度的模块的参数定义和使用](./Chapter6/ex_6_4.py)

**代码6.5-6.6** [用于情感分析的深度学习模型代码](./Chapter6/sentiment.py)

**代码6.7-6.9** [基于循环神经网络的语言模型代码](./Chapter6/lm.py)

**代码6.10-6.12** [Seq2Seq模型代码](./Chapter6/seq2seq.py)

**代码6.13-6.16** [BERT模型代码](./Chapter6/bert.py)

## 第七章 其他重要模型

**代码7.1** [宽深模型代码](./Chapter7/wide_deep.py)

**代码7.2** [CTC损失函数的定义](./Chapter7/ex_7_2.py)

**代码7.3** [DeepSpeech模型代码](./Chapter7/deep_speech.py)

**代码7.4-7.7** [Tacotron模型代码](./Chapter7/tacotron.py)

**代码7.8-7.9** [WaveNet模型代码](./Chapter7/wavenet.py)

**代码7.10-7.14** [DQN模型代码](./Chapter7/dqn.py)

**代码7.15-7.17** [半精度模型的训练](./Chapter7/half_prec.py)

## 第八章 PyTorch 高级应用

**代码8.1-8.10** [PyTorch自定义激活函数和梯度](./Chapter8/GELU)

**代码8.11-8.14** [PyTorch钩子使用方法示例](./Chapter8/ex_8_11.py)

**代码8.15-8.17** [PyTorch静态图使用方法示例](./Chapter8/ex_8_15.py)

**代码8.18-8.22** [PyTorch静态模型的保存和载入](./Chapter8/ex_8_18)

## 第九章 PyTorch 源代码解析 

**代码9.1** [native_functions.yaml文件的声明](./ex_9_1.yaml)

**代码9.2-9.4** [pybind11的简单例子](./py_cpp_interface)