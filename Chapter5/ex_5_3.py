""" 为了能够现实下列代码的执行效果，请在安装PyTorch和Scikit-Learn之后，在Python交互命令行界面，
    即在系统命令行下输入python这个命令回车后，在>>>提示符后执行下列代码
    （#号及其后面内容为注释，可以忽略）
"""

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',]
X = vectorizer.fit_transform(corpus)

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
transformer = TfidfTransformer()
transformer
X1 = transformer.fit_transform(X)
X1.to_array()
vectorizer = TfidfVectorizer()
vectorizer
X2 = vectorizer.fit_transform(corpus)
X2
X2.toarray()