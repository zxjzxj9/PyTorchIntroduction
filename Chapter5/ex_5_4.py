""" 该代码仅为演示函数签名和所用方法，并不能实际运行
"""

class sklearn.feature_extraction.text.TfidfTransformer(norm='l2',
    use_idf=True, smooth_idf=True, sublinear_tf=False)

class sklearn.feature_extraction.text.TfidfVectorizer(input='content',
    encoding='utf-8', decode_error='strict', strip_accents=None,
    lowercase=True, preprocessor=None, tokenizer=None, 
    analyzer='word',
    stop_words=None, token_pattern='(?u)\b\w\w+\b', 
    ngram_range=(1, 1),
    max_df=1.0, min_df=1, max_features=None, vocabulary=None,
    binary=False, dtype=<class 'numpy.float64'>, norm='l2', use_idf=True,
    smooth_idf=True, sublinear_tf=False)
