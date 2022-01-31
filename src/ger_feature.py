import numpy as np
import pandas as pd
import gensim

df = pd.read_csv("./data/data.csv")

embedding_vec = gensim.models.KeyedVectors.load_word2vec_format("./data/dm3.w2v")

num_samples = df.shape[0]
for i in range(num_samples):
    feature_val = []
    label = df.iloc[i,0]
    seq = df.iloc[i,-2]
    if label == 'P':
        feature_val.append(1)
    else:
        feature_val.append(0)

    for len_word in range(3,9):
        vec = np.zeros(100)
        for index in range(len(seq)-len_word+1):
            word = seq[index:index+len_word]
            word = word.replace('U','T')
            vec += embedding_vec[word]
        vec /= len(seq)-len_word+1
        list_vec = vec.tolist()
        feature_val.extend(list_vec)
    df_out = pd.DataFrame(feature_val).T
    df_out.to_csv("./data/feature.csv",mode='a',index=False,header=None)
