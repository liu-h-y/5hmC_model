import numpy as np
import gensim
import pandas as pd

embedding_vec = gensim.models.KeyedVectors.load_word2vec_format("./dm3.w2v")
f_data = open("./i5hmCVec_on_iRNA5hmC_data/iRNA5hmC_data.txt")

label_line = f_data.readline()
seq_line = f_data.readline()
while seq_line:
    label = label_line.split("_")[0]
    if label == '>pos':
        feature_val = [1]
    else:
        feature_val = [0]
    seq = seq_line.strip()

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
    df_out.to_csv("./i5hmCVec_on_iRNA5hmC_data/feature.csv",mode='a',index=False,header=None)

    label_line = f_data.readline()
    seq_line = f_data.readline()