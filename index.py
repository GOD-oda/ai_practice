from collections import Counter
import scipy.sparse as sparse
from sklearn.decomposition import LatentDirichletAllocation
import json
import numpy as np
import sys

with open('./ml_mysql_nouns.json', 'r') as f:
    doc_data = json.load(f)

duplicating_all_vocab = []
for data in doc_data.values():
    duplicating_all_vocab += data

# 重複を弾く
all_vocab = list(set(duplicating_all_vocab))

# 元々の単語数は3091
# print(len(duplicating_all_vocab))
# 重複を弾くと676
# print(len(all_vocab))

# trainingデータとtestデータに分けておく
# 文書のインデックス番号を入れておく
all_doc_index = doc_data.keys()
all_doc_index_ar = np.array(list(all_doc_index))

# trainingデータには全体の100%を入れる(これを0.9にすると、全体の90%という指定ができる)
train_portion = 1
train_num = int(len(all_doc_index) * train_portion)

# trainingデータのindexと、testデータのインデックス。
# データが多い場合、shuffleなどを使って、トレーニングデータと、テストデータのインデックスを割り振ると良い
train_doc_index = all_doc_index_ar[:train_num]
test_doc_index = all_doc_index_ar[train_num:]

# 取り敢えず、空の疎行列を作り、あとでここにデータを入れていく。
# (train_doc_indexの長さ) * (all_vocabの長さ) = 8 * 676
lil_train = sparse.lil_matrix(len(train_doc_index), len(all_vocab), dtype=np.int)
lil_test = sparse.lil_matrix(len(test_doc_index), len(all_vocab), dtype=np.int)
# TODO: 出力結果が違うので要確認
# print(lil_train.shape)

# train_total_elements_numは単語が何個あるかをカウントしている
train_total_elements_num = 0
all_vocab_arr = np.array(all_vocab)
for i in range(len(train_doc_index)):
    doc_index = train_doc_index[i]

    row_data = Counter(doc_data[doc_index])

    # row_dataにキーとして入っている文字がwordに入る
    # for word in row_data.keys():
    #     word_index = np.where(all_vocab_arr == word)[0][0]
    #     lil_train[i, word_index] = row_data[word]
    #     train_total_elements_num += 1

# print(train_total_elements_num)
