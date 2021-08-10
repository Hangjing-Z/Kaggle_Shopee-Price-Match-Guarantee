import math
import os
import pandas as pd
import numpy as np
import gensim
import sklearn
from gensim.models.doc2vec import Doc2Vec
from gensim.models import Doc2Vec
import time


def train(data):
    # 实例化一个模型
    model = gensim.models.Doc2Vec(vector_size=50, window=20, min_count=5,
                                  workers=4, alpha=0.2, min_alpha=0.2, epochs=10)
    model.build_vocab(data)
    print("开始训练...")
    # 训练模型
    start = time.clock()
    model.train(data, total_examples=model.corpus_count, epochs=model.epochs)
    end = time.clock()

    model.save("models/doc2vec.model")
    print("model saved", end - start)
    return model


def sent2vec(model, words):
    vect_list = []
    for w in words:
        try:
            vect_list.append(model.wv[w])
        except:
            continue
    vect_list = np.array(vect_list)
    vect = vect_list.sum(axis=0)
    return vect / np.sqrt((vect ** 2).sum())


def similarity(a_vect, b_vect):
    dot_val = 0.0
    a_norm = 0.0
    b_norm = 0.0
    cos = None
    for a, b in zip(a_vect, b_vect):
        dot_val += a * b
        a_norm += a ** 2
        b_norm += b ** 2
    if a_norm == 0.0 or b_norm == 0.0:
        cos = -1
    else:
        cos = dot_val / ((a_norm * b_norm) ** 0.5)
    return cos


def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    precision = intersection / len_y_pred
    recall = intersection / len_y_true
    return f1, precision, recall


if __name__ == '__main__':
    df_full = pd.read_csv('data/train.csv')

    df_full['fold'] = np.random.randint(low=0, high=10, size=df_full.shape[0])
    train = df_full[df_full['fold']>0].reset_index(drop=True)
    test = df_full[df_full['fold']==0].reset_index(drop=True)
    train_filename = list(train.image)
    test_filename = list(test.image)

    tmp = test.groupby(['label_group'])['posting_id'].unique().to_dict()
    test['matches'] = test['label_group'].map(tmp)
    test['matches'] = test['matches'].apply(lambda x: ' '.join(x))
    tmp = train.groupby(['label_group'])['posting_id'].unique().to_dict()
    train['matches'] = train['label_group'].map(tmp)
    train['matches'] = train['matches'].apply(lambda x: ' '.join(x))

    data = []
    for i, line in enumerate(test.title):
        tem = gensim.utils.simple_preprocess(test.title[i])
        data.append(gensim.models.doc2vec.TaggedDocument(tem, [i]))
    model = train(data)
    vect = np.zeros((len(test), 50))
    for i in range(len(test)):
        vect[i,] = sent2vec(model, gensim.utils.simple_preprocess(test.title[i]))

    dis_test = np.zeros((len(test), len(test)))
    for i in range(len(test)):
        for j in range(i + 1, len(test)):
            dis_test[i][j] = similarity(vect[i], vect[j])
            dis_test[j][i] = dis_test[i][j]
    dis_test[np.diag_indices_from(dis_test)] = 1

    thresholds = list(np.arange(0.6, 1.0, 0.01))
    scores = []
    p1s = []
    r1s = []
    for threshold in thresholds:
        predictions = []
        for k in range(len(test)):
            idx = np.where(dis_test[k,] > threshold)
            #         ids = indices[k,idx]
            #                 print("ids")
            #                 print(ids )
            posting_ids = ' '.join(test['posting_id'].iloc[idx].values)
            predictions.append(posting_ids)

        #             print("df")
        #             print(df['matches'])

        test['pred_matches'] = predictions
        test['f1'], test['precision'], test['recall'] = f1_score(test['matches'], test['pred_matches'])
        score = test['f1'].mean()
        p1 = test['precision'].mean()
        r1 = test['recall'].mean()
        print(f'Our f1 score for threshold {threshold} is {score}')
        scores.append(score)
        p1s.append(p1)
        r1s.append(r1)

    thresholds_scores = pd.DataFrame({'thresholds': thresholds, 'scores': scores, 'p1s': p1s, 'r1s': r1s})
    max_score = thresholds_scores[thresholds_scores['scores'] == thresholds_scores['scores'].max()]
    best_threshold = max_score['thresholds'].values[0]
    best_score = max_score['scores'].values[0]
    best_p = max_score['p1s'].values[0]
    best_r = max_score['r1s'].values[0]
    print(f'Our best score is {best_score} and has a threshold {best_threshold}')
