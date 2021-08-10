import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import time
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def get_neighbors(df, embeddings, KNN=50, image=True):
    model = NearestNeighbors(n_neighbors=KNN)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    print(indices.shape)

    # Iterate through different thresholds to maximize cv, run this in interactive mode, then replace else clause with a solid threshold
    thresholds = list(np.arange(2, 4, 0.1))
    scores = []
    p1s = []
    r1s = []
    for threshold in thresholds:
        predictions = []
        for k in range(embeddings.shape[0]):
            idx = np.where(distances[k,] < threshold)[0]
            ids = indices[k, idx]
            posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
            predictions.append(posting_ids)


        df['pred_matches'] = predictions
        df['f1'], df['precision'], df['recall'] = f1_score(df['matches'], df['pred_matches'])
        score = df['f1'].mean()
        p1 = df['precision'].mean()
        r1 = df['recall'].mean()
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
    del model, distances, indices
    gc.collect()
    return df, predictions, best_threshold, best_score, best_p, best_r


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


def des2feature(des, num_words, centures):
    img_feature_vec = np.zeros((1,num_words),'float32')
    for i in range(des.shape[0]):
        feature_k_rows=np.ones((num_words,128),'float32')
        feature=des[i]
        feature_k_rows=feature_k_rows*feature
        feature_k_rows=np.sum((feature_k_rows-centures)**2,1)
        index=np.argmax(feature_k_rows)
        img_feature_vec[0][index]+=1
    return img_feature_vec


def get_all_features(des_list, num_words, centres):
    # 获取所有图片的特征向量
    allvec = np.zeros((len(des_list), num_words), 'float32')
    for i in range(len(des_list)):
        if des_list[i] != []:
            allvec[i] = des2feature(des = des_list[i], num_words = num_words, centures = centres)
    return allvec

if __name__ == '__main__':
    df_full = pd.read_csv('data/train.csv')

    files = os.listdir('data/train_images')
    # train_filename, test_filename = train_test_split(files, test_size=0.1, random_state=12)
    df_full['fold'] = np.random.randint(low=0, high=10, size=df_full.shape[0])
    train = df_full[df_full["fold"] > 0].reset_index(drop=True)
    test = df_full[df_full["fold"] == 0].reset_index(drop=True)
    train_filename = list(train.image)
    test_filename = list(test.image)

    tmp = test.groupby(['label_group'])['posting_id'].unique().to_dict()
    test['matches'] = test['label_group'].map(tmp)
    test['matches'] = test['matches'].apply(lambda x: ' '.join(x))
    tmp = train.groupby(['label_group'])['posting_id'].unique().to_dict()
    train['matches'] = train['label_group'].map(tmp)
    train['matches'] = train['matches'].apply(lambda x: ' '.join(x))

    sift = cv2.xfeatures2d.SIFT_create(500)
    des_train = {}
    des_trainmatrix = np.zeros((1, 128))
    for file in train_filename:
        train = cv2.imread('data/train_images/' + file)
        kp, des = sift.detectAndCompute(train, None)
        des_train[file] = des
    for i in range(len(des_train)):
        des_trainmatrix = np.row_stack((des_trainmatrix, list(des_train.values())[i]))

    des_test = {}
    des_testmatrix = np.zeros((1, 128))
    # des_matrix = np.zeros((1,128))
    for file in test_filename:
        ima = cv2.imread('data/train_images/' + file)
        kp, des = sift.detectAndCompute(ima, None)

        des_test[file] = des
    for i in range(len(des_test)):
        des_testmatrix = np.row_stack((des_testmatrix, list(des_test.values())[i]))

    num_words = len(test['label_group'].unique())
    kmeans = KMeans(n_clusters=num_words, random_state=33)
    kmeans.fit(des_testmatrix)
    centres = kmeans.cluster_centers_  # 视觉聚类中心
    img_features = get_all_features(list(des_test.values()), num_words, centres)
    test, text_predictions, best_threshold, best_score, best_p, best_r = get_neighbors(test, img_features, KNN=20, image=Ture)