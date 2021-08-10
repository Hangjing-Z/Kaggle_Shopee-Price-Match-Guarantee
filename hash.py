import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import time

def hash_String(image, dim):
    hash_string = ""    #定义空字符串的变量，用于后续构造比较后的字符串
    image = cv2.resize(image,(dim,dim),interpolation=cv2.INTER_CUBIC )
    image = np.reshape(image, dim*dim)
    # 上一个函数grayscale_Image()缩放图片并返回灰度化图片，.getdata()方法可以获得每个像素的灰度值，使用内置函数list()将获得的灰度值序列化
    for row in range(1, dim*dim+1): #获取pixels元素个数，从1开始遍历
        if row % dim :  #因不同行之间的灰度值不进行比较，当与宽度的余数为0时，即表示当前位置为行首位，我们不进行比较
            if image[row-1] > image[row]: #当前位置非行首位时，我们拿前一位数值与当前位进行比较
                hash_string += '1'   #当为真时，构造字符串为1
            else:
                hash_string += '0'   #否则，构造字符串为0
          #最后可得出由0、1组64位数字字符串，可视为图像的指纹
    return int(hash_string,2)  #把64位数当作2进制的数值并转换成十进制数值


def Difference(dhash1, dhash2):
    difference = dhash1 ^ dhash2  #将两个数值进行异或运算
    return bin(difference).count('1') #异或运算后计算两数不同的个数，即个数<5，可视为同一或相似图片


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

    dis_test = np.zeros((len(test_filename), len(test_filename)))
    for num1 in range(len(test_filename)):
        for num2 in range(num1 + 1, len(test_filename)):
            train1 = cv2.imread('data/train_images/' + test_filename[num1])
            train2 = cv2.imread('data/train_images/' + test_filename[num2])
            train1 = cv2.cvtColor(train1,cv2.COLOR_RGB2GRAY)
            train2 = cv2.cvtColor(train2,cv2.COLOR_RGB2GRAY)
    #         dim = min(np.shape(train1)[0], np.shape(train1)[1], np.shape(train2)[0], np.shape(train2)[1])
            dim = 32
            hash1 = hash_String(train1, dim)
            hash2 = hash_String(train2, dim)
            dis_test[num1][num2] = Difference(hash1, hash2)
            dis_test[num2][num1] = dis_test[num1][num2]
    #         print(hamming_distance(hash1,hash2))


    dis_test_pre = np.array(dis_test)
    dimen = np.shape(dis_test)[0]
    dis_test = dis_test / np.max(dis_test)
    # dis_test[np.diag_indices_from(dis_test)] = 1

    # Iterate through different thresholds to maximize f1
    thresholds = list(np.arange(0.002, 0.1, 0.002))
    scores = []
    p1s = []
    r1s = []
    for threshold in thresholds:
        predictions = []
        for k in range(dimen):
            idx = np.where(dis_test[k,] < threshold)
            posting_ids = ' '.join(test['posting_id'].iloc[idx].values)
            predictions.append(posting_ids)

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