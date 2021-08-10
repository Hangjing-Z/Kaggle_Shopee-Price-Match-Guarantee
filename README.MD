## 代码结构与功能说明
```
├── input 输入文件夹
│   ├── pytorch-image-models-master timm仓库(需下载)
│   ├── shopee-folds 存放对数据集进行分割后的csv文件
│   └── shopee-product-matching 数据集 (需下载)
├── models 存放模型
│   └── best_model.pt 最佳模型参数(文件过大放在清华云盘,需下载)
├── dataset.py 数据集处理模块
├── effmodel.py 深度网络模型
├── eval.py 测试脚本
├── preprocess.py 预处理脚本
├── README.MD 描述文件
├── requirements.txt 需要的Python库
├── test_config.py 测试配置文件
├── train_config.py 训练配置文件
├── train_effb3.py 训练脚本
├── sift.py sift模型脚本
├── hash.py hash模型脚本
├── doc2vec.py doc2vec模型脚本
└── utils.py 工具类函数
```


## 环境配置

### 软件环境
- OS Ubuntu 18.04.1
- Python 3.8.8
- Pytorch torch==1.8.1
- Cuda release 11.2

其他Python环境包版本可于```requirements.txt```中查看.

### 硬件环境

- GeForce RTX 2080Ti 12GB x4
- Intel(R) Xeon(R) W-3275M CPU @ 2.50GHz x56
- RAM 376GB

## 复现流程

以下命令全部在本文件夹根目录执行

### 1.环境包安装：
```sh
pip install -r requirements.txt
```

### 2.下载数据
- 下载作业说明中提供的数据集
```sh
wget -c "https://cloud.tsinghua.edu.cn/f/5c7ba8c55e04478d86d9/?dl=1" -O shopee-product-matching.zip
```
- 下载最佳模型(超过50MB按要求放在了清华云盘上)

```sh
wget -c "https://cloud.tsinghua.edu.cn/f/ef6f327703554c9d968c/?dl=1" -O ./models/best_model.pt
```

### 3.整理数据

- 将数据集整理到input文件夹下
```
unzip shopee-product-matching.zip -d input/shopee-product-matching
```

- 下载最新版的timm库

```
cd input/ && git clone  https://github.com/rwightman/pytorch-image-models.git && cd ..
```

### 4.训练EfficientNet网络

参数设置可在```train_config.py```文件中调整, 默认即为最佳参数, 默认使用cuda:0对应的的gpu进行训练. 训练时每个epoch需20分钟左右时间, 一共18个epoch, 需要5-6个小时的训练时间.

```sh
python train_effb3.py
```

训练模型将在每个epoch结束时存储到```./models```目录下.

### 5.测试模型
参数设置可在```test_config.py```文件中调整, 默认即为最佳参数, 默认使用cuda:0对应的的gpu进行计算

- 测试上一步训练得到的模型:

```sh
python eval.py ./models/effb3_lr1e5sq2_decay8e-1_warmup5_batch8_epoch18_17.pt
```

- 测试提交的最优模型:

```sh
python eval.py ./models/best_model.pt 
```

- 预期结果应为:

```sh
Threshold-image: 4.500000  Threshold-text: 0.750000 F1: 0.946502 Precision: 0.980754 Recall: 0.938808
```

