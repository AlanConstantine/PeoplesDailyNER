# PeoplesDailyNER

##### #2018年7月6日更新BiLSTM+CRF模型#

使用BiLSTM对人民日报语料进行分词




`去年9月份初步尝试神经网络，今天拿出来写写，请路过的大神不吝赐教，感恩！`

### 环境
* ubuntu 16.04LTS
* python 3.5.2

### 技术栈
* keras
* numpy
* re

### 思路

##### 一、构造字典
将文本中所有出现的汉字加入到字典，每一个汉字对应唯一的数字值

##### 二、分词向量
对已经分好词的语料打上标签，实例如下：<br> 
一字词：<br> 
      `我    S     4`<br> 
二字词：<br> 
      `中    B     1`<br> 
      `国    E     3`<br> 
多字词（三个及以上）：<br> 
      `共    B     1`<br> 
      `产    M     2`<br> 
      `党    E     3`<br> 

##### 三、文本向量化
将文本根据标点符号进行切分成多条短文本
```splitpunc = ['。', '？', '！', '；']```
根据第一步构造的字典将短文本转换成向量，其中长度超过100的将超出部分切除，长度不足100的向量进行补0

##### 四、进入BiLSTM进行训练

### 神经网络结构
![](https://github.com/AlanConstantine/PeoplesDailyNER/raw/master/model.png) 

### 各文件说明
语料：
* 199801_people_s_daily.txt
    * 人民日报语料，里面的不仅仅做了分词处理，还有词性的标注，针对词性这不做研究

主要的程序：
* preprocess.py
    * 对文本预处理，将文本向量化，结果储存到PDdata.json
* PD_BiLSTM.py
    * 读取PDdata.json，进入神经网络训练
    * 生成PDmodel_epoch_150_batchsize_32_embeddingDim_100.h5　这个是模型，方便后续进行测试
* (可选)PD_BiLSTM－CRF.py
    * 加入了crf层
    * 读取PDdata.json，进入神经网络训练
    * 生成PDmodel_epoch_150_batchsize_32_embeddingDim_100.h5　这个是模型，方便后续进行测试
* LSTMpredict.py
    * 测试数据，代码中已经给出测试文本

其他（我学习keras的心路历程，入门的可以看看）：
* getError.py、test_to_categorical.py
    * 在加入嵌入层前的时候，我用来测试向量归一化处理写的，专门把keras内的utils.np_utils中to_categorical()函数取出来测试
* learnEmbedding.py
    * 对keras的嵌入层进行实际使用和熟悉
* testkeras.py
    * 用来熟悉keras的整个使用流程
* textpreprocess.1.py
    * 最开始写的文本向量化预处理程序

### 运行
        １．preprocess.py
        ２．PD_BiLSTM.py
        ３．LSTMpredict.py

### 结果
迭代到19次的时候已经是0.9726,去年9月份记得150次的结果是0.99多，召回率、准确率以及Ｆ值没有测试。
![](https://github.com/AlanConstantine/PeoplesDailyNER/raw/master/acc.png)

拿出去年生成的模型进行测试
![](https://github.com/AlanConstantine/PeoplesDailyNER/raw/master/result.png)
生成的向量对应分词向量标签
