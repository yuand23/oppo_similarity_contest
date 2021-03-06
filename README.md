# oppo_similarity_contest
2021全球人工智能技术创新大赛（赛道三）: oppo小布助手对话短文本语义匹配
[比赛页面](https://tianchi.aliyun.com/competition/entrance/531851/introduction?spm=5176.12281976.0.0.706d22c6WTCVDa)

## 比赛内容简介
给定两个短句，构建模型判断这两个句子的语义是否相似，本质上是个二分类问题。
难点在于官方给定的所有数据都是加密的，每个汉字表示为一个整数数字，对应关系未知，数字编号和语义无关。
给定训练集有10w个句子对，并不足以作为语料重新训练一个预训练模型。
官方baseline采用tfidf的方法结果为~76%

## 做的一些工作
* 用词频对应的方式将官方给定的数据集中的token id和bert模型的vocab对应。将出现次数过少的token用'[unknown]'替换（实际上给定训练集中有一半的token只出现了5次以下）。然后对bert模型接一层线性分类器，输入训练数据微调模型。（词频的对应方法借鉴[bojone的baseline](https://github.com/bojone/oppo-text-match)，微调训练方式不太相同，直接是对分类数据的微调）用了10个epoch左右，得到val_set正确率~83%
* 比赛主办方在比赛初期给过一个10w的未加密训练数据。因此尝试了一下先利用不加密数据对bert分类器进行微调。将微调好的模型作为训练加密数据的base模型。效果有提高，val_set正确率~85%
* 考虑到训练集中有一半的token被替换成通用字符'[unknown]'，理论上会降低模型的准确率（如果一个句子对中的两个句子都包含同一个token但是这个token在其他句子中没有出现过，将这个token用'[unknown]'替换后就和其他的'[unknown]'没有区别了，但是一个句子对中‘每个句子都出现同一个不常用的词（字）’这一特征可能对语义相似的判断很重要），为了加入一个句子对中‘有关键token共现’这一特征，对模型的外接分类dense层做了改造，添加了一个词共现得分特征（得分计算方式为：sum(1/e^freq), 其中freq为共现token在整个数据集中出现的频率（目的是去掉类似停用词的词），sum为对两个句子中所有的共现词求和）。然后对这个特征进行了min-max归一化。尝试了调低最后一层分类层的输入维度128 + 1 -> 64 + 1（1为外加的词共现特征），从结果来看准确率似乎并没有提高。这个有时间再研究一下。。。。。。待补充
