1.最近邻
Nearest Neighbors官方网址：
https://scikit-learn.org/stable/modules/neighbors.html

2.工作原理：
KNN是通过测量不同特征值之间的距离进行分类。它的思路是：K个最近的邻居，每个样本都可以用它最接近的K个邻居来代表，如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别，并具有这个类别上样本的特征， KNN算法的结果很大程度取决于K的选择，其中K通常是不大于20的整数。

KNN算法中，所选择的邻居都是已经正确分类的对象。该方法在定类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。

在KNN中，通过计算对象间距离来作为各个对象之间的非相似性指标，避免了对象之间的匹配问题，在这里距离一般使用欧氏距离或曼哈顿距离：

同时，KNN通过依据k个对象中占优的类别进行决策，而不是单一的对象类别决策。这两点就是KNN算法的优势。

接下来对KNN算法的思想总结一下：就是在训练集中数据和标签已知的情况下，输入测试数据，将测试数据的特征与训练集中对应的特征进行相互比较，找到训练集中与之最为相似的前K个数据，则该测试数据对应的类别就是K个数据中出现次数最多的那个分类，其算法的描述为：

计算测试数据与各个训练数据之间的距离；
按照距离的递增关系进行排序；
选取距离最小的K个点；
确定前K个点所在类别的出现频率；
返回前K个点中出现频率最高的类别作为测试数据的预测分类。
注：K取值较大时，抗噪能力；较小时，对噪声点的影响特别敏感

优点：易于理解，简单，无需训练；适用于稀有事件；适用于多分类问题
缺点：样本不平衡效果不佳；计算量较大，每一点求K个最近的邻居
改进方向：约简，删除对分类结果影响较小的属性；加权，和样本点距离小的邻居权值大
