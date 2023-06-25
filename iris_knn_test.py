# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import metrics
import matplotlib.pyplot as plt


iris = datasets.load_iris()
iris_features = iris.data[:, :2]
iris_target = iris.target
train_x, test_x, train_y, test_y = train_test_split(iris_features, iris_target, test_size=0.3, random_state=123)
clf = neighbors.KNeighborsClassifier(15)
clf.fit(train_x, train_y)
pred = clf.predict(test_x)
print('准确率:{:.1%}'.format(metrics.accuracy_score(test_y, pred)))

# 分类散点图
t = np.column_stack((test_x, pred))
plt.scatter(t[t[:, 2] == 0][:, 0], t[t[:, 2] == 0][:, 1], color='hotpink')
plt.scatter(t[t[:, 2] == 1][:, 0], t[t[:, 2] == 1][:, 1], color='#88c999')
plt.scatter(t[t[:, 2] == 2][:, 0], t[t[:, 2] == 2][:, 1], color='#6495ED')
plt.show()


# colormap = dict(zip(np.unique(iris_target), sns.color_palette()[:3]))
# plt.scatter(train_data[:, 0], train_data[:, 1], edgecolors=[colormap[x] for x in train_data[:, 2]],
# c='', s=80, label='all_data')
# plt.scatter(test_data[:, 0], test_data[:, 1], marker='^', color=[colormap[x] for x in Z], s=20, label='test_data')
# plt.legend()
# plt.show()
