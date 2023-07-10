from sklearn.datasets import make_blobs
from decisiontree import *


X, y = make_blobs(n_samples=100, n_features=2, centers=2,
                  cluster_std=3, random_state=0)
tree = ClassificationTree(criterion='gini', max_depth=4)
tree.fit(X, y)

plot_tree(tree)
