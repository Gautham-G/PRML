import scipy.io
from AccMeasure import acc_measure
from mycluster import cluster
from show_topics import display_topics
import numpy as np

mat = scipy.io.loadmat('data.mat')
mat = mat['X']
X = mat[:, :-1]

# Results = []

# for i in range(20):
#     print('Run', i)
#     idx = cluster(X, 4)
#     acc = acc_measure(idx)
#     Results.append(acc)

# Results = np.array(Results)
# print('Max :', Results.max(), 'Min :', Results.min(), 'Mean :', Results.mean())

idx = cluster(X, 4)
acc = acc_measure(idx)
print('accuracy %.4f' % (acc))

