import numpy as np
from matplotlib import pyplot as plt

kernels = ['poly', 'rbf', 'linear', 'sigmoid']
f_svm=[0.47, 0.56, 0.47, 0.47]
nn_data = np.load('results.npz')
nn_data2 = np.load('results2.npz')
acc = np.concatenate((nn_data['acc'],nn_data2['acc']),axis=0)
acc_test = np.concatenate((nn_data['acc_test'],nn_data2['acc_test']),axis=0)
# fig, ax = plt.subplots(1,7, sharey=True)
# fig.suptitle('Neurons per layer, 3 layers')
# for i in range(7):
#     ax[i].plot(acc[1,i])
#     ax[i].set_title(f'{(i+4)*2}')
#     ax[0].set_ylabel('Accuracy')

# plt.figure()
# plt.bar(kernels, f_svm)
# plt.title('Weighted average F1-score per kernel')
# plt.xlabel('Kernel')
# plt.ylabel('Weighted average F1-score')
fig, ax = plt.subplots(1, 4, sharey=True)
fig.suptitle('Number of layers vs. number of neurons and accuracy')
ax[0].set_ylabel('Accuracy')
for i in range(4):
    ax[i].plot(np.arange(8,22,2),acc_test[i,:])
    ax[i].set_title(f'{i+2}')
    ax[i].set_xlabel('Neurons per layer')

plt.show()