import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st

kernels = ['poly', 'rbf', 'linear', 'sigmoid']
f_svm=[0.47, 0.56, 0.47, 0.47]
a_svm=[.54,.62,.54,.54]
# nn_data = np.load('results.npz')
# nn_data2 = np.load('results2.npz')
# acc = np.concatenate((nn_data['acc'],nn_data2['acc']),axis=0)
# acc_test = np.concatenate((nn_data['acc_test'],nn_data2['acc_test']),axis=0)
nn_data3 = np.load('results4.npz')
f1scores = nn_data3['f1scores']
acc_test = nn_data3['acc_test']
# fig, ax = plt.subplots(1,7, sharey=True)
# fig.suptitle('Neurons per layer, 3 layers')
# for i in range(7):
#     ax[i].plot(acc[1,i])
#     ax[i].set_title(f'{(i+4)*2}')
#     ax[0].set_ylabel('Accuracy')

plt.figure()
plt.bar(kernels, f_svm)
plt.title('Accuracy per kernel')
plt.xlabel('Kernel')
plt.ylabel('Accuracy')

fig, ax = plt.subplots(2, 4, sharey=True, sharex=True)
fig.suptitle('Number of layers vs. number of neurons and accuracy/F1')
ax[0,0].set_ylabel('Accuracy')
ax[1,0].set_ylabel('F1-score')

# f=open('out.csv','a')
# f.write('Acc_avg,acc_std,f1_avg,f1_se\n')
for i in range(4):
    at_avg=np.average(acc_test, axis=1)
    at_se=st.t.interval(0.95, df=acc_test.shape[1]-1, loc=np.mean(acc_test[i,:]), scale=st.sem(acc_test[i,:]))
    f1_avg=np.average(f1scores, axis=1)
    f1_se=st.t.interval(0.95, df=acc_test.shape[1]-1, loc=np.mean(f1scores[i,:]), scale=st.sem(f1scores[i,:]))

    # f.write('{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(at_avg[i],at_avg[i]-at_se[0],f1_avg[i],f1_avg[i]-f1_se[0]))
    ax[0,i].plot(np.arange(8,22,2),acc_test[i,:])
    ax[0,i].set_title(f'{i+2}')
    ax[1,i].set_xlabel('Neurons per layer')
    ax[1,i].plot(np.arange(8,22,2), f1scores[i,:])

    ax[0,i].plot(15,at_avg[i], 'ko')
    ax[0,i].hlines(at_se[0],14,16,'k')
    ax[0,i].hlines(at_se[1],14,16,'k')
    ax[0,i].vlines(15,at_se[0],at_se[1],'k')

    ax[1,i].plot(15,f1_avg[i], 'ko')
    ax[1,i].hlines(f1_se[0],14,16,'k')
    ax[1,i].hlines(f1_se[1],14,16,'k')
    ax[1,i].vlines(15,f1_se[0],f1_se[1],'k')
    # red = ax[0,i].hlines(0.51, 8,22, 'r')
    # green = ax[1,i].hlines(0.49, 8,22, 'g')
# f.close()
# plt.figlegend([red, green], ['1 sigma accuracy upper bound', '1 sigma F1 upper bound'])

xv, yv = np.meshgrid(np.arange(8,22,2),np.arange(4)+2)
xvf = xv.flatten()
yvf = yv.flatten()
nparams = np.zeros(acc_test.shape[0]*acc_test.shape[1])
acc_flat = acc_test.flatten()
nparams = (xvf**2+1) * yvf

# plt.figure()
# plt.scatter(nparams, acc_flat)

# fig2 = plt.figure()
# ax = fig2.add_subplot(111, projection='3d')
# print(xv.shape)
# print(f1scores.shape)
# ax.plot_wireframe(xv, yv, f1scores)

plt.show()