import numpy as np
import matplotlib.pyplot as plt

loaded = np.load('worst_hel_check2.npz')
robustReturns = loaded['robust']
detReturns = loaded['det']

robArray = np.transpose(np.array(robustReturns))
detArray = np.transpose(np.array(detReturns))

print(robArray.shape)
print(detArray.shape)


plt.figure(figsize=(7, 5))

plt.plot(list(range(10, 1001, 10)), robArray[2], color='b', label='Mean robust')
plt.fill_between(list(range(10, 1001, 10)), robArray[0], robArray[1], color='b', alpha=0.2, label='Range robust')
plt.plot(list(range(10, 1001, 10)), detArray[2], color='r', label='Mean nonrobust')
plt.fill_between(list(range(10, 1001, 10)), detArray[0], detArray[1], color='r', alpha=0.2, label='Range nonrobust')

plt.legend(loc="center right", prop={'size': 12})
plt.xticks(list(range(0, 1001, 100)))
plt.yticks(list(range(-10, 5, 2)))
plt.xlim(xmin=0, xmax=1000)
plt.ylim(ymin=-10, ymax=5)
plt.xlabel('N')
plt.ylabel('Return')

plt.savefig('worst_hel_check2.png')
plt.show()