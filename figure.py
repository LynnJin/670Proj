import numpy as np
import data
import matplotlib.pyplot as plt

def outSample():
    loaded = np.load('sum_cre_alpha_100.npz')
    robustReturns = loaded['robust']
    #detReturns = loaded['det']

    robArray = np.transpose(np.array(robustReturns))
    #detArray = np.transpose(np.array(detReturns))

    plt.figure(figsize=(7, 5))
    alpha = [0.0001, 0.001, 0.01, 0.1]
    alphaTest = data.alphaSet(alpha)

    #plt.plot(alphaTest, robArray, color='b')
    plt.plot(alphaTest, robArray[3], color='r', label='Robust solution')
    plt.plot(alphaTest, robArray[2], color='b', label='Mean robust')
    plt.fill_between(alphaTest, robArray[0], robArray[1], color='b', alpha=0.2, label='Range robust')
    #plt.plot(alphaTest, detArray[2], color='r', label='Mean nonrobust')
    #plt.fill_between(alphaTest, detArray[0], detArray[1], color='r', alpha=0.2, label='Range nonrobust')

    plt.legend(loc="lower right", prop={'size': 12})
    plt.xscale("log")
    #plt.xticks(list(range(0, 1001, 100)))
    plt.yticks(list(range(60, 171, 10)))
    #plt.yticks(list(np.arange(0, 1, 0.1)))
    plt.xlim(xmin=0.0001, xmax=0.4)
    plt.ylim(ymin=60, ymax=170)
    #plt.ylim(ymin=0, ymax=1)
    plt.xlabel('alpha')
    plt.ylabel('Return')
    #plt.ylabel('Reliability')
    plt.title("N = 100")

    plt.savefig('sum_cre_alpha_100.png')
    plt.show()


def sanityCheck():
    loaded = np.load('sum_cre_alpha.npz')
    robustReturns = loaded['robust']
    detReturns = loaded['det']

    robArray = np.transpose(np.array(robustReturns))
    detArray = np.transpose(np.array(detReturns))

    plt.figure(figsize=(7, 5))

    plt.plot(list(range(10, 1001, 10)), robArray[2], color='b', label='Mean robust')
    plt.fill_between(list(range(10, 1001, 10)), robArray[0], robArray[1], color='b', alpha=0.2, label='Range robust')
    plt.plot(list(range(10, 1001, 10)), detArray[2], color='r', label='Mean nonrobust')
    plt.fill_between(list(range(10, 1001, 10)), detArray[0], detArray[1], color='r', alpha=0.2, label='Range nonrobust')

    plt.legend(loc="lower right", prop={'size': 12})
    plt.xticks(list(range(0, 1001, 100)))
    plt.yticks(list(range(80, 181, 20)))
    plt.xlim(xmin=0, xmax=1000)
    plt.ylim(ymin=80, ymax=180)
    plt.xlabel('N')
    plt.ylabel('Return')

    plt.savefig('sum_cre_check3.png')
    plt.show()

outSample()