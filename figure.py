import numpy as np
import data
import matplotlib.pyplot as plt

# draw the figure of the results for different N
def sanityCheck():
    phiType = "cre"
    objType = "worst"
    loaded = np.load(objType + '_' + phiType + '_final.npz')
    robustReturns = loaded['robust']
    detReturns = loaded['SAA']

    robArray = np.transpose(np.array(robustReturns))
    detArray = np.transpose(np.array(detReturns))

    plt.figure(figsize=(7, 5))

    plt.plot(list(range(10, 1001, 10)), robArray[2], color='b', label='Mean robust')
    plt.fill_between(list(range(10, 1001, 10)), robArray[0], robArray[1], color='b', alpha=0.2, label='Range robust')
    plt.plot(list(range(10, 1001, 10)), detArray[2], color='r', label='Mean nonrobust')
    plt.fill_between(list(range(10, 1001, 10)), detArray[0], detArray[1], color='r', alpha=0.2, label='Range nonrobust')

    plt.legend(loc="lower right", prop={'size': 12})
    plt.xticks(list(range(0, 1001, 100)))
    plt.yticks(list(range(-15, 9, 2)))
    plt.xlim(xmin=0, xmax=1000)
    plt.ylim(ymin=-15, ymax=9)
    plt.xlabel('N')
    plt.ylabel('Return')

    plt.savefig(objType + '_' + phiType + '_final.png')
    plt.show()

# draw the figure for the out of sample performance with fixed N and different alpha
def outSample():
    name = 'm-chi'
    N = 10
    loaded = np.load('sum_' + name + '_alpha_' + str(N) + '.npz')
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
    plt.yticks(list(range(30, 181, 10)))
    #plt.yticks(list(np.arange(0, 1, 0.1)))
    plt.xlim(xmin=0.0001, xmax=0.4)
    plt.ylim(ymin=30, ymax=181)
    #plt.ylim(ymin=0, ymax=1)
    plt.xlabel('alpha')
    plt.ylabel('Return')
    #plt.ylabel('Reliability')
    plt.title("N = " + str(N))

    plt.savefig('sum_' + name + '_alpha_' + str(N) + '.png')
    plt.show()

# draw the figure of reliability for different alpha with fixed N
def rel():
    name = 'm-chi'
    type = "sum"
    loaded = np.load(type + '_' + name + '_alpha_10' + '.npz', allow_pickle=True)
    r10 = loaded['robust']
    loaded = np.load(type + '_' + name + '_alpha_30' + '.npz', allow_pickle=True)
    r30 = loaded['robust']
    loaded = np.load(type + '_' + name + '_alpha_50' + '.npz', allow_pickle=True)
    r50 = loaded['robust']
    loaded = np.load(type + '_' + name + '_alpha_100' + '.npz', allow_pickle=True)
    r100 = loaded['robust']

    #detReturns = loaded["det"]

    a10 = np.transpose(np.array(r10))
    a30 = np.transpose(np.array(r30))
    a50 = np.transpose(np.array(r50))
    a100 = np.transpose(np.array(r100))
    #detArray = np.transpose(np.array(detReturns))

    alpha = [0.0001, 0.001, 0.01, 0.1]
    alphaTest = data.alphaSet(alpha)


    plt.plot(alphaTest, a10[4], color='b', label='N=10')
    plt.plot(alphaTest, a30[4], color='r', label='N=30')
    plt.plot(alphaTest, a50[4], color='g', label='N=50')
    plt.plot(alphaTest, a100[4], color='m', label='N=100')
    #plt.plot(alphaTest, detArray[4], color='r', label='Reliability SAA')

    plt.xscale("log")
    #plt.xticks(list(range(0, 1001, 100)))
    plt.yticks(list(np.arange(0, 1.1, 0.1)))
    plt.xlim(xmin=0.0001, xmax=0.4)
    plt.ylim(ymin=0, ymax=1.1)
    plt.xlabel('alpha')
    #plt.ylabel('Return')
    plt.ylabel('Reliability')
    plt.legend(loc="lower left", prop={'size': 12})
    plt.title("Reliability")

    plt.savefig('sum_' + name + '_rel_.png')
    plt.show()

#outSample()
sanityCheck()