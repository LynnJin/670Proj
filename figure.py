import numpy as np
import data
import matplotlib.pyplot as plt


def sanityCheck(objType, phiType):
    """ draw and save the figure of the mean, range for different methods and N

    :param objType: objective type - worst/sum
    :param phiType: phi-divergence type - cre/chi/m-chi
    """
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
    plt.xlim(xmin=0, xmax=1000)
    plt.xlabel('N')

    if objType == 'sum':
        plt.yticks(list(range(100, 170, 10)))
        plt.ylim(ymin=100, ymax=161)
    elif objType == 'worst':
        plt.yticks(list(range(-13, 10, 2)))
        plt.ylim(ymin=-10, ymax=7)
    else:
        raise Exception("Wrong model type")
    plt.ylabel('Return')

    plt.savefig('/data/' + objType + '_' + phiType + '_final.png')
    plt.show()


def outSample(objType, phiType, N):
    """ draw and save the figure of the mean, range, reliability for robust model with different alpha

    :param objType: objective type - worst/sum
    :param phiType: phi-divergence type - cre/chi/m-chi
    """
    loaded = np.load(objType + '_' + phiType + '_alpha_' + str(N) + '.npz')
    robustReturns = loaded['robust']

    robArray = np.transpose(np.array(robustReturns))

    fig, ax1 = plt.subplots(figsize=(7, 5))
    alpha = [0.0001, 0.001, 0.01, 0.1]
    alphaTest = data.alphaSet(alpha)

    ax1.plot(alphaTest, robArray[2], color='b', label='Mean robust')
    ax1.fill_between(alphaTest, robArray[0], robArray[1], color='b', alpha=0.2, label='Range robust')

    ax1.set_xscale("log")

    if objType == 'sum':
        ax1.set_yticks(list(range(30, 181, 10)))
        ax1.set_ylim(ymin=30, ymax=181)
    elif objType == 'worst':
        ax1.set_yticks(list(range(-13, 8, 2)))
        ax1.set_ylim(ymin=-5, ymax=7)
    else:
        raise Exception("Wrong model type")

    ax1.set_xlim(xmin=0.0001, xmax=0.4)


    ax1.set_xlabel('alpha')
    ax1.set_ylabel('Return')
    ax1.plot(np.nan, 'r', label='Reliability')

    ax2 = ax1.twinx()
    ax2.plot(alphaTest, robArray[3], color='r', label='Reliability')
    ax2.set_ylim(ymin=0, ymax=1.1)
    ax2.set_yticks(list(np.arange(0, 1.1, 0.1)))
    ax2.set_ylabel('Reliability')

    ax1.legend(loc="lower right", prop={'size': 12})
    plt.title("N = " + str(N))

    plt.savefig('/data/' + objType + '_' + phiType + '_alpha_' + str(N) + '.png')
    plt.show()