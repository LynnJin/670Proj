import data
import model
import evaluate
import figure
import numpy as np


def sanityCheck(phiType, objType):
    """ reproduce the results in the paper, solve optimization model on empirical distribution,
    test on samples true distribution centered at the empirical distribution

    :param phiType: phi-divergence type - cre/chi/m-chi
    :param objType: objective type - worst/sum
    """
    # set parameter
    c, v, s, l, Q = data.read("data/data.csv")
    budget = 1000
    demand = [4, 8, 10]
    N_max = 1010

    # number of iterations
    it = 10000
    alpha = 0.05
    numDemand = Q.shape[0]
    numItem = Q.shape[1]

    # return for different N
    detReturns = []
    robustReturns = []
    for N in range(10, N_max, 10):
        # calculate rho
        rhoc = []
        for j in range(numItem):
            rhoc.append(data.rhoc(alpha, numDemand, phiType, N, Q[:, j]))

        # evaluate the returns
        modelType = "robust"
        minReturn, maxReturn, meanReturn = \
            evaluate.stat(c, v, s, l, Q, budget, demand, modelType, objType, phiType, it, rhoc)
        robustReturns.append([minReturn, maxReturn, meanReturn])
        modelType = "det"
        minReturn, maxReturn, meanReturn = evaluate.stat(c, v, s, l, Q, budget, demand, modelType, objType, phiType, it,
                                                         rhoc)
        detReturns.append([minReturn, maxReturn, meanReturn])

        # print the process
        if N % 50 == 0 and N >= 50:
            print(str(N / 10) + " of " + str(N_max / 10 - 1) + " experiments done")

    # save the data
    np.savez_compressed('data/' + objType + '_' + phiType + '_' + 'check3.npz', robust=robustReturns, det=detReturns)

    figure.sanityCheck(objType, phiType)


def outSample(phiType, objType, N):
    """ sample empirical distribution, solve optimization model,
    test the out of sample performance on true distribution for different alpha with small N

    :param phiType: phi-divergence type - cre/chi/m-chi
    :param objType: objective type - worst/sum
    :param N: number of training sample
    """
    # set parameter
    c, v, s, l, trueProb = data.read("data/data.csv")
    budget = 1000
    demand = [4, 8, 10]

    # get the alpha to be tested
    alpha=[0.0001, 0.001, 0.01, 0.1]
    alphaTest = data.alphaSet(alpha)

    # number of iterations
    it = 100
    numDemand = trueProb.shape[0]
    numItem = trueProb.shape[1]

    # return for different alpha
    robustReturns = []
    n = 0

    # generate rho for the robust optimization problem
    for alpha in alphaTest:
        # evaluate the returns
        modelType = "robust"
        minReturn, maxReturn, meanReturn, time = evaluate.statOut(
            c, v, s, l, budget, demand, modelType,objType, phiType, alpha, it, N, trueProb)
        robustReturns.append([minReturn, maxReturn, meanReturn, time])

        # print the process
        n = n + 1
        if n % 10 == 0 and n >= 10:
            print(str(n) + " of " + str(60) + " experiments done")

    # save the data
    np.savez_compressed('data/' + objType + '_' + phiType + '_alpha_' + str(N)+'.npz', robust=robustReturns)

    figure.outSample(objType, phiType, N)


def crossValidation(phiType, objType):
    """ use cross validation to select the best alpha when N is small

    :param phiType: phi-divergence type - cre/chi/m-chi
    :param objType: objective type - worst/sum
    """
    # set parameter
    c, v, s, l, trueProb = data.read("data/data.csv")
    budget = 1000
    demand = [4, 8, 10]
    # max of N
    N_max = 50

    # get the set of candidates of alpha
    alpha = [0.001, 0.01, 0.1]
    alphaTest = data.alphaSet(alpha)

    numDemand = trueProb.shape[0]
    numItem = trueProb.shape[1]

    # use 2 - fold because the limited data we have
    K = 2
    bestAlphas = [0.0]*int((N_max - 10)/10)
    # iterate for all N to be tested
    for N in range(10, N_max, 10):
        # calculate the number of data points for each set
        length = int(2*N / K)
        # store the distribution of each set
        allProb = []
        for k in range(K):
            allProb.append(data.sampleData(trueProb, length))

        results = np.zeros((len(alphaTest), 1))
        # iterate K times
        for k in range(K):
            # calculate the training and testing distribution
            testProb = allProb[k]
            trainProb = (sum(allProb) - allProb[k])/(K-1)

            # test the performance for each candidate alpha
            for i in range(len(alphaTest)):
                alpha = alphaTest[i]
                # calculate rho
                rhoc = []
                for j in range(numItem):
                    rhoc.append(data.rhoc(alpha, numDemand, phiType, N, trainProb[:, j]))

                # evaluate the returns
                m = model.robustModel(c, v, s, l, trainProb, budget, demand, rhoc, objType, phiType)
                m.optimize()
                order = [m.getVarByName("Q[" + str(j) + "]").getAttr("x") for j in range(numItem)]
                # get the value of the solution using the test data
                simObj = evaluate.objVal(c, v, s, l, testProb, order, demand, objType)
                results[i] = simObj

            # select the best alpha
            bestAlpha = 0
            for i in range(len(alphaTest) - 1):
                if (results[i + 1] - results[bestAlpha]) / results[bestAlpha] >= 0.005:
                    bestAlpha = i + 1
            bestAlphas[int(N/10 - 1)] += alphaTest[bestAlpha]

        # print the process
        print(str(N) + " experiments done")

    # calculate the average results
    alphaOut = [max(0.05, alpha / 2) for alpha in bestAlphas]
    # save the results
    np.savez_compressed('data/' + objType + '_' + phiType + '_bestAlpha.npz', alpha=alphaOut)


def afterCV(phiType, objType):
    """ test the out of sample performance on different N with its best alpha

    :param phiType: phi-divergence type - cre/chi/m-chi
    :param objType: objective type - worst/sum
    """
    # set parameter
    c, v, s, l, trueProb = data.read("data/data.csv")
    budget = 1000
    demand = [4, 8, 10]
    N_max = 1010

    # number of tests
    it = 100
    numDemand = trueProb.shape[0]
    numItem = trueProb.shape[1]

    # load the best alpha for small N
    loaded = np.load('data/' + objType + '_' + phiType + '_bestAlpha.npz', allow_pickle=True)
    bestAlphas = loaded['alpha']

    # return for different N
    SAAReturns = []
    robustReturns = []

    # fix the alpha for large N
    for N in range(10, N_max, 10):
        if N >= 50:
            alpha = 0.05
        else:
            alpha = bestAlphas[int(N/10 - 1)]

        Q = data.sampleData(trueProb, N)
        # calculate rho for the robust optimization problem
        rhoc = []
        for j in range(numItem):
            rhoc.append(data.rhoc(alpha, numDemand, phiType, N, Q[:, j]))

        # evaluate the returns
        modelType = "robust"
        minReturn, maxReturn, meanReturn, time = evaluate.statOut(
            c, v, s, l, budget, demand, modelType, objType, phiType, alpha, it, N, trueProb)

        robustReturns.append([minReturn, maxReturn, meanReturn])

        modelType = "det"
        minReturn, maxReturn, meanReturn, time = evaluate.statOut(
            c, v, s, l, budget, demand, modelType, objType, phiType, alpha, it, N, trueProb)

        SAAReturns.append([minReturn, maxReturn, meanReturn])

        # print the process
        if N % 50 == 0 and N >= 50:
            print(str(N / 10) + " of " + str(N_max / 10 - 1) + " experiments done")

    # save the data
    np.savez_compressed('data/' + objType + '_' + phiType + '_final.npz', robust=robustReturns, SAA=SAAReturns)

    figure.sanityCheck(objType, phiType)


def main(task):
    """ main function for executing the tasks

    :param task: name of the task
    """
    phiList = ["cre", "chi", "m-chi"]
    objList = ["sum", "worst"]

    if task == "sanity":
        for obiType in objList:
            sanityCheck("cre", obiType)
    elif task == "outSample":
        for phiType in phiList:
            for obiType in objList:
                for N in [10, 30, 50 ,100]:
                    outSample(phiType, obiType, N)
    elif task == "CV":
        for phiType in phiList:
            for obiType in objList:
                crossValidation(phiType, obiType)
    elif task == "final":
        for phiType in phiList:
            for obiType in objList:
                afterCV(phiType, obiType)
    else:
        raise Exception("Wrong task!")


if __name__ == "__main__":
    taskList = ["sanity", "outSample", "CV", "final"]
    main(taskList[2])


