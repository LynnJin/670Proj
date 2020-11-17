import data
import model
import evaluate
import numpy as np

def sanityCheck():
    # set parameter
    c, v, s, l, Q = data.read("data.csv")
    budget = 1000
    demand = [4, 8, 10]
    N_max = 1010
    it = 10000
    alpha = 0.05
    numDemand = Q.shape[0]
    numItem = Q.shape[1]
    phiType = "cre"
    objType = "sum"

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
        minReturn, maxReturn, meanReturn, solution = evaluate.stat(c, v, s, l, Q, budget, demand, modelType, objType, phiType, it,
                                                         rhoc, rhoc)
        robustReturns.append([minReturn, maxReturn, meanReturn, solution])
        modelType = "det"
        minReturn, maxReturn, meanReturn, solution = evaluate.stat(c, v, s, l, Q, budget, demand, modelType, objType, phiType, it,
                                                         rhoc, rhoc)
        detReturns.append([minReturn, maxReturn, meanReturn, solution])

        # print the process
        if N % 50 == 0 and N >= 50:
            print(str(N / 10) + " of " + str(N_max / 10 - 1) + " experiments done")

    np.savez_compressed(objType + '_' + phiType + '_' + 'check3.npz', robust=robustReturns, det=detReturns)

def outSample():
    # set parameter
    c, v, s, l, trueProb = data.read("data.csv")
    budget = 1000
    demand = [4, 8, 10]
    N = 10
    Q = data.sampleData(trueProb, N)
    alpha=[0.0001, 0.001, 0.01, 0.1]
    alphaTest = data.alphaSet(alpha)

    it = 1000
    numDemand = Q.shape[0]
    numItem = Q.shape[1]
    phiType = "m-chi"
    objType = "sum"

    # return for different N
    #SAAReturns = []
    robustReturns = []
    n = 0

    rhoTest = []
    for j in range(numItem):
        rhoTest.append(data.rhoc(0.05, numDemand, phiType, N, Q[:, j]))

    for alpha in alphaTest:
        # calculate rho
        rhoa = [data.rhoa(alpha, numDemand, phiType, N)] * numItem
        rhoc = []
        for j in range(numItem):
            rhoc.append(data.rhoc(alpha, numDemand, phiType, N, Q[:, j]))

        # evaluate the returns
        modelType = "robust"
        minReturn, maxReturn, meanReturn, solution, time = evaluate.stat(c, v, s, l, Q, budget, demand, modelType, objType, phiType, it,
                                                         rhoc, rhoTest, trueProb)
        robustReturns.append([minReturn, maxReturn, meanReturn, solution, time])

        '''
        modelType = "det"
        minReturn, maxReturn, meanReturn, solution, time = evaluate.stat(c, v, s, l, Q, budget, demand, modelType, objType, phiType, it,
                                                          rhoc, rhoTest, trueProb)
        SAAReturns.append([minReturn, maxReturn, meanReturn, solution, time])
        '''

        # print the process
        n = n + 1
        if n % 10 == 0 and n >= 10:
            print(str(n) + " of " + str(167) + " experiments done")

    np.savez_compressed(objType + '_' + phiType + '_alpha_' + str(N)+'.npz', robust=robustReturns)

def afterCV():
    # set parameter
    c, v, s, l, trueProb = data.read("data.csv")
    budget = 1000
    demand = [4, 8, 10]
    N_max = 1010

    it = 5000
    numDemand = trueProb.shape[0]
    numItem = trueProb.shape[1]
    phiType = "m-chi"
    objType = "worst"

    loaded = np.load(objType + '_' + phiType + '_bestAlpha.npz', allow_pickle=True)
    bestAlphas = loaded['alpha']

    # return for different N
    SAAReturns = []
    robustReturns = []
    for N in range(10, N_max, 10):
        if N >= 50:
            alpha = 0.05
        else:
            alpha = bestAlphas[int(N/10 - 1)]

        Q = data.sampleData(trueProb, N)
        # calculate rho
        rhoc = []
        for j in range(numItem):
            rhoc.append(data.rhoc(alpha, numDemand, phiType, N, Q[:, j]))

        rhoTest = []
        for j in range(numItem):
            rhoTest.append(data.rhoc(0.05, numDemand, phiType, N, Q[:, j]))

        # evaluate the returns
        modelType = "robust"
        minReturn, maxReturn, meanReturn, solution, time = evaluate.stat(c, v, s, l, Q, budget, demand, modelType, objType, phiType, it,
                                                         rhoc, rhoTest, trueProb)
        robustReturns.append([minReturn, maxReturn, meanReturn, solution, time])


        modelType = "det"
        minReturn, maxReturn, meanReturn, solution, time = evaluate.stat(c, v, s, l, Q, budget, demand, modelType, objType, phiType, it,
                                                          rhoc, rhoTest, trueProb)
        SAAReturns.append([minReturn, maxReturn, meanReturn, solution, time])


        # print the process
        if N % 50 == 0 and N >= 50:
            print(str(N / 10) + " of " + str(N_max / 10 - 1) + " experiments done")

    np.savez_compressed(objType + '_' + phiType + '_final.npz', robust=robustReturns, SAA=SAAReturns)

def crossValidation():
    # set parameter
    c, v, s, l, trueProb = data.read("data.csv")
    budget = 1000
    demand = [4, 8, 10]
    N_max = 50

    alpha = [0.0001, 0.001, 0.01, 0.1]
    alphaTest = data.alphaSet(alpha, 1)

    numDemand = trueProb.shape[0]
    numItem = trueProb.shape[1]
    phiType = "chi"
    objType = "worst"

    robustReturns = []

    #relLimits = [0.21, 0.34, 0.4, 0.4]
    relLimits = [0.05, 0.1, 0.2, 0.4]

    K = 2
    bestAlphas = []
    for N in range(10, N_max, 10):
        length = int(N / K)
        allProb = []
        for k in range(K):
            allProb.append(data.sampleData(trueProb, length))

        results = np.zeros((len(alphaTest), 1))
        for k in range(K):
            testProb = allProb[k]
            trainProb = (sum(allProb) - allProb[k])/(K-1)

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
                simObj = evaluate.objVal(c, v, s, l, testProb, order, demand, objType)
                results[i] += simObj

        results = results/K
        bestAlpha = 0
        for i in range(len(alphaTest)-1):
            if (results[i+1]-results[bestAlpha])/results[bestAlpha] >= 0.005:
                bestAlpha = i+1

        if alphaTest[bestAlpha] <= 0.05:
            bestAlphas.append(0.05)
        elif alphaTest[bestAlpha] >= relLimits[int(N/10 - 1)]:
            bestAlphas.append(relLimits[int(N/10 - 1)])
        else:
            bestAlphas.append(alphaTest[bestAlpha])
        # print the process
        print(str(N) + " experiments done")

    print(bestAlphas)
    np.savez_compressed(objType + '_' + phiType + '_bestAlpha.npz', alpha=bestAlphas)


if __name__ == "__main__":
    afterCV()


