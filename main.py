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

    '''
    m = model.detModel(c, v, s, l, Q, budget, demand, objType)
    m.optimize()
    '''

    # return for different N
    detReturns = []
    robustReturns = []
    for N in range(10, N_max, 10):
        # calculate rho
        rhoa = [data.rhoa(alpha, numDemand, phiType, N)] * numItem
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

if __name__ == "__main__":
    outSample()
