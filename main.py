import data
import evaluate
import numpy as np

if __name__ == "__main__":
    #set parameter
    c, v, s, l, Q = data.read("data.csv")
    budget = 1000
    demand = [4, 8, 10]
    N_max = 1010
    it = 10000
    alpha = 0.05
    numDemand = Q.shape[0]
    numItem = Q.shape[1]
    phiType = "hel"
    objType = "worst"

    # return for different N
    detReturns = []
    robustReturns = []
    for N in range(10, N_max, 10):
        #calculate rho
        rhoa = [data.rhoa(alpha, numDemand, phiType, N)]*numItem
        rhoc = []
        for j in range(numItem):
            rhoc.append(data.rhoc(alpha, numDemand, phiType, N, Q[:, j]))

        # evaluate the returns
        modelType = "robust"
        minReturn, maxReturn, meanReturn = evaluate.stat(c, v, s, l, Q, budget, demand, modelType, objType, phiType, it, rhoc)
        robustReturns.append([minReturn, maxReturn, meanReturn])
        modelType = "det"
        minReturn, maxReturn, meanReturn = evaluate.stat(c, v, s, l, Q, budget, demand, modelType, objType, phiType, it, rhoc)
        detReturns.append([minReturn, maxReturn, meanReturn])

        # print the process
        if N % 50 == 0 and N >= 50:
            print(str(N/10) + " of " + str(N_max/10 - 1) + " experiments done")

    np.savez_compressed(objType+'_'+phiType+'_'+'check1.npz', robust=robustReturns, det=detReturns)
