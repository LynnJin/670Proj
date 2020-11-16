import model
import data

def objVal(c, v, s, l, Q, order, demand, objType):
    numDemand = Q.shape[0]
    numItem = Q.shape[1]

    if objType == "sum":
        objVal = 0
        for j in range(numItem):
            profits = [min(s[j] * order[j] - demand[i] * (s[j] - v[j]),
                           -l[j] * demand[i] + (v[j] + l[j]) * order[j]) for i in range(numDemand)]
            objVal += sum([profit*prob for profit, prob in zip(profits, Q[:, j])])

        objVal -= sum([order[j]*c[j] for j in range(numItem)])

    elif objType == "worst":
        objVal = 1000
        for j in range(numItem):
            profits = [min(s[j] * order[j] - demand[i] * (s[j] - v[j]),
                           -l[j] * demand[i] + (v[j] + l[j]) * order[j]) for i in range(numDemand)]
            temp = sum([profit*prob for profit, prob in zip(profits, Q[:, j])]) - order[j]*c[j]
            objVal = min(objVal, temp)

    else:
        raise Exception('Wrong objective type!')

    return objVal

def stat(c, v, s, l, Q, budget, demand, modelType, objType, phiType, it, rho=0, rhoTest=0, trueProb = []):
    numDemand = Q.shape[0]
    numItem = Q.shape[1]
    # solve robust model and get the optimal solutions
    if modelType == "robust":
        m = model.robustModel(c, v, s, l, Q, budget, demand, rho, objType, phiType)
    elif modelType == "det":
        m = model.detModel(c, v, s, l, Q, budget, demand, objType)
    else:
        raise Exception('Wrong model type!')

    m.optimize()
    order = [m.getVarByName("Q[" + str(j) + "]").getAttr("x") for j in range(numItem)]
    # evaluate the return
    if len(trueProb):
        Q = trueProb
    results = []
    times = 0
    for t in range(it):
        # sample probability
        prob = data.sampleProb(Q, rhoTest, numDemand)
        # evaluate the obj
        simObj = objVal(c, v, s, l, prob, order, demand, objType)
        results.append(simObj)

        if simObj >= m.objVal:
            times = times + 1

    # calculate the mean and range
    minReturn = min(results)
    maxReturn = max(results)
    meanReturn = sum(results) / len(results)

    return minReturn, maxReturn, meanReturn, m.objVal, times/it
