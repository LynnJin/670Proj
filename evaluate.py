import model
import data


def objVal(c, v, s, l, Q, order, demand, objType):
    """ calculate the objective value given the sample probability of each demand

    :param c: purchase cost
    :param v: salvage value
    :param s: selling price
    :param l: cost of lost sale
    :param Q: true distribution of demand scenarios
    :param order: order of each item
    :param demand: demand scenarios
    :param objType: objective type - worst/sum
    :return: objective value under the true distribution Q
    """
    numDemand = Q.shape[0]
    numItem = Q.shape[1]

    if objType == "sum":
        objVal = 0
        for j in range(numItem):
            profits = [
                min(s[j] * order[j] - demand[i] * (s[j] - v[j]), -l[j] * demand[i] + (v[j] + l[j]) * order[j])
                for i in range(numDemand)]
            objVal += sum([profit*prob for profit, prob in zip(profits, Q[:, j])])

        objVal -= sum([order[j]*c[j] for j in range(numItem)])

    elif objType == "worst":
        objVal = 1000
        for j in range(numItem):
            profits = [
                min(s[j] * order[j] - demand[i] * (s[j] - v[j]), -l[j] * demand[i] + (v[j] + l[j]) * order[j])
                for i in range(numDemand)]
            temp = sum([profit*prob for profit, prob in zip(profits, Q[:, j])]) - order[j]*c[j]
            objVal = min(objVal, temp)

    else:
        raise Exception('Wrong objective type!')

    return objVal


def stat(
        c, v, s, l, Q, budget, demand,
        modelType, objType, phiType,
        it, rho
):
    """ for the solutions on empirical model, sample true distributions and return the mean and range

    :param modelType: model type - robust/det(deterministic)
    :param it: number of tests
    :param rho: upper bound of the phi-divergence for sample the tru distribution
    :return: mean, range of the results of sample distributions
    """
    numDemand = Q.shape[0]
    numItem = Q.shape[1]
    # solve optimization problem
    if modelType == "robust":
        m = model.robustModel(c, v, s, l, Q, budget, demand, rho, objType, phiType)
    elif modelType == "det":
        m = model.detModel(c, v, s, l, Q, budget, demand, objType)
    else:
        raise Exception('Wrong model type!')

    m.optimize()
    # get the optimal solutions
    order = [m.getVarByName("Q[" + str(j) + "]").getAttr("x") for j in range(numItem)]
    # evaluate the return
    results = []
    times = 0
    for t in range(it):
        # sample probability
        prob = data.sampleProb(Q, rho, numDemand)
        # evaluate the obj
        simObj = objVal(c, v, s, l, prob, order, demand, objType)
        results.append(simObj)

    # calculate the mean and range
    minReturn = min(results)
    maxReturn = max(results)
    meanReturn = sum(results) / len(results)

    return minReturn, maxReturn, meanReturn

def statOut(
        c, v, s, l, budget, demand,
        modelType, objType, phiType,
        alpha, it, N, trueProb
):
    """ solve the optimization problem on sampled empirical distribution,
    test the out of sample performance on the true distribution

    :param alpha: candidate alpha
    :param it: number of iterations
    :param N: number of samples
    :param trueProb: true distribution of the demands
    :return: mean, range and the times when result is lower than the optimal value
    """
    times = 0

    numDemand = trueProb.shape[0]
    numItem = trueProb.shape[1]

    results = []
    for i in range(it):
        Q = data.sampleData(trueProb, N)
        rhoc = []
        for j in range(numItem):
            rhoc.append(data.rhoc(alpha, numDemand, phiType, N, Q[:, j]))

        # solve optimization problem
        if modelType == "robust":
            m = model.robustModel(c, v, s, l, Q, budget, demand, rhoc, objType, phiType)
        elif modelType == "det":
            m = model.detModel(c, v, s, l, Q, budget, demand, objType)
        else:
            raise Exception('Wrong model type!')

        m.optimize()
        # get the optimal solutions
        order = [m.getVarByName("Q[" + str(j) + "]").getAttr("x") for j in range(numItem)]
        # evaluate the obj
        simObj = objVal(c, v, s, l, trueProb, order, demand, objType)
        # collect the times of result is lower than the optimal value
        if simObj >= m.objVal:
            times = times + 1
        results.append(simObj)

    # calculate the mean and range
    minReturn = min(results)
    maxReturn = max(results)
    meanReturn = sum(results) / len(results)

    return minReturn, maxReturn, meanReturn, times / it