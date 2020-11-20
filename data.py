import pandas as pd
import numpy as np
from scipy.stats import chi2


def read(fileName):
    """ Read the model parameter from data file

    :param fileName: name of the data file
    :return: model parameters
    """
    allData = pd.read_csv(fileName)
    c = np.array(allData["c"])
    v = np.array(allData["v"])
    s = np.array(allData["s"])
    l = np.array(allData["l"])
    q1 = np.array(allData["q1"])
    q2 = np.array(allData["q2"])
    q3 = np.array(allData["q3"])
    Q = np.concatenate(([q1], [q2], [q3]), axis=0)
    return c, v, s, l, Q


def rhoa(alpha, M, type, N):
    """ calculate the approximated rho

    :param alpha: confidence region
    :param M: number of scenarios
    :param type: type pf phi-divergence
    :param N: number of samples
    :return: approximated upper bound of the phi-divergence
    """
    phi_grad = {"chi": 2, "m-chi": 2, "hel": 0.5, "cre": 1}
    grad = phi_grad.get(type)

    chi2_p = chi2.ppf(1 - alpha, M - 1)
    rho = chi2_p * grad / (2 * N)

    return rho


def rhoc(alpha, M, type, N, p):
    """ calculate the corrected rho

    :param alpha: confidence region
    :param M: number of scenarios
    :param type: type pf phi-divergence
    :param N: number of samples
    :param p: empirical distribution
    :return: corrected upper bound of the phi-divergence
    """
    # second - forth derivative of different types of divergence
    phi_grad = {'chi':[2, -6, 24], 'm-chi': [2, 0, 0], 'hel': [0.5, -0.75, 1.875], 'cre': [1, -1.5, 3.75]}
    for item in p:
        if item == 0:
            rho = rhoa(alpha, M, type, N)
            return rho

    S = sum([1/item for item in p])
    grad = phi_grad.get(type)

    delta = 2 - 2*M - np.square(M) + S
    delta += 2*grad[1]*(4 - 6*M - np.square(M) + 3*S)/grad[0]
    delta += np.square(grad[1]/grad[0])*(4 - 6*M - 3*np.square(M) + 5*S)/3
    delta += 2*grad[2]*(1 - 2*M + S)/grad[0]
    delta = 1 + delta/(2*(M-1)*N)

    eta = grad[1]*(2 - 3*M + S)/(3*grad[0]) + grad[2]*(1 - 2*M + S)/(4*grad[0])
    eta = (M-1)*(1 - np.sqrt(delta)) + eta/N

    chi2_p = chi2.ppf(1 - alpha, M-1)

    rho = (eta + np.sqrt(delta)*chi2_p)*grad[0]/(2*N)

    return rho


def sampleProb(Q, rho, M):
    """ sample true distribution for testing

    :param Q: 3*12 array, empirical distribution
    :param rho: 1*12 list, upper bound for each item
    :param M: number of scenarios
    :return: 3*12 array sampled true distribution
    """
    prob = np.ones((Q.shape[0], Q.shape[1]))
    for j in range(Q.shape[1]):
        while sum(prob[0:Q.shape[0]-1, j]) > 1:
            for i in range(Q.shape[0]-1):
                # use the method in paper, take 95% confidence
                delta = min(0.5*Q[i, j], 0.5*np.sqrt(rho[j]*Q[i, j]/M))
                prob[i][j] = np.random.normal(Q[i, j], delta)
                while prob[i][j] <= 0:
                    prob[i][j] = np.random.normal(Q[i, j], delta)
        prob[Q.shape[0]-1, j] = 1 - sum(prob[0:Q.shape[0]-1, j])

    return prob


def sampleData(Q, N):
    """ sample empirical distribution for training

    :param Q: 3*12 array true distribution
    :param N: number of training sample
    :return: 3*12 array sampled empirical distribution
    """
    numItem = Q.shape[1]
    numDemand = Q.shape[0]
    # sample data based on the true distribution
    probSim = np.zeros((numDemand, numItem))
    for n in range(N):
        s = np.random.uniform(0, 1, numItem)
        for j in range(numItem):
            if s[j] <= Q[0, j]:
               probSim[0][j] += 1
            elif s[j] <= Q[1, j] + Q[0, j]:
                probSim[1][j] += 1
            else:
                probSim[2][j] += 1
    # calculate the empirical distribution
    probSim = np.array([[round(probSim[j][i]/N, 4) for i in range(numItem)] for j in range(numDemand)])

    # make sure the total probability is 1
    for j in range(numDemand):
        if abs(sum(probSim[:, j]) - 1) <= 0.001:
            probSim[2][j] = 1 - sum(probSim[0:2, j])
        else:
            raise Exception('Wrong simulation!')

    return probSim


def alphaSet(alpha):
    """ generate the list of alpha for testing, increase by 10% when alpha <= 0.1, 1% otherwise

    :param alpha: break point of confidence level
    :return: list of candidate alpha
    """
    alphaTest = []
    for i in range(len(alpha)):
        if i != len(alpha) - 1:
            diff = (alpha[i + 1] - alpha[i]) / 9.0
            alphaTest.extend(np.arange(alpha[i], alpha[i + 1], diff))
    alphaTest.extend(np.arange(0.1, 0.401, 0.01))
    alphaTest = np.round_(alphaTest, 6)
    return alphaTest
