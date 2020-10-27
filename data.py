import pandas as pd
import numpy as np
from scipy.stats import chi2

def read(fileName):
    allData = pd.read_csv(fileName)
    #print(allData)
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
    phi_grad = {"chi": 2, "m-chi": 2, "hel": 0.5}
    grad = phi_grad.get(type)

    chi2_p = chi2.ppf(1 - alpha, M - 1)

    rho = chi2_p*grad/(2*N)

    return rho


def rhoc(alpha, M, type, N, p):
    phi_grad = {'chi':[2, -6, 24], 'm-chi':[2, 0, 0], 'hel':[0.5, -0.75, 1.875]}
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
    prob = np.ones((Q.shape[0], Q.shape[1]))
    for j in range(Q.shape[1]):
        while sum(prob[0:Q.shape[0]-1, j]) > 1:
            for i in range(Q.shape[0]-1):
                delta = min(0.5*Q[i, j], 0.5*np.sqrt(rho[j]*Q[i, j]/M))
                prob[i][j] = np.random.normal(Q[i, j], delta)
        prob[Q.shape[0]-1, j] = 1 - sum(prob[0:Q.shape[0]-1, j])

    return prob
