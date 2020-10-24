import pandas as pd
import numpy as np

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
    Q = np.concatenate(([q1], [q2], [q3]), axis=0).T
    return c, v, s, l, Q


