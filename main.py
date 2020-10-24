import gurobipy as gp
from gurobipy import GRB
import data

if __name__ == "__main__":
    c, v, s, l, Q = data.read("data.csv")
    print(c,v,s,l,Q)

