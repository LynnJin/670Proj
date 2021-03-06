from gurobipy import *


def robustModel(c, v, s, l, Q, budget, demand, rho, objType, phiType):
    """ create a robust model

    :param c: purchase cost
    :param v: salvage value
    :param s: selling price
    :param l: cost of lost sale
    :param Q: true distribution of demand scenarios
    :param budget: order budget
    :param demand: demand scenarios
    :param objType: objective type - worst/sum
    :param phiType: phi-divergence type - cre/chi/m-chi
    :return: a robust model
    """
    numDemand = Q.shape[0]
    numItem = Q.shape[1]

    # create the model
    m = Model("newsVendor")
    # don't show log in console
    m.params.logtoconsole = 0

    # creat decision variables
    # primal decision variables
    q = m.addVars(numItem, vtype=GRB.CONTINUOUS, name="Q")
    z = m.addVars(numItem, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="z")
    # artificial decision variables
    f = m.addVars(numDemand, numItem, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="f")

    # dual decision variables
    lam = m.addVars(numItem, vtype=GRB.CONTINUOUS, name="lambda")
    eta = m.addVars(numItem, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="eta")
    y = m.addVars(numDemand, numItem, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="y")
    if phiType == "m-chi":
        w = m.addVars(numDemand, numItem, vtype=GRB.CONTINUOUS, name="w")

    # set objective function
    if objType == "sum":
        m.setObjective(quicksum(z[j] for j in range(numItem)), sense=GRB.MAXIMIZE)
    elif objType == "worst":
        min_z = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="worstReturn")
        m.addGenConstrMin(min_z, z)
        m.setObjective(min_z, sense=GRB.MAXIMIZE)
    else:
        raise Exception('Wrong objective type!')

    # add constraint
    # budget constraint
    m.addConstr(
        quicksum(c[j] * q[j] for j in range(numItem)) <= budget, name="budget")
    for j in range(numItem):
        # constraint in original objective function
        m.addConstrs(
            (-f[i, j] >= -s[j] * q[j] + demand[i] * (s[j] - v[j]) for i in range(numDemand)),
            name="unsoldCstr_" + str(j))
        m.addConstrs(
            (-f[i, j] >= l[j] * demand[i] - (v[j] + l[j]) * q[j] for i in range(numDemand)),
            name="lostCstr_" + str(j))

        # robustified constraint
        if phiType == "chi":
            m.addConstr(
                c[j]*q[j] + eta[j] + rho[j]*lam[j] + 2*lam[j]
                - 2*quicksum(y[i, j] * Q[i][j] for i in range(numDemand)) <= -z[j],
                name="robustified_"+str(j))
        elif phiType == "hel":
            m.addConstr(
                c[j] * q[j] + eta[j] + rho[j] * lam[j] - lam[j]
                + quicksum(y[i, j] * Q[i][j] for i in range(numDemand)) <= -z[j],
                name="robustified_"+str(j))
        elif phiType == "m-chi":
            m.addConstr(
                c[j] * q[j] + eta[j] + rho[j] * lam[j] - lam[j]
                + 0.25*quicksum(y[i, j] * Q[i][j] for i in range(numDemand)) <= -z[j],
                name="robustified_"+str(j))
        elif phiType == "cre":
            m.addConstr(
                c[j] * q[j] + eta[j] + rho[j] * lam[j] - 2*lam[j]
                + 2*quicksum(y[i, j] * Q[i][j] for i in range(numDemand)) <= -z[j],
                name="robustified_"+str(j))
        else:
            raise Exception('Wrong phi type!')

        for i in range(numDemand):
            # calculate expression in conjugate term
            con = LinExpr(-f[i, j] - eta[j])
            con1 = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="con1" + str(i) + str(j))
            con2 = m.addVar(vtype=GRB.CONTINUOUS, name="con2" + str(i) + str(j))
            if phiType == "chi":
                m.addConstr(con <= lam[j], name="conjugate_"+str(i)+str(j))
                m.addConstr(con1 == 0.5 * con)
                m.addConstr(con2 == 0.5 * (2*lam[j] - con))
                m.addQConstr(y[i, j]*y[i, j] + con1*con1 <= con2*con2, name="soc_"+str(i)+str(j))
            elif phiType == "hel":
                m.addConstr(con <= lam[j], name="conjugate_"+str(i)+str(j))
                m.addConstr(con1 == 0.5*(y[i, j] - lam[j] + con))
                m.addConstr(con2 == 0.5*(y[i, j] + lam[j] - con))
                m.addQConstr(lam[j]*lam[j] + con1*con1 <= con2*con2, name="soc_"+str(i)+str(j))
            elif phiType == "m-chi":
                m.addConstr(con + 2*lam[j] <= w[i, j], name="conjugate_"+str(i)+str(j))
                m.addConstr(con1 == 0.5*(lam[j] - y[i, j]))
                m.addConstr(con2 == 0.5*(lam[j] + y[i, j]))
                m.addQConstr(w[i, j]*w[i, j] + con1*con1 <= con2*con2, name="soc_"+str(i)+str(j))
            elif phiType == "cre":
                m.addConstr(con <= 2*lam[j], name="conjugate_"+str(i)+str(j))
                m.addConstr(con1 == 0.5*(y[i, j] - lam[j] + 0.5*con))
                m.addConstr(con2 == 0.5*(y[i, j] + lam[j] - 0.5*con))
                m.addQConstr(lam[j]*lam[j] + con1*con1 <= con2*con2, name="soc_"+str(i)+str(j))

    return m


def detModel(c, v, s, l, Q, budget, demand, objType):
    """ create a deterministice model

    :param c: purchase cost
    :param v: salvage value
    :param s: selling price
    :param l: cost of lost sale
    :param Q: true distribution of demand scenarios
    :param budget: order budget
    :param demand: demand scenarios
    :param objType: objective type - worst/sum
    :return: a deterministic model
    """
    numDemand = Q.shape[0]
    numItem = Q.shape[1]

    # create the model
    m = Model("newsVendor")
    # don't show log in console
    m.params.logtoconsole = 0

    # creat decision variables
    # primal decision variables
    q = m.addVars(numItem, vtype=GRB.CONTINUOUS, name="Q")
    z = m.addVars(numItem, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="z")
    # artificial decision variables
    f = m.addVars(numDemand, numItem, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="f")

    # set objective function
    if objType == "sum":
        m.setObjective(quicksum(z[j] for j in range(numItem)), sense=GRB.MAXIMIZE)
    elif objType == "worst":
        min_z = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="worstReturn")
        m.addGenConstrMin(min_z, z)
        m.setObjective(min_z, sense=GRB.MAXIMIZE)
    else:
        raise Exception('Wrong objective type!')

    # add constraint
    # budget constraint
    m.addConstr(
        quicksum(c[j] * q[j] for j in range(numItem)) <= budget, name="budget")
    for j in range(numItem):
        # constraint in original objective function
        m.addConstrs(
            (-f[i, j] >= -s[j]*q[j] + demand[i]*(s[j] - v[j]) for i in range(numDemand)),
            name="unsoldCstr_"+str(j))
        m.addConstrs(
            (-f[i, j] >= l[j]*demand[i] - (v[j] + l[j])*q[j] for i in range(numDemand)),
            name="lostCstr_"+str(j))

        m.addConstr(
            c[j]*q[j] - quicksum(Q[i, j]*f[i, j] for i in range(numDemand)) <= -z[j],
            name="robustified_"+str(j))

    return m
