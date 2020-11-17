import data

c, v, s, l, trueProb = data.read("data.csv")
length = int(10 / 2)
allProb = []
for k in range(5):
    allProb.append(data.sampleData(trueProb, length))
for k in range(5):
    testProb = allProb[k]
    trainProb = (sum(allProb) - allProb[k])/4
    print(trainProb)