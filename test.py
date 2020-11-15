import data

c, v, s, l, Q = data.read("data.csv")

alpha = [0.0001, 0.001, 0.01, 0.1]
alphaTest = data.alphaSet(alpha)

print(alphaTest)
print(len(alphaTest))