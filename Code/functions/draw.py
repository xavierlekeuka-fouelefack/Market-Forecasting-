import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

con = sqlite3.connect('run_data.db')
data = pd.read_sql_query("SELECT * from X20 WHERE LATENT_DIM=11 ORDER BY RETURN DESC LIMIT 1", con)

def avg(l):
    return sum(l)/len(l)

pips = []
for i in range(len(data.PIPS)):
    pips.append([float(x) for x in data.PIPS[i].split(";")])
# Liste pour chaque exécution de l'algorithme la liste des PIP moyens des clusters obtenus

sizes = []
for i in range(len(data.SIZES)):
    sizes.append([int(x) for x in data.SIZES[i].split(";")])
# Liste pour chaque exécution de l'algorithme la liste des tailles des clusters obtenus

spreads = []
for i in range(len(data.SPREADS)):
    spreads.append([float(x) for x in data.SPREADS[i].split(";")])
# Liste pour chaque exécution de l'algorithme la liste des spreads (écart-type au centre, distance euclidienne) des clusters obtenus

avg_pip = []
avg_size = []
avg_spread = []
for i in range(len(data.SPREADS)):
    avg_pip.append(avg(pips[i]))
    avg_size.append(avg(sizes[i]))
    avg_spread.append(avg(spreads[i]))

PnLs = []
for line in data.CLUSTER_PnLs:
    info = line.split('|')[1:]
    filtered = [x if len(x)>0 else "No trade" for x in info]
    PnLs.append([[float(x) if x!="No trade" else x for x in cluster_data.split(';') ] for cluster_data in filtered])

avg_PnLs = [[avg(l) if l[0]!="No trade" else "No trade" for l in line] for line in PnLs]

X = [abs((float(x))) for line in spreads for x in line]
Y = [x for l in avg_PnLs for x in l]
Z = [(float(a)) for line in sizes for a in line]


real_data = [Y[i]!="No trade" for i in range(len(X))]
X = [X[i] for i in range(len(X)) if real_data[i]]
Y = [Y[i] for i in range(len(Y)) if real_data[i]]
Z = [Z[i] for i in range(len(Z)) if real_data[i]]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.grid()
ax.scatter3D(X,Z,Y)
ax.set_xlabel("Dispersion moyenne du cluster")
ax.set_zlabel("Gain moyen du cluster")
ax.set_ylabel("Taille du cluster")
plt.show()

"""
reg = LinearRegression()
reg.fit([[x] for x in X], Y)
print("Coefficient de la droite : "+str(reg.coef_))

y_predit = reg.predict([[x] for x in X])
print("Coefficent de corrélation : "+str(r2_score(Y,y_predit)))

plt.figure()
plt.scatter(X, Y)
plt.plot(X,y_predit)
plt.xlabel("pips")
plt.ylabel("avg_PnLs")
plt.show()
"""

