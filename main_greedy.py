from greedy import Greedy
from GeneradorDataset import GenerarDatos
import numpy as np
import pandas as pd 
import random, time

#data = GenerarDatos(10000, ['economia', 'diplomacia', 'estado', 'sociedad'])
#data.to_csv("./dataset/Dataset4ejes_10000.csv")

data = pd.read_csv("./dataset/Dataset4ejes_10000.csv", index_col=0)
print("Dataset listo")

N, M, L = 100, 10, 10
data = data.head(N)


# 




print("GREEDY")
g = Greedy(data.to_numpy(), M, L, 5e-1)
start = time.time()
sol = g.Usar()
end = time.time()

print("Tiempo : ", end-start)

for i in range(L):
	print("\nGrupo ", i, ": ")
	for p in sol[i]:
		print(p)
