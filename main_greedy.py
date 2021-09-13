from greedy import Greedy
from GeneradorDataset import GenerarDatos
import numpy as np
import pandas as pd 
import random, time
import matplotlib.pyplot as plt
import argparse

def plot_ejes(f, ax, i, data, titles, colors, ylabel):

	for j in range(4):
		ax[i][j].hist( data[:,j], bins=4, color=colors[j] )
		ax[i][j].set_title( "{} ({:.3f})".format(titles[j], np.mean(data[:,j]) ) , size=20 )
	ax[i][0].set_ylabel(ylabel, size=20)



def graficar_greedy(args):
	#data = GenerarDatos(10000, ['economia', 'diplomacia', 'estado', 'sociedad'])
	#data.to_csv("./dataset/Dataset4ejes_10000.csv")

	data = pd.read_csv("./dataset/Dataset4ejes_10000.csv", index_col=0)

	N, M, L = args.num_personas, args.max_personas, args.num_grupos
	data = data.head(N)

	g = Greedy(data.to_numpy(), M, L, 0.05)
	start = time.time()
	sol = g.Usar()
	end = time.time()

	print("Tiempo : ", end-start)

	colors = ["firebrick", "limegreen", "turquoise", "purple"]
	f, ax = plt.subplots( L+1, 4, figsize=(30, 6*(L+1)) )

	plot_ejes(f, ax, 0, data.to_numpy(), ['economia', 'diplomacia', 'estado', 'sociedad'], colors, "Gráfico de la población")

	for i in range(L):
	    plot_ejes(f, ax, i+1, np.array(sol[i]), ["", "", "", ""], colors, "Grupo " + str(i+1) )


	#f.suptitle("Gráficos para {} personas: {} grupos de {}.".format(N, L, M), size=25)
	f.tight_layout()
	plt.savefig( "./Gráficos/N={} M={} L={}".format(N,M,L) )
	#plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Formación de grupos colabnorativos con greedy')
	parser.add_argument('-n', '--num_personas', type=int, default=10)
	parser.add_argument('-l', '--num_grupos', type=int, default=5)
	parser.add_argument('-m', '--max_personas', type=int, default=2)
	args = parser.parse_args()
	print(args)

	graficar_greedy(args)