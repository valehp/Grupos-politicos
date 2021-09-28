from greedy import *
from GeneradorDataset import GenerarDatos
import numpy as np
import pandas as pd 
import random, time
import matplotlib.pyplot as plt
import argparse
import scipy.stats as ss


def plot_ejes(f, ax, i, data, bins, titles, colors, ylabel, p=False, k=False):
	
	for j in range(4):
		ax[i][j].hist( data[:,j], bins=bins[j], color=colors[j] )
		num = ""
		if p: num = np.mean(data[:,j])
		if k: num = ss.kurtosis(data[:,j])
		ax[i][j].set_title( "{} ({:.3f})".format(titles[j], num ) , size=20 )
	ax[i][0].set_ylabel(ylabel, size=20)



def graficar_greedy(args):
	#data = GenerarDatos(10000, ['economia', 'diplomacia', 'estado', 'sociedad'])
	if args.dataset == 1:
		file = "./dataset/Dataset4ejes_10000.csv"
		out  = "./Gráficos/"
	elif args.dataset == 2:
		file = "./dataset/Dataset4ejes_10000_(2).csv"
		out  = "./Gráficos/(E) "
	elif args.dataset == 3:
		file = "./dataset/Dataset4ejes_10000_(3).csv"
		out  = "./Gráficos/(S) "
	elif args.dataset == 4:
		#data.to_csv("./dataset/Dataset4ejes_10000_(4).csv")		# -> cambio distribución de sociedad y estado
		file = "./dataset/Dataset4ejes_10000_(4).csv"
		out  = "./Gráficos/(E+S) "

	

	#data.to_csv("./dataset/Dataset4ejes_10000.csv")
	#data.to_csv("./dataset/Dataset4ejes_10000_(2).csv")		# -> cambio distribución de estado a "U"
	#data.to_csv("./dataset/Dataset4ejes_10000_(3).csv")		# -> cambio distribución de sociedad a inclinación derecha
	#data.to_csv("./dataset/Dataset4ejes_10000_(4).csv")		# -> cambio distribución de sociedad y estado

	data = pd.read_csv(file, index_col=0)

	N, M, L = args.num_personas, args.max_personas, args.num_grupos
	data = data.head(N)

	if args.kurtosis:
		g = GreedyKurtosis(data.to_numpy(), M, L, 0.05)
		out += "K-"
	elif args.promedio: g = Greedy(data.to_numpy(), M, L, 0.05)
	start = time.time()
	sol = g.Usar()
	end = time.time()

	print("Tiempo : ", end-start)
	if args.time:
		with open ("times.txt", "a") as file:
			tipo = "Promedio" if (args.promedio) else "Kurtosis"
			file.write( "* {} N={} M={} L={} dataset {} : {} s\n".format(tipo, N,M,L, args.dataset, end-start) )

	if args.grafico:
		colors = ["firebrick", "limegreen", "turquoise", "purple"]
		BINS = []
		for i in range(4):
			_, b = np.histogram( data.to_numpy()[:,i], bins=args.bins )
			BINS.append(b)

		f, ax = plt.subplots( L+1, 4, figsize=(30, 6*(L+1)) )

		plot_ejes(f, ax, 0, data.to_numpy(), BINS, ['economia', 'diplomacia', 'estado', 'sociedad'], colors, "Gráfico de la población", p=args.promedio, k=args.kurtosis)

		for i in range(L):
		    plot_ejes(f, ax, i+1, np.array(sol[i]), BINS, ["", "", "", ""], colors, "Grupo " + str(i+1), p=args.promedio, k=args.kurtosis )


		#f.suptitle("Gráficos para {} personas: {} grupos de {}.".format(N, L, M), size=25)
		f.tight_layout()
		plt.savefig( "{}N={} M={} L={} B={}".format(out, N,M,L, args.bins) )
		#plt.show()




if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Formación de grupos colabnorativos con greedy')
	parser.add_argument('--p', dest='promedio', action='store_true')
	parser.add_argument('--np', dest='promedio', action='store_false')
	parser.add_argument('--k', dest='kurtosis', action='store_true')
	parser.add_argument('--nk', dest='kurtosis', action='store_false')
	parser.set_defaults(promedio=True)
	parser.set_defaults(kurtosis=False)
	parser.add_argument('--T', dest='time', action='store_true')
	parser.add_argument('--G', dest='grafico', action='store_true')
	parser.add_argument('--NT', dest='time', action='store_false')
	parser.add_argument('--NG', dest='grafico', action='store_false')
	parser.set_defaults(time=False)
	parser.set_defaults(grafico=True)
	parser.add_argument('-n', '--num_personas', type=int, default=10)
	parser.add_argument('-l', '--num_grupos', type=int, default=5)
	parser.add_argument('-m', '--max_personas', type=int, default=2)
	parser.add_argument('-d', '--dataset', type=int, default=1)
	parser.add_argument('-b', '--bins', type=int, default=4)
	args = parser.parse_args()
	print(args)

	graficar_greedy(args)

