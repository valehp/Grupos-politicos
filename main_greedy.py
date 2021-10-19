from greedy import *
from GeneradorDataset import GenerarDatos
import numpy as np
import pandas as pd 
import random, time
import matplotlib.pyplot as plt
import argparse
import scipy.stats as ss
import os

EJES = ['economia', 'diplomacia', 'estado', 'sociedad']
TITLES=['Economía', 'Diplomacia', 'Estado', 'Sociedad', 'Género']

def plot_ejes(f, ax, i, data, bins, titles, colors, ylabel, p=False, k=False):
	
	for j in range(4):
		ax[i][j].hist( data[:,j], bins=bins[j], color=colors[j] )
		num = ""
		if p: num = np.mean(data[:,j])
		if k: num = ss.kurtosis(data[:,j])
		ax[i][j].set_title( "{} ({:.3f})".format(titles[j], num ) , size=20 )
	ax[i][0].set_ylabel(ylabel, size=20)



def graficar_greedy(args):
	N, M, L = args.num_personas, args.max_personas, args.num_grupos
	carpeta = "./Cambio promedio2/"


	#carpeta, promedios = "./Resultados/", np.array([])

	params = "N={} L={} M={}".format(N, L, M)
	if params not in os.listdir(carpeta): os.mkdir(carpeta + params)
	out = carpeta + params + "/"
	
	if args.dataset == 1:								# Todas las distribuciones son normales
		file = "./dataset/Dataset4ejes_10000.csv"

	elif args.dataset == 2:								# Cambio de distribución de estado
		file = "./dataset/Dataset4ejes_10000_(2).csv"
		if "Dataset 2" not in os.listdir(out): os.mkdir(out + "Dataset 2")
		out += "Dataset 2/"

	elif args.dataset == 3:								# Cambio de distribución de sociedad
		file = "./dataset/Dataset4ejes_10000_(3).csv"
		if "Dataset 3" not in os.listdir(out): os.mkdir(out + "Dataset 3")
		out += "Dataset 3/"

	elif args.dataset == 4:								# Cambio de distribución de estado y sociedad
		file = "./dataset/Dataset4ejes_10000_(4).csv"
		if "Dataset 4" not in os.listdir(out): os.mkdir(out + "Dataset 4")
		out += "Dataset 4/"
		#promedios = np.array([0.4, 0.4, 0.4, 0.4])
	

	data = pd.read_csv(file, index_col=0)
	data = data.head(N)
	promedios = []
	for eje in EJES:
		promedios.append(  np.mean(data[eje]) )
	promedios = np.array(promedios)
	promedios += 0.1

	if args.kurtosis:
		g = GreedyKurtosis(data.to_numpy(), M, L, len(EJES), 0.05)
		out += "K-"
	elif args.promedio: g = PromedioGreedy(data.to_numpy(), M, L, len(EJES), promedios,  0.05)
	start = time.time()
	sol, x = g.Usar()
	end = time.time()

	print("Tiempo : ", end-start)

	print("Promedios: ", g.ObjGenerales)

	mejor, peor = g.Mejor_Peor( out+"promedios.txt" )
	print( "MEJOR -> Grupo {} \n\t {} -> dif: {}  \n".format(mejor[1], g.obj_grupales[mejor[1]-1] , mejor[0]) )
	print( "PEOR  -> Grupo {} \n\t {} -> dif: {}  \n".format(peor[1], g.obj_grupales[peor[1]-1] , peor[0]) )


	if args.grafico:

		# Graficar distribuciones por grupos
		colors = ["lightcoral", "limegreen", "turquoise", "violet"]
		BINS = []
		for i in range(4):
			_, b = np.histogram( data.to_numpy()[:,i], bins=args.bins )
			BINS.append(b)

			# --- Graficar las 4 distribuciones de toda la población --- #		hist( data[:,j], bins=bins[j], color=colors[j] )
		size_grafico = (20, 5)
		
		print("Graficando distribuciones...")
		f, ax = plt.subplots( 1, 4, figsize=size_grafico )
		for e in range(len(EJES)):
			ax[e].hist( data.to_numpy()[:,e], bins=BINS[e], color=colors[e] )
			ax[e].set_title(TITLES[e], size=20)
		f.tight_layout()
		f.savefig( "{}Poblacion".format(out) )
		plt.close(f)


			# --- Graficar cada uno de los grupos por eje --- #
		print("Graficando grupos...")
		for j in range(L):
			grupo = np.array(sol[j])
			f, ax = plt.subplots( 1, 4, figsize=size_grafico )
			for e in range(len(EJES)):
				ax[e].hist( grupo[:,e], bins=BINS[e], color=colors[e] )
				ax[e].set_title(TITLES[e], size=20)
			f.tight_layout()
			f.savefig( "{}Grupo {}".format(out, j+1) )
			plt.close(f)

		

			# --- Graficar promedios por grupos --- #
		print("Graficando promedios...")
		f, ax = plt.subplots( 2, 2, figsize=(10, 10))
		x = [ i+1 for i in range(L) ]
		i, j = 0, 0
		for e in range(len(EJES)):
			y = []
			for l in range(L):
				grupo = np.array(sol[l])
				y.append( np.mean(grupo[:,e]) )
			ax[i][j].bar( x, y, color=colors[e] )
			ax[i][j].axhline( y=g.ObjGenerales[e], ls='--', color="black", lw=3.5, label="Promedio general" )
			ax[i][j].legend()
			ax[i][j].set_title(TITLES[e], size=20)
			i += 1
			if i == 2: i, j = 0, 1
		f.tight_layout()
		plt.savefig( "{}Promedios".format(out) )





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
	parser.set_defaults(time=True)
	parser.set_defaults(grafico=False)
	parser.add_argument('-n', '--num_personas', type=int, default=10)
	parser.add_argument('-l', '--num_grupos', type=int, default=5)
	parser.add_argument('-m', '--max_personas', type=int, default=2)
	parser.add_argument('-d', '--dataset', type=int, default=1)
	parser.add_argument('-b', '--bins', type=int, default=4)
	args = parser.parse_args()
	print(args)

	graficar_greedy(args)

	# 	métrica -> error absoluto
	# AGREGAR GENERO
	# PONDERADOR ENTRE GÉNERO E IDEOLOGÍA POLÍTICA
	#	-> kurtosis podría tener una ventaja??

	# TERMINAR PROPUESTA
	# pedir a secretaria el formato de tesis de postgrado -> empezar a escribir 
	# expresar el problema como modelo de optimización


	# PROPUESTA
	#poner nuevo modelo
	# poner que ya se tiene una heuristica con greedy
	# poner graficos con resultados
	# Distribución objetivo y genero será entregada


	# EN LA DEFINICION DEL PROBLEMA -> PONER LO DE QUE VAMOS A BUSCAR OTROS PROMEDIOS (PAG 7)
		# - CAMBIO DEL VALOR OBJETIVO



	# aleatoriedad -> epsilon greedy (por ej: 5 mejores grupos y tiro la moneda)


	# PARA 26/OCT:
		# GRASP SOLO CON GREEDY ALEATORIZADO -> GREEDY RANDOM ITERATIVO