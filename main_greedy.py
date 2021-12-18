from greedy import *
from GeneradorDataset import GenerarDatos
import numpy as np
import pandas as pd 
import random, time
import matplotlib.pyplot as plt
import argparse
import scipy.stats as ss
import os
from Metricas import *

EJES  = ['economia-p', 'diplomacia-p', 'estado-p', 'sociedad-p', 'genero', 'economia-i', 'diplomacia-i', 'estado-i', 'sociedad-i']
TITLES= ['Economía (P)', 'Diplomacia (P)', 'Estado (P)', 'Sociedad (P)', 'Género', 'Economía (IDL)', 'Diplomacia (IDL)', 'Estado (IDL)', 'Sociedad (IDL)']



def graficar_greedy(args):
	N, M, L = args.num_personas, args.max_personas, args.num_grupos

	out = "./Resultados/{}/".format(args.dataset)
	params = "N={} L={} M={}".format(N, L, M)
	if params not in os.listdir(out): os.mkdir(out + params)
	out += params + "/"
	guardar_dataset = out

	file = "./Dataset/Dataset{}.csv".format(args.dataset)


 
	"""
	params = "N={} L={} M={}".format(N, L, M)
	if params not in os.listdir(carpeta): os.mkdir(carpeta + params)
	out = carpeta + params + "/"

	guardar_dataset = out
	
	if args.dataset == 1:								# Todas las distribuciones son normales
		file = "./5ejes/dataset_1.csv"

	elif args.dataset == 2:								# Cambio de una distribución
		file = "./5ejes/dataset_2.csv"
		if "Dataset 2" not in os.listdir(out): os.mkdir(out + "Dataset 2")
		out += "Dataset 2/"

	elif args.dataset == 3:								# Cambio de dos distribuciones
		file = "./5ejes/dataset_3.csv"
		if "Dataset 3" not in os.listdir(out): os.mkdir(out + "Dataset 3")
		out += "Dataset 3/"

	elif args.dataset == 4:								# Cambio de 3 distribuciones (todas distintas)
		nuevo = "Dataset 4"
		file = "./5ejes/dataset_4.csv"
		if nuevo not in os.listdir(out): os.mkdir(out + nuevo)
		out += nuevo + "/"
	"""

	data = pd.read_csv(file, index_col=0)
	data = data.head(N)

	promedios = []
	for eje in EJES:
		promedios.append(  np.mean(data[eje]) )
	promedios = np.array(promedios)
	if args.objetivo_custom:
		promedios = np.random.rand(len(EJES))
		print("NUEVO PROMEDIO: ", promedios)
	elif args.objetivo_cambio:
		if args.tipo_cambio == "+": promedios += 0.1
		else: promedios -= 0.1 


	if not args.objetivo_cambio: cambiop = 0
	else:
		if args.tipo_cambio == "+": cambiop = 1
		else: cambiop = -1

	"""
	if guardar_dataset:
		# Columnas: num, cambio promedio {-1, 0, 1}, promedio economia, promedio diplomacia, promedio estado, promedio sociedad, promedio genero 

		if "datasets.csv" not in os.listdir(guardar_dataset):
			df = pd.DataFrame()
			df["num"] = [args.dataset]
			df["Cambio promedio"] = [cambiop]
			for i in range(len(TITLES)):
				df[TITLES[i]] = [promedios[i]]
			df.to_csv(guardar_dataset + "datasets.csv")

		else:
			df = pd.read_csv( guardar_dataset+"datasets.csv", index_col=0 )
			df2= pd.DataFrame()
			aux =  df[ (df["num"] == args.dataset) & (df["Cambio promedio"] == cambiop) ].to_numpy()
			print( aux.any() , aux.all() , "\n" )
			if args.objetivo_custom or  (aux.any() and aux.all()):
				df2["num"] = [args.dataset]
				df2["Cambio promedio"] = [cambiop]
				for i in range(len(TITLES)):
					df2[TITLES[i]] = [promedios[i]]
				df = df.append(df2, ignore_index=True)
				df.to_csv(guardar_dataset + "datasets.csv")
	"""


	if args.tipo_greedy == "normal": 	  g = PromedioGreedy(data.to_numpy(), M, L, len(EJES), promedios,  0.05)
	elif args.tipo_greedy == "random": 	  g = RandomGreedy(data.to_numpy(), M, L, len(EJES), promedios,  0.05)
	elif args.tipo_greedy == "iterativo": g = IteratedGreedy(data.to_numpy(), M, L, len(EJES), promedios,  0.05, it=30)



	start = time.time()
	sol, x = g.Usar()
	end = time.time()


	"""
	Guardar valores de la solución:
	- Tiempo
	- Valor objetivo obtenido
	- Promedio por eje
	"""

	print("Tiempo : ", end-start)

	print("Promedios: ", g.ObjGenerales)

	mejor, peor = g.Mejor_Peor( out+"promedios.txt" )
	# diferencia, indice
	print( "MEJOR -> Grupo {} \n\t {} -> dif: {} \t valor: {} \n".format(mejor[1], g.obj_grupales[mejor[1]-1] , mejor[0], g.FuncionObjetivo()) )
	print( "PEOR  -> Grupo {} \n\t {} -> dif: {} \t valor: {} \n".format(peor[1], g.obj_grupales[peor[1]-1] , peor[0], g.FuncionObjetivo()) )

	# Guardar resultados
	df = pd.DataFrame()
	df["Tipo greedy"] 		= [args.tipo_greedy]
	df["Cambio promedio"] 	= [cambiop]
	df["Valor objetivo"] 	= [g.Objetivo]
	df["Tiempo"] 			= [end - start]
	for i in range(len(TITLES)):
		df["Promedio "+TITLES[i]] = [g.Objetivo_ejes[i]]

	resultados = "resultados.csv"
	if resultados not in os.listdir(out):
		df.to_csv( out+resultados )
	else:
		new_df = pd.read_csv( out+resultados, index_col=0 )
		new_df = new_df.append( df, ignore_index=True )
		new_df.to_csv( out + resultados )




	if args.grafico:

		# Graficar distribuciones por grupos
		#colors = ["lightcoral", "limegreen", "turquoise", "violet", "lightslategrey"]
		colors = ["firebrick", "darkorange", "sienna", "gold", "lightslategrey", "seagreen", "turquoise", "mediumblue", "blueviolet"]
		BINS = []
		for i in range(len(EJES)):
			bins = 3 if (EJES[i] == 'genero') else args.bins
			_, b = np.histogram( data.to_numpy()[:,i], bins=bins )
			BINS.append(b)

		# --- Graficar las 4 distribuciones de toda la población --- #		hist( data[:,j], bins=bins[j], color=colors[j] )
		size_grafico = (15, 15)
		e = 0
		
		if not "Poblacion.png" in os.listdir(out):
			print("Graficando distribuciones...")
			f, axs = plt.subplots( 3, 3, figsize=size_grafico )
			for i in range(3):
				for j in range(3):
					axs[i][j].hist(data.to_numpy()[:,e], bins=BINS[e], color=colors[e])
					axs[i][j].set_title(TITLES[e], size=20)
					if EJES[e] == 'genero':
						axs[i][j].set_xticks([0.2, 0.5, 0.8])
						axs[i][j].set_xticklabels(["F", "M", "NB"])
					e +=1 
			f.tight_layout()
			f.savefig( "{}Poblacion".format(out) )
			plt.close(f)
				#plt.show()


		# --- Graficar cada uno de los grup
		# Graficar mejor grupo
		mejor_grupo = sol[ mejor[1] ]
		f, axs = plt.subplots( 3, 3, figsize=size_grafico )
		e = 0 
		for i in range(3):
			for j in range(3):
				axs[i][j].hist(np.array(mejor_grupo)[:,e], bins=BINS[e], color=colors[e])
				axs[i][j].set_title(TITLES[e], size=20)
				if EJES[e] == 'genero':
					axs[i][j].set_xticks([0.2, 0.5, 0.8])
					axs[i][j].set_xticklabels(["F", "M", "NB"])
				e +=1 
		f.tight_layout()
		f.savefig( "{}Mejor {} {}".format(out, args.tipo_greedy, cambiop) )
		plt.close(f)



		# Graficar peor grupo
		peor_grupo = sol[ peor[1] ]
		f, axs = plt.subplots( 3, 3, figsize=size_grafico )
		e = 0 
		for i in range(3):
			for j in range(3):
				axs[i][j].hist(np.array(peor_grupo)[:,e], bins=BINS[e], color=colors[e])
				axs[i][j].set_title(TITLES[e], size=20)
				if EJES[e] == 'genero':
					axs[i][j].set_xticks([0.2, 0.5, 0.8])
					axs[i][j].set_xticklabels(["F", "M", "NB"])
				e +=1 
		f.tight_layout()
		f.savefig( "{}Peor {} {}".format(out, args.tipo_greedy, cambiop) )
		plt.close(f)



			# --- Graficar promedios por grupos --- #
		print("Graficando promedios...")
		f, axs = plt.subplots( 3, 3, figsize=size_grafico )		
		x = [ i+1 for i in range(L) ]	
		e = 0 
		for i in range(3):
			for j in range(3):

				y = []
				for l in range(L):
					grupo = np.array(sol[l])
					y.append( np.mean(grupo[:,e]) )
				axs[i][j].bar( x, y, color=colors[e] )
				axs[i][j].axhline( y=g.ObjGenerales[e], ls='--', color="black", lw=3.5, label="Promedio general" )
				axs[i][j].legend()
				axs[i][j].set_title(TITLES[e], size=20)
				e +=1 
		f.tight_layout()
		plt.savefig( "{}Promedios {} {}".format(out, args.tipo_greedy, cambiop) )

"""

				axs[i][j].hist(np.array(mejor_grupo)[:,e], bins=BINS[e])
				axs[i][j].set_title(TITLES[e], size=20)
				if EJES[e] == 'genero':
					axs[i][j].set_xticks([0.2, 0.5, 0.8])
					axs[i][j].set_xticklabels(["F", "M", "NB"])
				e +=1 
		f.tight_layout()
		f.savefig( "{}Mejor {} {}".format(out, args.tipo_greedy, cambiop) )
		plt.close(f)

		f, ax = plt.subplots( 2, 3, figsize=(20, 10))
		
		x = [ i+1 for i in range(L) ]
		i, j = 0, 0
		for e in range(len(EJES)):
			y = []
			for l in range(L):
				grupo = np.array(sol[l])
				y.append( np.mean(grupo[:,e]) )
			ax[j][i].bar( x, y, color=colors[e] )
			ax[j][i].axhline( y=g.ObjGenerales[e], ls='--', color="black", lw=3.5, label="Promedio general" )
			ax[j][i].legend()
			ax[j][i].set_title(TITLES[e], size=20)
			i += 1
			if i == 3: i, j = 0, 1
		f.tight_layout()
		plt.savefig( "{}Promedios {} {}".format(out, args.tipo_greedy, cambiop) )
"""





if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Formación de grupos colabnorativos con greedy')
	parser.add_argument('--p', dest='promedio', action='store_true')
	parser.add_argument('--np', dest='promedio', action='store_false')
	parser.add_argument('--k', dest='kurtosis', action='store_true')
	parser.add_argument('--nk', dest='kurtosis', action='store_false')
	parser.set_defaults(promedio=True)
	parser.set_defaults(kurtosis=False)
	parser.add_argument('--t', dest='time', action='store_true')
	parser.add_argument('--g', dest='grafico', action='store_true')
	parser.add_argument('--nt', dest='time', action='store_false')
	parser.add_argument('--ng', dest='grafico', action='store_false')
	parser.set_defaults(time=True)
	parser.set_defaults(grafico=False)
	parser.add_argument('--prom', dest='objetivo_promedio', action='store_true')
	parser.add_argument('--nprom', dest='objetivo_promedio', action='store_false')
	parser.add_argument('--cambio', dest='objetivo_cambio', action='store_true')
	parser.add_argument('--ncambio', dest='objetivo_cambio', action='store_false')
	parser.set_defaults(objetivo_promedio=True)
	parser.set_defaults(objetivo_cambio=False)
	parser.add_argument('--custom', dest='objetivo_custom', action='store_true')
	parser.set_defaults(objetivo_custom=False)


	parser.add_argument( '-tipog', '--tipo_greedy', type=str, choices=["normal", "random", "iterativo"], default="normal" )
	parser.add_argument( '-tipoc', '--tipo_cambio', type=str, choices=["+", "-", ""], default="" )

	parser.add_argument('-n', '--num_personas', type=int, default=10)
	parser.add_argument('-l', '--num_grupos', type=int, default=5)
	parser.add_argument('-m', '--max_personas', type=int, default=2)
	parser.add_argument('-d', '--dataset', type=int, default=1, choices=[1, 2, 3])
	parser.add_argument('-b', '--bins', type=int, default=4)
	args = parser.parse_args()
	print("\n\n", "="*100, "\n",args)

	graficar_greedy(args)