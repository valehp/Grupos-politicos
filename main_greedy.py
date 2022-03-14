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

	data = pd.read_csv(file, index_col=0)
	data = data.head(N)
	vector_generos = np.array([0.33, 0.33, 0.33])

	promedios = [] 		# Valores objetivos
	for e in range(4): 	# Ejes politicos + genero, que usan promedios
		eje = EJES[e]
		promedios.append(  np.mean(data[eje]) )
	promedios.append(vector_generos)
	for e in range(5, len(EJES), 1):	# eje de grupos colaborativos, que usa la desviación estandar
		eje = EJES[e]
		#promedios.append( np.std(data[eje]) )
		promedios.append(0)
	promedios = np.array(promedios)
	print("Valores objetivos: ", promedios)

	if args.objetivo_custom:
		promedios = np.random.rand(len(EJES))
		print("NUEVO PROMEDIO: ", promedios)
	elif args.objetivo_cambio:
		promedios[0] += 0.2
		promedios[1] -= 0.3
		#if args.tipo_cambio == "+": promedios += 0.1
		#else: promedios -= 0.1 


	if not args.objetivo_cambio: cambiop = 0
	else:
		if args.tipo_cambio == "+": cambiop = 1
		else: cambiop = -1



	if args.tipo_greedy == "normal": 	  g = PromedioGreedy(data.to_numpy(), M, L, len(EJES), promedios,  0.05)
	elif args.tipo_greedy == "random": 	  g = IteratedRandomGreedy(data.to_numpy(), M, L, len(EJES), promedios,  0.05, it=20)
	elif args.tipo_greedy == "iterativo": g = IteratedDestructiveGreedy(data.to_numpy(), M, L, len(EJES), promedios,  0.05, it=20)



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

	#print("Promedios: ", g.ObjGenerales)

	""" pendiente arreglar
	mejor, peor = g.Mejor_Peor( out+"promedios.txt" )
	with open(out+"resultados.txt", "a") as f:
		f.write( "Greedy {} : {} | cambio: {}\n\n".format(args.tipo_greedy, g.FuncionObjetivo(), args.objetivo_cambio) )
	"""

	print("Valor objetivo: ", g.Objetivo)
	# diferencia, indice
	#print( "MEJOR -> Grupo {} \n\t {} -> dif: {} \t valor: {} \n".format(mejor[1], g.obj_grupales[mejor[1]-1] , mejor[0], g.FuncionObjetivo()) )
	#print( "PEOR  -> Grupo {} \n\t {} -> dif: {} \t valor: {} \n".format(peor[1], g.obj_grupales[peor[1]-1] , peor[0], g.FuncionObjetivo()) )

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
		GENEROS= ["dimgray","silver","lightslategrey"]
		BINS = []
		print("BINS: ", end="\t")
		for i in range(len(EJES)):
			bins = 3 if (EJES[i] == 'genero') else args.bins
			print(bins, end=" ")
			_, b = np.histogram( data.to_numpy()[:,i], bins=bins )
			BINS.append(b)
		print()

		# --- Graficar las 4 distribuciones de toda la población --- #		hist( data[:,j], bins=bins[j], color=colors[j] )
		size_grafico = (15, 15)
		e = 0
		
		#if not "Poblacion.png" in os.listdir(out):
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
		# PENDIENTE
		"""
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

		"""

			# --- Graficar promedios por grupos --- #
		print("Graficando promedios...")
		f, axs = plt.subplots( 3, 3, figsize=size_grafico )		
		x = [ i+1 for i in range(L) ]	
		e = 0 
		for i in range(3):
			for j in range(3):

				if EJES[e] == "genero":
					fe, m, nb = [], [], []
					for l in range(L):
						# 0		f
						# 0.5	nb
						# 1		m
						grupo = np.array(sol[l])
						fe.append( (grupo == 0).sum() / len(grupo) )
						nb.append( (grupo == 0.5).sum() / len(grupo) )
						m.append( (grupo == 1).sum() / len(grupo) )
					fe = np.array(fe)
					nb = np.array(nb)
					m = np.array(m)
					axs[i][j].bar( x, fe, color=GENEROS[0], label="F" )
					axs[i][j].bar( x, nb, color=GENEROS[1], bottom=fe, label="NB" )
					axs[i][j].bar( x, m, color=GENEROS[2], bottom=nb+fe, label="M" )
					#axs[i][j].axhline( y=g.ObjGenerales[e], ls='--', color="black", lw=3.5, label="Promedio general" )

				else:
					y = []
					for l in range(L):
						grupo = np.array(sol[l])
						if "IDL" in TITLES[e]: y.append( np.std(grupo[:,e]) )
						else: y.append( np.mean(grupo[:,e]) )

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
		print("Terminado")





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