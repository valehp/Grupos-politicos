from docplex.mp.model import Model
from docplex.mp.progress import *
import numpy as np
import pandas as pd
import argparse
import os
import time
import sys



EJES = ['economia', 'diplomacia', 'estado', 'sociedad']
BASE = "./dataset/"
FILE = "Registro_soluciones.csv"

def get_data(n, file):
	try:
		# Obtener dataset y dejar los datos en matriz 3d de dimensiones (4 x n x 4) = (num_ejes x n x num_segmentos)
		d = pd.read_csv( BASE+file, index_col=0 )
		assert len(d.index < n), "Dataset muy pequeño!"

	except FileNotFoundError:
		print("Archivos no encontrados")
		sys.exit()

	d = d.head(n).to_numpy()
	o = pd.DataFrame()
	datos = [0, 0, 0, 0]
	for i, j in zip( list(range(0, d.shape[1], 4)), list(range(4)) ):
		datos[j] = d[:, i:i+4]
		orden = []
		for jj in range(datos[j].shape[1]):
			orden.append( np.sum(datos[j][:,jj]) )

		aux = list(zip( orden, list(range(4)) ))
		aux.sort(reverse=True)
		o[EJES[j]] = [ aux[k][1] for k in range(len(aux)) ]

	return np.array(datos), o.to_numpy()


def modelo(data, orden, n, l, m, h=4, n_axs=4):
	model = Model("Modelo n={} l={} m={}".format(n, l, m))
	model.add_progress_listener( TextProgressListener() )
	#print("Data: ", data, "\n")

	# Variables -> matrix X de LxN
	x = []
	for j in range(l):
		x.append( model.binary_var_list(n, name="x{}".format(j)) )
	x = np.array(x)



	# Constraints // sujeto a:

	# ------------------------------------------------------------------------- #
	# 1) Para cada grupo, la cantidad de integrantes no debe superar el máximo m
	# La suma de cada fila debe ser <= m
	for j in range(l):
		model.add_constraint( model.sum([ x[j,i] for i in range(n) ]) <= m )


	# ------------------------------------------------------------------------- #
	# 2) Cada persona pertenece SOLO a UN grupo y ninguna se queda sin grupo
	# La suma de cada columna debe ser = 1
	for i in range(n):
		model.add_constraint( model.sum([ x[j,i] for j in range(l) ]) == 1 )




	# Función objetivo
	model.minimize(model.sum([ model.max([ model.scal_prod(x[j,:], data[e, :, orden[k+1,e]]) - model.scal_prod(x[j,:], data[e, :, orden[k,e]]), 0 ]) for k in range(h-1) for e in range(n_axs) for j in range(l) ]))
	
	# --- Escribir o imprimir el modelo --- #
	#model.write("modelo{}.lp".format(n))
	#model.get_cplex().write("modelito.lp")
	#with open("modelito_binaryvar.txt", "w") as file: file.write(model.export_as_lp_string())
	#print(model.export_as_lp_string())

	# k+1 = orden[k+1, e] ; k = orden[k, e]
	print("Modelo listo...")
	
	inicio = time.time()
	model.parameters.timelimit = 120 
	sol = model.solve(log_output=False)
	final = time.time()

	if sol:
		if not FILE in os.listdir(os.getcwd()):
			df_sol = pd.DataFrame(columns=["N", "Solucion", "Tiempo"])
		else:
			df_sol = pd.read_csv(FILE, index_col=0)

		print("Solución: ", sol.get_objective_value())
		print("Tiempo: ", final-inicio)
		df_sol = df_sol.append( pd.DataFrame( {"N":[n], "Solucion":[sol.get_objective_value()], "Tiempo":[final-inicio]} ), ignore_index=True )
		df_sol.to_csv(FILE)
		xx = []
		grupos = [ [] for i in range(l) ]
		#print(sol)

		for j in range(l):
			first = True
			aux = []
			for i in range(n):
				if sol.get_value( "x{}_{}".format(j,i)) == 1:
					grupos[j].append(str(i))
				aux.append(sol.get_value("x{}_{}".format(j,i)) )
			xx.append(aux)

		for i in range(l):
			print( "Grupo {}: {}".format(i, "\t".join(grupos[i])) )



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Formación de grupos colabnorativos usando cplex')
	parser.add_argument('-n', '--num_personas', type=int, default=10)
	parser.add_argument('-l', '--num_grupos', type=int, default=5)
	parser.add_argument('-m', '--max_personas', type=int, default=2)
	parser.add_argument('-seg', '--num_segmentos', type=int, default=4)	# segmentos = h
	parser.add_argument('-d', '--dataset_file', type=str, default="Dataset10000.csv")
	args = parser.parse_args()
	print(args)

	data, orden = get_data( args.num_personas, args.dataset_file )
	modelo( data, orden, args.num_personas, args.num_grupos, args.max_personas )


#model.minimize(
#	model.sum([ model.max([ model.scal_prod(x[j,:], data[e, :, orden[k+1,e]]) - model.scal_prod(x[j,:], data[e, :, orden[k,e]]), 0 ]) for k in range(h-1) for e in range(n_axs) for j in range(l) ])
#)