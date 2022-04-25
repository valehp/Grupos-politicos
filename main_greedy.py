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
    N, M, L, E = args.num_personas, args.max_personas, args.num_grupos, args.exp

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

    if args.tipo_greedy == "normal": 	  g = PromedioGreedy(data.to_numpy(), M, L, len(EJES), promedios,  0.05, exponente=args.exp)
    elif args.tipo_greedy == "random": 	  g = IteratedRandomGreedy(data.to_numpy(), M, L, len(EJES), promedios,  0.05, it=20, exponente=args.exp)
    elif args.tipo_greedy == "iterativo": g = IteratedDestructiveGreedy(data.to_numpy(), M, L, len(EJES), promedios,  0.05, it=20, exponente=args.exp, D=args.delete, n=args.best_group, r=args.random)



    start = time.time()
    sol, x, objetivo, dimensiones = g.Usar()
    end = time.time()
    print(g.Objetivo)

    if args.save:
        """
        Guardar valores de la solución:
        - Tiempo
        - Valor objetivo obtenido
        - Promedio por eje
        """
        with open("Greedy.txt", "a") as file:
            file.write( "\n - N: {} M: {} L: {} Dataset: {} exponente: {} minmax: {}\n\tSolución:{}\n\tValor Objetivo:{}\n\tTiempo: {}\n\tPolítica: {} ; Género: {} ; Colaborativo: {}\n".format(N, M, L, args.dataset, args.exp, False, sol, objetivo, end-start, dimensiones[0], dimensiones[1], dimensiones[2]) )


        print("Tiempo : ", end-start)

        print("Valor objetivo: ", g.Objetivo)
        # Guardar resultados
        df = pd.DataFrame()
        df["Tipo greedy"] 		= [args.tipo_greedy]
        df["Cambio promedio"] 	= [cambiop]
        df["Valor objetivo"] 	= [g.Objetivo]
        df["Tiempo"] 			= [end - start]

        resultados = "resultados.csv"
        if resultados not in os.listdir(out):
            df.to_csv( out+resultados )
        else:
            new_df = pd.read_csv( out+resultados, index_col=0 )
            new_df = new_df.append( df, ignore_index=True )
            new_df.to_csv( out + resultados )


        df = pd.DataFrame()
        df["Tipo greedy"] 		= [args.tipo_greedy]
        df["Cambio promedio"] 	= [cambiop]
        df["N"] = [N]
        df["M"] = [M]
        df["L"] = [L]
        df["Dataset"] = [args.dataset]
        df["MinMax"] = [False]
        df["Exp"] = [args.exp]

        df["Politica"] = [dimensiones[0]]
        df["Genero"] = [dimensiones[1]]
        df["Colaborativo"] = [dimensiones[2]]

        df["Valor objetivo"] = [g.Objetivo]
        df["Tiempo"] 		 = [end - start]
        df["Solucion"] 		 = [sol]

        
        if "ResultadosGreedy.csv" not in os.listdir("./"):
            df.to_csv( "ResultadosGreedy.csv" )
        else:
            new_df = pd.read_csv( "ResultadosGreedy.csv", index_col=0 )
            new_df = new_df.append( df, ignore_index=True )
            new_df.to_csv( "ResultadosGreedy.csv" )
	
    if args.grafico:
        out = "./Resultados/{}/".format(args.dataset)
        params = "N={} L={} M={}".format(N, L, M)
        if params not in os.listdir(out): os.mkdir(out + params)
        out += params + "/"
        guardar_dataset = out

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
        plt.savefig( "{}Promedios exp{} {} {}".format(out, args.exp, args.tipo_greedy, cambiop) )
        






if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Formación de grupos colabnorativos con greedy')
	parser.add_argument('--t', dest='time', action='store_true')
	parser.add_argument('--g', dest='grafico', action='store_true')
	parser.add_argument('--nt', dest='time', action='store_false')
	parser.add_argument('--ng', dest='grafico', action='store_false')
	parser.set_defaults(time=False)
	parser.set_defaults(grafico=False)
	parser.add_argument('--s', dest='save', action='store_true')
	parser.set_defaults(save=False)

	parser.add_argument( '-tipog', '--tipo_greedy', type=str, choices=["normal", "random", "iterativo"], default="iterativo" )

	parser.add_argument('-n', '--num_personas', type=int, default=10)
	parser.add_argument('-l', '--num_grupos', type=int, default=5)
	parser.add_argument('-m', '--max_personas', type=int, default=2)
	parser.add_argument('-ds', '--dataset', type=int, default=1)
	parser.add_argument('-b', '--bins', type=int, default=4)
	parser.add_argument('-e', '--exp', type=int, default=1)

	parser.add_argument('-r', '--random', type=float, default=0.05)     # Porcentaje de aleatoriedad
	parser.add_argument('-bg', '--best_groups', type=int, default=1)    # Número de mejores grupos a guardar en lista elegible
	parser.add_argument('-d', '--delete', type=int, default=1)          # Número de grupos a eliminar 

	args = parser.parse_args()
	print("\n\n", "="*100, "\n",args)

	graficar_greedy(args)