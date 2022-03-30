import numpy as np
import pandas as pd
from itertools import permutations, combinations
import time
import argparse
import matplotlib.pyplot as plt

min_p, max_p = 0, 4
min_g, max_g = 0, 2
min_c, max_c = 0, 4
exponente = 1

def same(l):
    value = True
    for i in range(1, len(l), 1):
        if l[i] != l[i-1]:
            value = False
            break
    return value

def FuncionObjetivo(alpha, ejes_alpha, beta, ejes_beta, gamma, ejes_gamma, grupos, N, M, L, ObjGenerales, scale=False):
    print("FUNCIÓN OBJETIVO")
    # Componente política
    P = 0
    for j in range(L):
        grupo = np.array(grupos[j])
        #print("grupo ", j, ": ",grupo.shape)
        for i in range(ejes_alpha):
            promedios = np.mean( grupo[:,i] )
            dif = abs( ObjGenerales[i] - promedios ) ** exponente
            P += np.sum(dif)
    print("Componente política: ", P)


    # Componente de género
    G = 0
    for j in range(L):
        grupo = np.array(grupos[j])[:, ejes_alpha]
        f  = 0 if ((grupo==0).sum() == 0 ) else (grupo==0).sum()/len(grupo)
        nb = 0 if ((grupo==0.5).sum() == 0 ) else (grupo==0.5).sum()/len(grupo)
        m  = 0 if ((grupo==1).sum() == 0 ) else (grupo==1).sum()/len(grupo)
        #f, nb, m = (grupo==0).sum()/len(grupo), (grupo==0.5).sum()/len(grupo), (grupo==1).sum()/len(grupo)
        generos = np.array([f, nb, m])
        dif = abs( ObjGenerales[ejes_alpha] - generos ) ** exponente
        G += np.sum(dif)
    print("Componente género: ", G)


    # Componente de grupos colaborativos
    C = 0
    for j in range(L):
        grupo = np.array(grupos[j])
        for i in range(ejes_alpha+ejes_beta, ejes_alpha+ejes_beta+ejes_gamma, 1):
            std = np.std( grupo[:,i] ) ** exponente
            C += np.sum(std)
    print("Componente colab: ", C )

    if scale:
        P = (P - min_p) / (max_p - min_p)
        G = (G - min_g) / (max_g - min_g)
        C = (C - min_c) / (max_c - min_c)
    Objetivo = alpha * P + beta * G + gamma * C
    print(Objetivo)


def FuncionGrupal(alpha, ejes_alpha, beta, ejes_beta, gamma, ejes_gamma, grupo, N, M, L, PROMEDIOS, GENEROS, scale=False, imprimir=False  ):
    # Componente política
    P = 0
    for i in range(ejes_alpha):
        promedios = np.mean( grupo[:,i] )
        dif = abs( PROMEDIOS[i] - promedios ) ** exponente
        P += np.sum(dif)


    # Componente de género
    G = 0
    f  = 0 if ((grupo[:,ejes_alpha]==0).sum() == 0 ) else (grupo==0).sum()/len(grupo)
    nb = 0 if ((grupo[:,ejes_alpha]==0.5).sum() == 0 ) else (grupo==0.5).sum()/len(grupo)
    m  = 0 if ((grupo[:,ejes_alpha]==1).sum() == 0 ) else (grupo==1).sum()/len(grupo)
    generos = np.array([f, nb, m])
    dif = abs( GENEROS - generos ) ** exponente
    if imprimir: print("dif generos => {} - {} = {} ".format(GENEROS, generos, dif))
    G += np.sum(dif)


    # Componente de grupos colaborativos -> desviación estándar
    # CAMBIAR
    C = 0
    for i in range(ejes_alpha+ejes_beta, ejes_alpha+ejes_beta+ejes_gamma, 1):
        std = np.std( grupo[:,i] ) ** exponente
        C += np.sum(std)

    #Objetivo = alpha * P + beta * G + gamma * C

    if imprimir: print("Dimensión política: {} \nDimensión de género: {} \nDimensión colaborativa: {}\nTotal:{}\n\n".format(P, G, C, alpha*P+beta*G+gamma*C))
    if scale:
        P = (P - min_p) / (max_p - min_p)
        G = (G - min_g) / (max_g - min_g)
        C = (C - min_c) / (max_c - min_c)

    return np.array([P, G, C])


def Grupos(data, sizes):
    aux = []
    i = 0
    for j in sizes:
        aux.append( data[i:j,:] )
    return aux



def FuerzaBruta(data, alpha, ejes_alpha, beta, ejes_beta, gamma, ejes_gamma, N, M, L, PROMEDIOS, GENEROS, sizes, scale ):
    indices = list(range(N))
    best = float('inf')      # Guarda mejor valor de la función objetivo hasta el momento
    solucion = []
    peor = float('-inf')     # Guarda el peor valor de la función objetivo hasta el momento
    peor_sol = []
    peor_dims = []
    cont = 0
    
    if same(sizes): 
        combinaciones = {}
        tiempo = time.time()
        mins = 1
        dims = []
        for per in permutations(indices):
            #print("PER: ", cont)
            i = 0
            aux = []
            suma = np.zeros(3)
            for j in sizes:
                aux = []
                aux_g = list(per[i:i+j])
                #print("auxg: ", aux_g)
                aux_g.sort()
                aux_g = tuple(aux_g)
                if aux_g in combinaciones:
                    suma_grupal = combinaciones[aux_g]
                else: 
                    for x in aux_g:
                        aux.append( data[x,:] )
                    aux = np.array(aux)
                    suma_grupal = FuncionGrupal(alpha, ejes_alpha, beta, ejes_beta, gamma, ejes_gamma, aux, N, M, L, PROMEDIOS, GENEROS, scale )

                    combinaciones[aux_g] = suma_grupal
                suma += suma_grupal
                i += j
                cont += 1
            if len(suma.shape) == 1: suma_total = alpha * suma[0] + beta * suma[1] + gamma * suma[2]
            else: suma_total = alpha * suma[:,0] + beta * suma[:,1] + gamma * suma[:,2]
            if suma_total < best:
                print("MEJORA!", suma_total, " - en permutación #", cont, " => ", per)
                best = suma_total
                solucion = per
                dims = suma
            if suma_total > peor:
                peor_sol = per
                peor = suma_total
                peor_dims = suma
            if time.time()-tiempo >= 60:
                tiempo = time.time()
                print( "Minuto\t", mins, "\t -> permutación #", cont )
                mins += 1
        print("Mejor solución: ", solucion, " -> ", best, dims, "\n\n")
        return best, solucion, dims, peor, peor_sol, peor_dims


    for per in permutations(indices):
        print("PER: ", cont)
        i = 0
        aux = []
        suma = 0
        for j in sizes:
            aux = data[i:i+j, :]
            suma += FuncionGrupal(alpha, ejes_alpha, beta, ejes_beta, gamma, ejes_gamma, aux, N, M, L, PROMEDIOS, GENEROS )
            i += j
            cont += 1

        if suma < best:
            print("MEJORA!", suma)
            best = suma
            solucion = per
            
    return suma, solucion
    


def GroupSizes(M, L, N):
    sizes = [ M for i in range(L) ]
    total = M * L
    extra = total - N
    assert (extra >= 0), "ERROR! \nCon {} grupos y con maximo {} integrantes por grupo, sobran {} personas...".format(M, L, extra)

    if extra == 0: return sizes
    i = L-1
    for _ in range(extra):
        sizes[i] -= 1
        i -= 1
    return sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Formación de grupos colabnorativos con greedy')
    parser.add_argument('-n', '--num_personas', type=int, default=10)
    parser.add_argument('-l', '--num_grupos', type=int, default=5)
    parser.add_argument('-m', '--max_personas', type=int, default=2)
    parser.add_argument('-d', '--dataset', type=int, default=1)
    parser.add_argument('-s', '--escala', type=bool, default=True)      #"cambiar escala"  = normalizar con min max
    parser.add_argument('-e', '--exponente', type=int, default=1)      # exponente para cada dimensión de la función objetivo
    args = parser.parse_args()
    print("args: ", args)

    N, M, L, D, SCALE, exponente = args.num_personas, args.max_personas, args.num_grupos, args.dataset, args.escala, args.exponente

    #entrada = input( "Ingrese N M L Dataset scale: " )
    #entrada = entrada.split(" ")
    #print(entrada)
    #N, M, L, D, SCALE = int(entrada[0]), int(entrada[1]), int(entrada[2]), int(entrada[3]), bool(entrada[4])
    
    # Constantes
    alpha , beta, gamma = 1, 1, 1
    ejes_alpha = 4
    ejes_beta  = 1
    ejes_gamma = 4

    sizes = GroupSizes(M, L, N)
    df = pd.read_csv( "./Dataset/Dataset{}.csv".format(D), index_col=0)
    df = df.head(N)
    data = df.to_numpy()

    # Calculamos los promedios de la distribución para los ejes políticos
    promedios = []
    for i in range(ejes_alpha):
        promedios.append( np.mean(data[i,:]) )
    
    # Calculamos los porcentajes de cada género para el eje de género
    generos = []
    #print (data)
    for i in range(ejes_beta):
        grupo = data[:,i+ejes_alpha]
        f, nb, m = (grupo==0).sum()/len(grupo), (grupo==0.5).sum()/len(grupo), (grupo==1).sum()/len(grupo)
        generos = [f, nb, m]

    
    print(promedios, generos)

    print(sizes)

    start = time.time()
    valor, solucion, dimenciones, peor, peor_sol, peor_dims  = FuerzaBruta(data, alpha, ejes_alpha, beta, ejes_beta, gamma, ejes_gamma, N, M, L, promedios, generos, sizes, SCALE)
    end = time.time()
    print("\n\nValor: ", valor)
    print("Solución: ", solucion)
    print("Tiempo: ", end-start , "\n\n")
    x = []

    
    x = []
    i = 0
    for j in sizes:
        aux = solucion[i:i+j]
        grupo = []
        for k in aux:
            grupo.append( list(data[k,:]) )
        x.append(grupo)
        i += j

    x = np.array(x)
    print(x.shape)
    promedios_totales = promedios.copy()
    
    EJES  = ['economia-p', 'diplomacia-p', 'estado-p', 'sociedad-p', 'genero', 'economia-i', 'diplomacia-i', 'estado-i', 'sociedad-i']
    TITLES= ['Economía (P)', 'Diplomacia (P)', 'Estado (P)', 'Sociedad (P)', 'Género', 'Economía (IDL)', 'Diplomacia (IDL)', 'Estado (IDL)', 'Sociedad (IDL)']
    colores = ["firebrick", "darkorange", "sienna", "gold", "lightslategrey", "seagreen", "turquoise", "mediumblue", "blueviolet"]
    GENEROS = ["dimgray","silver","lightslategrey"]
    BINS = []
    e = 0
    size_grafico = (15, 15)
    grupos = x.copy()


    f, axs = plt.subplots( 3, 3, figsize=size_grafico )
    X = [ i+1 for i in range(L) ]	
    for i in range(3):
        for j in range(3):
            #print("EJE: ", EJES[e])
            if EJES[e] == "genero":
                fe, m, nb = [], [], []
                for grupo in grupos:
                    # 0		f
                    # 0.5	nb
                    # 1		m
                    #grupo = np.array(sol[l])
                    fe.append( (grupo == 0).sum() / len(grupo) )
                    nb.append( (grupo == 0.5).sum() / len(grupo) )
                    m.append( (grupo == 1).sum() / len(grupo) )
                fe = np.array(fe)
                nb = np.array(nb)
                m = np.array(m)
                axs[i][j].bar( X, fe, color=GENEROS[0], label="F" )
                axs[i][j].bar( X, nb, color=GENEROS[1], bottom=fe, label="NB" )
                axs[i][j].bar( X, m, color=GENEROS[2], bottom=nb+fe, label="M" )
            else:
                y = []
                for grupo in grupos:
                    #print("grupo! ", grupo.shape)
                    if "IDL" in TITLES[e]: y.append( np.std(grupo[:,e]) )
                    else: y.append( np.mean(grupo[:,e]) )

                axs[i][j].bar( X, y, color=colores[e] )
                if e < 4: axs[i][j].axhline( y=promedios_totales[e], ls='--', color="black", lw=3.5, label="Promedio general" )

            axs[i][j].set_title(TITLES[e], size=20)            
            axs[i][j].legend()       
            e += 1

    f.suptitle( "N={} M={} L={} E={}".format(N, M, L, exponente), fontsize=25 )
    plt.savefig("N={} M={} L={} E={}".format(N, M, L, exponente))
    #plt.show()


    """
    with open("FuerzaBruta.txt", "a") as file:
        file.write( "\n - N: {} M: {} L:{} Dataset: {} exponente: {} minmax: {}\n\tSolución:{}\n\tValor Objetivo:{}\n\tTiempo: {}\n\tPolítica: {} ; Género: {} ; Colaborativo: {}\n".format(N, M, L, D, exponente, SCALE, solucion, valor, end-start, dimenciones[0], dimenciones[1], dimenciones[2]) )
        

    with open("FuerzaBruta_PeorSolucion.txt", "a") as file: 
        file.write( "\n - N: {} M: {} L:{} Dataset: {} exponente: {} minmax: {}\n\tSolución:{}\n\tValor Objetivo:{}\n\tTiempo: {}\n\tPolítica: {} ; Género: {} ; Colaborativo: {}\n".format(N, M, L, D, exponente, SCALE, peor_sol, peor, end-start, peor_dims[0], peor_dims[1], peor_dims[2] ) )
    """