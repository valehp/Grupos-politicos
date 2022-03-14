import sys, getopt
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
import argparse

base = "./dataset/"

def shape_data(df, file, h=4):      # toma el df segmentado y lo convierte en una matriz de bools 
    n = len(df.index)
    data = []
    df_ejes = pd.DataFrame()

    for eje in df.columns:
        print(eje)
        segs = np.zeros( (n, h), dtype=int )

        for i in range(n):
            k = df[eje][i]
            segs[i][k] = 1

        for i in range(segs.shape[1]):
            df_ejes[eje+str(i)] = segs[:,i]
        #df_eje.to_csv(base + eje + "_" + file)
    
    print(df_ejes.head())
    df_ejes.to_csv(base + file)


def get_distribution(opciones_ejes, distribuciones_ejes, plots, random_ax):
    if plots: f, axs = plt.subplots( 2, 2, figsize=(15, 15) )
    i, j = 0, 0

    for k in distribuciones_ejes.keys():
        if random_ax: dist, r, l = random.choice( opciones_ejes )
        else: dist, r, l = opciones_ejes[0]
        distribuciones_ejes[k] = [dist, r, l]

        if plots:
            x = np.linspace(r, l, 1000)
            axs[i][j].plot( x, dist.pdf(x) )
            axs[i][j].set_title( k )
            i += 1
            if i == 2:
                i = 0
                j = 1

    if plots: plt.show()



def get_hist(df, ejes, plots=False):
    new_df = pd.DataFrame()
    order_df = pd.DataFrame()
    for eje in ejes:
        print(eje)
        h, b = np.histogram( df[eje], bins=n_segmentos )
        segmentos = []

        # Recorrer todos los n datos
        for i in df.index:
            # Recorrer los n_segmentos del eje actual
            for k in range(n_segmentos):
                x = df[eje][i]
                if ( k==0 and x>=b[k] and x<=b[k+1] ) or ( x>b[k] and b[k+1]>=x ):
                    segmentos.append(k)
                    break
        
        new_df[eje] = segmentos
        aux = list(zip( h, [i for i in range(len(eje))] ))
        aux.sort(reverse=True)
        order_df[eje] = [ aux[i][1] for i in range(len((aux))) ]
    return new_df
    # new_df    -> df con datos segmentados (0, 1, 2, 3)
    # order_df  -> df con el histograma


def generar(personas, distribuciones, ejes ):
    df = pd.DataFrame()
    datos = dict(zip( distribuciones.keys(), [ [] for i in range(ejes) ] ))

    for i in range(personas):
        for eje in distribuciones.keys():
          datos[eje].append( distribuciones[eje][0].rvs() )

    print(datos.keys())

    for eje in distribuciones.keys():
        df[eje] = ( datos[eje] - np.min(datos[eje]) ) / (np.max(datos[eje]) - np.min(datos[eje]))   # minmax scaler

        if eje == 'genero':
            # 0     -> F
            # 0.5   -> No binario
            # 1     -> M
            for i in range(len(df[eje])):
                if df.at[i,eje] <= 0.33: df.at[i,eje] = 0
                elif df.at[i,eje] < 0.67: df.at[i,eje] = 0.5
                else: df.at[i,eje] = 1

    return df


def GenerarDatos(personas, ejes=4, distribuciones=[]):     # Crear dataset sin venir del main
    if not distribuciones: 
        distribuciones = {
            'economia'  : [ss.norm(), -4, 4], 
            'diplomacia': [ss.norm(), -4, 4],
            'estado'    : [ss.beta(0.5, 0.5), 0, 1],
            'sociedad':  [ss.beta(3, 1.5), 0, 1],
            'genero': [ss.beta(2, 5), 0, 1],
        }

    data = generar(personas, distribuciones, ejes)
    return data


"""
distribuciones = {
    'economia'  : [ss.norm(), -4, 4], 
    'diplomacia': [ss.norm(), -4, 4], 
    'estado'    : [ss.norm(), -4, 4], 
    'sociedad':  [ss.beta(0.5, 0.5), 0, 1],
    'genero': [ss.beta(2, 5), 0, 1],
}

opciones_ejes = [
    [ss.norm(), -4, 4], 
    [ss.beta(2, 5), 0, 1],
    [ss.beta(0.5, 0.5), 0, 1],
    [ss.beta(3, 1.5), 0, 1],
]

"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generador de datos')
    parser.add_argument('-n', '--num_personas', type=int, default=10)
    args = parser.parse_args()
    print(args)

    # declarar variables importantes
    n_personas      = args.num_personas
    n_segmentos     = 4
    archivo_dataset = "Dataset{}.csv".format(args.num_personas)
    random_axis     = False     # False -> cada eje tiene distribución normal ; True -> ejes tienen distribución aleatoria (de opciones_ejes)
    plots           = False     # False -> no grafica ; True -> Grafica los datos


    ejes = ['economia', 'diplomacia', 'estado', 'sociedad', 'genero']
    opciones_ejes = [
        [ss.norm(), -4, 4], 
        [ss.beta(2, 5), 0, 1],
        [ss.beta(0.5, 0.5), 0, 1],
        [ss.beta(3, 1.5), 0, 1],
    ]
    distribuciones_ejes = {
        'economia'  : '', 
        'diplomacia': '',
        'estado'    : '',
        'sociedad':  '',
        'genero':  '',
    }


    get_distribution(opciones_ejes, distribuciones_ejes, plots, random_axis)
    df = generar(n_personas, distribuciones_ejes, ejes)
    print("df:")
    print(df.head(), "\n")

    df_segmentos = get_hist(df, ejes, plots)
    print("\nSegmentos:")
    print(df_segmentos.head(), "\n")

    print("\nShape data:")
    shape_data(df_segmentos, archivo_dataset, n_segmentos)
