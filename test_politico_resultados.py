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

def calc_scores(score, max):
    aux =  (max+score) / (2*max)
    return round(aux, 3)


EJES = ["economia", "diplomacia", "estado", "sociedad"]
respuestas = pd.read_csv("TestPolitico.csv")
efectos = pd.read_csv("EfectosRespuestas.csv", index_col=0)

# Columnas con preguntas respuestas palta
#respuestas = list(df.columns[9:,])
resp = {
    "Totalmente de acuerdo": 1.0,
    "De acuerdo": 0.5,
    "Neutral": 0.0,
    "En desacuerdo": -0.5,
    "Totalmente en desacuerdo": -1.0,
}
suma_efectos = {
    "economia": 0,      # -> equality
    "diplomacia": 0,    # -> peace
    "estado": 0,        # -> liberty
    "sociedad": 0,      # -> progress
}
max_scores = {
    "economia" : 0,
    "diplomacia" : 0,
    "estado" : 0,
    "sociedad" : 0,
}

for i in range(70):
    mult = resp[ respuestas.iat[0, i+9] ]
    for eje in efectos.columns[1:]:
        suma_efectos[eje] += mult * efectos.at[i, eje] 
        max_scores[eje] += abs(efectos.at[i, eje])


#print("Max scores: ", max_scores)
#print("suma efectos: ", suma_efectos), "\n"

"""
- EJE ECONÓMICO:    equality    ----    wealth
- EJE DIPLOMACIA:   peace       ----    might
- EJE ESTADO:       liberty     ----    authority
- EJE SOCIEDAD:     progress    ----    tradition

ORDEN
- EJE ECONÓMICO:    igualdad    ----    mercado
- EJE DIPLOMACIA:   nacion      ----    global / peace
- EJE ESTADO:       libertad    ----    autoridad
- EJE SOCIEDAD:     tradicion   ----    progreso


"""

# Eje económico : igualdad < - > mercado
igualdad =  calc_scores( suma_efectos["economia"], max_scores["economia"] )
mercado  =  round(1 - igualdad, 3)

# Eje dipomacia : nacion < - > global
peace  = calc_scores( suma_efectos["diplomacia"], max_scores["diplomacia"] )
nacion = round(1 - peace, 3)

# Eje estado : libertad < - > autoridad
libertad = calc_scores( suma_efectos["estado"], max_scores["estado"] )
autoridad= round(1 - libertad, 3)

# Eje sociedad : tradicion < - > progreso
progreso = calc_scores( suma_efectos["sociedad"], max_scores["sociedad"] )
tradicion= round(1 - progreso, 3)

print("ECONOMÍA: \t {} \t ---- \t {}".format(igualdad, mercado) )
print("DIPLOMACIA: \t {} \t ---- \t {}".format(nacion, peace) )
print("ESTADO: \t {} \t ---- \t {}".format(libertad, autoridad) )
print("SOCIEDAD: \t {} \t ---- \t {}".format(tradicion, progreso) )