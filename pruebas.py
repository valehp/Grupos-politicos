import os
from scipy.stats import norm, kurtosis
import numpy as np
import pandas as pd 
import random, time
import matplotlib.pyplot as plt
from GeneradorDataset import *


def run():
	# N, M, L, BINS
	params = [
		(20, 5, 4, 4), 
		(100, 10, 10, 5),
		(100, 20, 5, 5),
		(100, 20, 5, 10),
		(1000, 100, 10, 5),
		(10000, 200, 50, 10),
		(10000, 50, 200, 50),
	]


	D = [1, 2, 3, 4]

	for d in D:
		for N, M, L, BINS in params:
			com = "python main_greedy.py -n {} -m {} -l {} -d {} -b {}".format(N, M, L, d, BINS)
			print(com, "\n")
			os.system(com)



def run2():
	D = [1, 2, 3, 4]
	params = ["--np --k", "--p --nk"]
	for p in params:
		for d in D:
			com = "python main_greedy.py -n 1000 -m 50 -l 20 -b 10 --T --NG {} -d {}".format(p, d)
			print(com, "\n")
			os.system(com)


"""
- normal:   [ss.norm(), -4, 4], 
- beta-U:   [ss.beta(0.5, 0.5), 0, 1],
- beta-Der: [ss.beta(3, 1.5), 0, 1],
- bera-izq: [ss.beta(2, 5), 0, 1],
"""
def dataset():
	N = 100000
	distribuciones = {
		'economia'  : 	[ss.norm(), -4, 4], 
		'diplomacia':	[ss.norm(), -4, 4],
		'estado'    :	[ss.norm(), -4, 4], 
		'sociedad':  	[ss.norm(), -4, 4], 
		'genero': 		[ss.norm(), -4, 4], 
	}
	datos = GenerarDatos(N, len(distribuciones.keys()), distribuciones)
	datos.to_csv("./dataset/5ejes_100k_normales.csv")
	print(datos.head())


def run3():
	tipo_greedy = ["normal", "random", "iterativo"]
	tipo_cambio = ["+", "-"]
	command = "python main_greedy.py -n 1000 -l 50 -m 20 -b 10 --g -d 4 "

	for greedy in tipo_greedy:
		aux = "{} -tipog {} --ncambio".format(command, greedy)
		print(aux)
		os.system(aux)

		for cambio in tipo_cambio:
			aux = "{} -tipog {} --cambio -tipoc {}".format(command, greedy, cambio)
			print(aux)
			os.system(aux)


def run4():
	tipo_greedy = ["normal", "random", "iterativo"]
	dataset = [2]
	exp = [1, 2, 3, 4]
	command = "python main_greedy.py -n 30 -l 6 -m 5 -b 10 --g "


	for greedy in tipo_greedy:
		for d in dataset:
			for e in exp:
				aux = "{} -tipog {} -d {} -e {}".format(command, greedy, d, e)
				print(aux)
				os.system(aux)




def save_data():
	pass

run4()