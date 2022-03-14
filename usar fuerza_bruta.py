import os
import numpy as np
import pandas as pd 
import random, time
import matplotlib.pyplot as plt
from GeneradorDataset import *


def run():
    dataset = 2
    minmax = ["", True]
    datos = [
        [4, 2, 2],
        [6, 2, 3],
        [6, 3, 2],
        [8, 2, 4],
        [8, 4, 2],
        [10, 2, 5],
        [10, 5, 2],
    ]
    exponente = [1, 2, 3, 4]

    for m in minmax:
        #print("MINMAX: ",m)
        for exp in exponente:
            for N, M, L in datos:
                if m: command = "python FuerzaBruta.py -n {} -m {} -l {} -d {} --escala {} -e {}".format(N, M, L, dataset, m, exp)
                else: command = "python FuerzaBruta.py -n {} -m {} -l {} -d {} -e {}".format(N, M, L, dataset, exp)
                print(command, "\n")
                os.system( command )
            with open("FuerzaBruta.txt", "a") as file:
                file.write( "\n" + "="*100 + "\n" )


if __name__ == "__main__":
    run()