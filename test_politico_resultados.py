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

df = pd.read_csv("TestPolitico.csv")