import numpy as np 

def ErrorAbsolutoMedio(total, grupos):
	err = []
	n = len(total)
	for grupo in grupos:
		err.append( np.abs( total - grupo )/n ) 
	return err


def ErrorCuadraticoMedio(total, grupos):
	err = []
	n = len(total)
	for grupo in grupos:
		err.append( (( total - grupo )**2)/n )
	return err


