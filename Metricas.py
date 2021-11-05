import numpy as np 

def ErrorAbsolutoMedio(total, grupos):
	err = np.zeros( (len(grupos), len(total)) )
	n = len(total)

	for i in range(len(grupos)):
		grupo = grupos[i]
		for persona in grupo:
			err[i] += abs( persona - total )

	print("err: ", err.shape)



	return err / n



def ErrorCuadraticoMedio(total, grupos):
	err = []
	n = len(total)
	for grupo in grupos:
		err.append( (( total - grupo )**2)/n )
	return sum(err)




def CalcularEjes(total, grupos):	# Función auxiliar para calcular función objetivo
	suma = np.zeros( len(total) )

	for i in range(len(grupos)):
		grupo = np.array(grupos[i])
		#print("grupo ", i , " : ", grupo)
		promedios = np.array([ np.mean(grupo[:,j]) for j in range(len(total)) ])
		suma += abs( total - promedios )
		#print("\n")

	return suma