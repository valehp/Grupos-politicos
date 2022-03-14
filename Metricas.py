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



def FuncionObjetivo(alpha, ejes_alpha, beta, ejes_beta, gamma, ejes_gamma, grupos, N, M, L, ObjGenerales ):
	# Componente política
	P = 0
	for j in range(L):
		grupo = np.array(grupos[j])
		promedios = np.mean( grupo[:ejes_alpha] )
		dif = abs( ObjGenerales[:ejes_alpha] - promedios )
		P += np.sum(dif)


	# Componente de género
	G = 0
	for j in range(L):
		grupo = np.array(grupos[j])[:, ejes_alpha]
		f, nb, m = (grupo==0).sum()/len(grupo), (grupo==0.5).sum()/len(grupo), (grupo==1).sum()/len(grupo)
		generos = np.array([f, nb, m])
		dif = abs( ObjGenerales[ejes_alpha] - generos )
		G += np.sum(dif)


	# Componente de grupos colaborativos
	C = 0
	for j in range(L):
		grupo = np.array(grupos[j])
		promedios = np.mean( grupo[ejes_alpha+ejes_beta:] )
		dif = abs( ObjGenerales[ejes_alpha+ejes_beta:] - promedios )
		C += np.sum(dif)

	Objetivo = alpha * P + beta * G + gamma * C
	return Objetivo