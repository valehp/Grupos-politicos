import numpy as np
from itertools import compress 


class Greedy:
	def __init__(self, data, M, L, tol=5e-1):
		self.data = data 										# Datos de las personas (4 numeros reales por persona)
		self.M = M 												# Cantidad máxima de personas por grupo
		self.L = L 												# Número de grupos
		self.tol = tol 											# Tolerancia de cada grupo con respecto al promedio por eje
		self.grupos = [ [] for i in range(L) ] 					# Aquí se guardan los grupos
		self.listos = np.array([ False for i in range(L) ]) 	# Lista de bools que indica si los grupos están cerrados (completos) o no
		self.prom_grupos = [ np.array([]) for i in range(L) ] 	# Promedio de cada grupo en cada uno de sus ejes (4 ejes)
		self.promediosGenerales = self.PromedioEjes(self.data) 	# Promedio general de todos los datos por cada eje 
		self.sizes = self.GroupSizes(M, L, len(data))			# Tamaño que le corresponde a cada grupo


	def GroupSizes(self, M, L, N):
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


	def PromedioEjes(self, data):
		promedios = []
		if len(data.shape) == 1: return data
		for i in range(4):
			promedios.append( np.mean( data[:,i] ) )

		return np.array(promedios)


	def CheckGroup(self, index):
		if len(self.grupos[index]) >= self.sizes[index]:
			self.listos[index] = True
			return True
		return False


	def UpdateGroup(self, index_grupo, index_data, promedios):
		if len(self.grupos[index_grupo]) == 0: self.grupos[index_grupo] = [ self.data[index_data].tolist().copy() ]
		else: 
			grupo = self.grupos[index_grupo].copy()
			grupo.append( self.data[index_data].tolist().copy() )
			self.grupos[index_grupo] = grupo.copy()

		self.prom_grupos[index_grupo] = promedios
		self.CheckGroup(index_grupo)


	def Usar(self):

		for i in range(len(self.data)):
			menor_promedio = np.ones(4)		# Guarda los promedios en todos los grupos, en caso que no cumpla la tolerancia
			indice_grupal = 0
			i_listo = False 					# Indica si el dato i se ingresó a un grupo o no

			for j in list(compress( list(range(self.L)), ~self.listos )):

				if len(self.grupos[j]) == 0:
					#nuevo_promedio = self.PromedioEjes( self.data[i] )
					self.UpdateGroup(j, i, self.data[i])
					i_listo = True
					break

				aux = self.grupos[j].copy()
				aux.append( self.data[i].tolist() )

				nuevo_promedio = self.PromedioEjes( np.array(aux, dtype=object) )

				if (nuevo_promedio >= (self.promediosGenerales - self.tol)).all() and (nuevo_promedio <= (self.promediosGenerales - self.tol)).all():
					self.UpdateGroup(j, i, nuevo_promedio)
					i_listo = True
					break

				dif = self.promediosGenerales - nuevo_promedio
				if (dif < menor_promedio).all():
					menor_promedio = dif
					indice_grupal = j
					

			if not i_listo:
				self.UpdateGroup( indice_grupal, i, menor_promedio )

		return self.grupos