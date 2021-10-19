import numpy as np
import scipy.stats as ss
from itertools import compress 


class BaseGreedy:
	def __init__(self, data, M, L, num_ejes, objetivos=np.array([]), tol=0.05):
		self.data = data 										# Datos de las personas (4 numeros reales por persona)
		self.M = M 												# Cantidad máxima de personas por grupo
		self.L = L 												# Número de grupos
		self.ejes = num_ejes
		self.tol = tol 											# Tolerancia de cada grupo con respecto al promedio por eje
		self.ObjGenerales = objetivos 							# Valor objetivo que debe alcanzar cada grupo
		self.sizes = self.GroupSizes(M, L, len(data))			# Tamaño que le corresponde a cada grupo
		self.X = np.zeros( (len(data), L), dtype=bool )			# Variable X (matriz) donde se guardará la pertenencia del dato i al grupo j

		self.grupos = [ [] for i in range(L) ] 					# Aquí se guardan los grupos
		self.listos = np.array([ False for i in range(L) ]) 	# Lista de bools que indica si los grupos están cerrados (completos) o no
		self.obj_grupales = [ np.array([]) for i in range(L) ] 	# Promedio de cada grupo en cada uno de sus ejes (4 ejes)


	# ------- Funciones para grupos ------- #
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


	def CheckGroup(self, index):
		if len(self.grupos[index]) >= self.sizes[index]:
			self.listos[index] = True
			return True
		return False


	def UpdateGroup(self, index_grupo, index_data, new_obj):
		if len(self.grupos[index_grupo]) == 0: self.grupos[index_grupo] = [ self.data[index_data].tolist().copy() ]
		else: 
			grupo = self.grupos[index_grupo].copy()
			grupo.append( self.data[index_data].tolist().copy() )
			self.grupos[index_grupo] = grupo.copy()

		self.obj_grupales[index_grupo] = new_obj
		self.X[index_data, index_grupo] = 1
		self.CheckGroup(index_grupo)


	# --------- Usar algoritmo greedy --------- #
	def Greedy(self):

		for i in range(len(self.data)):
			mejor_obj = np.ones(self.ejes)		# Guarda los promedios en todos los grupos, en caso que no cumpla la tolerancia
			indice_grupal = 0					# Guarda el índice al mejor grupo para agregar el dato i
			i_listo = False 					# Indica si el dato i se ingresó a un grupo o no

			for j in list(compress( list(range(self.L)), ~self.listos )):

				if len(self.grupos[j]) == 0:
					self.UpdateGroup(j, i, self.data[i])
					i_listo = True
					break

				aux = self.grupos[j].copy()
				aux.append( self.data[i].tolist() )

				nuevo_obj = self.CalcularObjetivo( np.array(aux, dtype=object) )

				if (nuevo_obj >= (self.ObjGenerales - self.tol)).all() and (nuevo_obj <= (self.ObjGenerales + self.tol)).all():
					self.UpdateGroup(j, i, nuevo_obj)
					i_listo = True
					break

				dif = abs( abs(self.ObjGenerales) - abs(nuevo_obj) )
				if (dif < mejor_obj).all():
					mejor_obj = nuevo_obj
					indice_grupal = j
					

			if not i_listo:
				self.UpdateGroup( indice_grupal, i, mejor_obj )

		return self.grupos, self.X


	# ------ Imprimir datos ------ #
	def print_proms(self, file=""):
		if not file:
			for i in range(self.L):
				print ("GRUPO {}: {}".format(i, self.obj_grupales[i]))
			print("\n")
		if file:
			with open (file, 'w') as f:
				f.write("VALOR OBJETIVO: {}\n\n".format(self.ObjGenerales))
				for i in range(self.L):
					f.write( "GRUPO {}: {}\n".format(i+1, self.obj_grupales[i]) )


	def Mejor_Peor(self, file=""): 							# Retorna el mejor y el peor grupo encontrado
		self.print_proms(file)
		mejor, peor = float('inf'), -float('inf')
		im, ip = 0, 0
		for i in range(self.L):
			dif = abs(self.obj_grupales[i] - self.ObjGenerales)
			if (dif <= mejor).all(): mejor, im = dif, i
			if (dif >= peor).all(): peor, ip = dif, i

		with open(file, "a") as f:
			f.write( "\nMEJOR -> Grupo {} \n\t {} -> dif: {}  \n".format(im+1, self.obj_grupales[im], mejor) )
			f.write( "\nPEOR  -> Grupo {} \n\t {} -> dif: {}  \n".format(ip+1, self.obj_grupales[ip], peor) )

		return (mejor, im+1), (peor, ip+1)





# ----------------------------------------------------------- #
# --- Greedy con promedio como valor objetivo en cada eje --- #
# ----------------------------------------------------------- #

class PromedioGreedy(BaseGreedy):
	def __init__(self, data, M, L, num_ejes, objetivos=np.array([]), tol=0.05):
		BaseGreedy.__init__(self, data, M, L, num_ejes, objetivos, tol=0.05)
		if not objetivos.all() or len(objetivos) < num_ejes:
			self.ObjGenerales = self.PromedioEjes(self.data)


	def PromedioEjes(self, data):
		promedios = []
		if len(data.shape) == 1: return data
		for i in range(self.ejes):
			promedios.append( np.mean( data[:,i] ) )

		return np.array(promedios)


	def Usar(self):
		self.CalcularObjetivo = self.PromedioEjes
		return BaseGreedy.Greedy(self)






# ------------------------------------------------------ #
# --- Greedy con Kurtosis por eje en vez de promedio --- #
# ------------------------------------------------------ #

class GreedyKurtosis(BaseGreedy):
	def __init__(self, data, M, L, num_ejes, tol=0.05):
		BaseGreedy.__init__(self, data, M, L, tol=0.05)
		self.kurtosis_grupos = [ np.array([]) for i in range(L) ]
		self.KurtosisGeneral = self.KurtosisEjes(data)

	def KurtosisEjes(self, data):
		k = []
		if len(data.shape) == 1: return data
		for i in range(self.ejes):
			k.append( ss.kurtosis(data[:,i]) )

		return np.array(k)

	def UpdateGroup(self, index_grupo, index_data, kurt):
		if len(self.grupos[index_grupo]) == 0: self.grupos[index_grupo] = [ self.data[index_data].tolist().copy() ]
		else: 
			grupo = self.grupos[index_grupo].copy()
			grupo.append( self.data[index_data].tolist().copy() )
			self.grupos[index_grupo] = grupo.copy()

		self.kurtosis_grupos[index_grupo] = kurt
		self.X[index_data, index_grupo] = 1
		self.CheckGroup(index_grupo)


	def Usar(self):

		for i in range(len(self.data)):
			menor_k = 10*np.ones(self.ejes)		# Guarda los promedios en todos los grupos, en caso que no cumpla la tolerancia
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

				nuevo_k = self.KurtosisEjes( np.array(aux, dtype=object) )

				if (nuevo_k >= (self.KurtosisGeneral - self.tol)).all() and (nuevo_k <= (self.KurtosisGeneral - self.tol)).all():
					self.UpdateGroup(j, i, nuevo_k)
					i_listo = True
					break

				dif = abs( abs(self.KurtosisGeneral) - abs(nuevo_k) )
				if (dif < menor_k).all():
					menor_k = dif
					indice_grupal = j
					

			if not i_listo:
				self.UpdateGroup( indice_grupal, i, menor_k )

		return self.grupos, self.X





# ------------------------------------------------------ #
# ----------- "Epsilon-Greedy" con promedio ------------ #
# ------------------------------------------------------ #

class RandomGreedy(PromedioGreedy):
	def __init__(self, data, M, L, num_ejes, objetivos=np.array([]), tol=0.05, r=0.8, n=5):
		BaseGreedy.__init__(self, data, M, L, num_ejes, objetivos, tol=0.05)
		self.r = r
		self.n = n


	def Usar(self):
		for i in range(len(self.data)):
			mejor_obj = np.ones(self.ejes)		# Guarda los promedios en todos los grupos, en caso que no cumpla la tolerancia
			indice_grupal = 0					# Guarda el índice al mejor grupo para agregar el dato i
			i_listo = False 					# Indica si el dato i se ingresó a un grupo o no
			lista = [] 							# Guarda los n mejores grupos para añadir el dato i


			for j in list(compress( list(range(self.L)), ~self.listos )):

				if len(self.grupos[j]) == 0:
					self.UpdateGroup(j, i, self.data[i])
					i_listo = True
					break

				aux = self.grupos[j].copy()
				aux.append( self.data[i].tolist() )

				nuevo_obj = self.CalcularObjetivo( np.array(aux, dtype=object) )

				if (nuevo_obj >= (self.ObjGenerales - self.tol)).all() and (nuevo_obj <= (self.ObjGenerales + self.tol)).all():
					self.UpdateGroup(j, i, nuevo_obj)
					i_listo = True
					break

				dif = abs( abs(self.ObjGenerales) - abs(nuevo_obj) )
				if (dif < mejor_obj).all():
					mejor_obj = nuevo_obj
					indice_grupal = j
					

			if not i_listo:
				self.UpdateGroup( indice_grupal, i, mejor_obj )

		return self.grupos, self.X
