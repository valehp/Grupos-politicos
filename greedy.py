from hashlib import new
import numpy as np
import scipy.stats as ss
from itertools import compress 
import random
#from Metricas import *


class BaseGreedy:
	def __init__(self, data, M, L, num_ejes, objetivos=np.array([]), tol=0.025, alpha=1, ejes_alpha=4, beta=1, ejes_beta=1, gamma=1, ejes_gamma=4, generos=[0, 0.5, 1], exponente=1, minmax=True):
		self.data = data 										# Datos de las personas (4 numeros reales por persona)
		self.M = M 												# Cantidad máxima de personas por grupo
		self.L = L 												# Número de grupos
		self.ejes = num_ejes
		self.tol = tol 											# Tolerancia de cada grupo con respecto al promedio por eje
		self.ObjGenerales = objetivos 							# Valor objetivo que debe alcanzar cada grupo
		self.sizes = self.GroupSizes(M, L, len(data))			# Tamaño que le corresponde a cada grupo
		self.X = np.zeros( (len(data), L), dtype=bool )			# Variable X (matriz) donde se guardará la pertenencia del dato i al grupo j
		self.generos = generos
		self.exp = exponente
		self.mm = minmax

		self.grupos = [ [] for i in range(L) ] 					# Aquí se guardan los grupos
		self.listos = np.array([ False for i in range(L) ]) 	# Lista de bools que indica si los grupos están cerrados (completos) o no
		self.obj_grupales = [ np.array([]) for i in range(L) ] 	# Promedio de cada grupo en cada uno de sus ejes (4 ejes)

		self.alpha = alpha
		self.beta  = beta
		self.gamma = gamma
		self.ejes_alpha = ejes_alpha
		self.ejes_beta  = ejes_beta
		self.ejes_gamma = ejes_gamma 
		self.Objetivo, self.dimensiones = None, None
		self.Objetivo_ejes = [ 0 for i in range(self.ejes) ]
		self.MejorMetrica  = ""

		self.min_p, self.max_p = 0, 4
		self.min_g, self.max_g = 0, 2
		self.min_c, self.max_c = 0, 4


	def reset(self):
		self.X = np.zeros( (len(self.data), self.L), dtype=bool )
		self.grupos = [ [] for i in range(self.L) ] 
		self.listos = np.array([ False for i in range(self.L) ])
		self.obj_grupales = [ np.array([]) for i in range(self.L) ]


	def FuncionObjetivo(self):
		# Componente política
		P = 0
		for j in range(self.L):
			grupo = np.array(self.grupos[j])
			for i in range(self.ejes_alpha):
				if len(grupo.shape) > 1: promedios = np.mean( grupo[:,i] )
				else: promedios = grupo[i]
				dif = abs( self.ObjGenerales[i] - promedios ) ** self.exp
				P += np.sum(dif)


		# Componente de género
		G = 0
		for j in range(self.L):
			grupo = np.array(self.grupos[j])[:, self.ejes_alpha]
			f  = 0 if ((grupo==0).sum() == 0 ) else (grupo==0).sum()/len(grupo)
			nb = 0 if ((grupo==0.5).sum() == 0 ) else (grupo==0.5).sum()/len(grupo)
			m  = 0 if ((grupo==1).sum() == 0 ) else (grupo==1).sum()/len(grupo)
			#f, nb, m = (grupo==0).sum()/len(grupo), (grupo==0.5).sum()/len(grupo), (grupo==1).sum()/len(grupo)
			generos = np.array([f, nb, m])
			dif = abs( self.ObjGenerales[self.ejes_alpha] - generos ) ** self.exp
			G += np.sum(dif)


		# Componente de grupos colaborativos
		C = 0
		for j in range(self.L):
			grupo = np.array(self.grupos[j])
			for i in range(self.ejes_alpha+self.ejes_beta, self.ejes_alpha+self.ejes_beta+self.ejes_gamma, 1):
				std = np.std( grupo[:,i] ) ** self.exp
				C += np.sum(std)

		if self.mm:
			P = (P - self.min_p) / (self.max_p - self.min_p)
			G = (G - self.min_g) / (self.max_g - self.min_g)
			C = (C - self.min_c) / (self.max_c - self.min_c)

		self.Objetivo = self.alpha * P + self.beta * G + self.gamma * C
		return self.Objetivo, (P, G, C)


	def FruncionObjetivoGrupo(self, grupo, index_group):
		P = 0
		grupo = np.array(grupo)
		for i in range(self.ejes_alpha):
			promedios = np.mean(grupo[:,i])
			dif = abs( self.ObjGenerales[i] - promedios ) ** self.exp
			P += np.sum(dif)

		G = 0
		grupo_g = grupo[:, self.ejes_alpha]
		f  = 0 if ((grupo_g==0).sum() == 0 ) else (grupo_g==0).sum()/len(grupo_g)
		nb = 0 if ((grupo_g==0.5).sum() == 0 ) else (grupo_g==0.5).sum()/len(grupo_g)
		m  = 0 if ((grupo_g==1).sum() == 0 ) else (grupo_g==1).sum()/len(grupo_g)
		generos = np.array([f, nb, m])
		dif = abs( self.ObjGenerales[self.ejes_alpha] - generos ) ** self.exp
		G += np.sum(dif)

		C = 0
		for i in range(self.ejes_alpha+self.ejes_beta, self.ejes_alpha+self.ejes_beta+self.ejes_gamma, 1):
			std = np.std( grupo[:,i] ) ** self.exp
			C += np.sum(std)

		if self.mm:
			P = (P - self.min_p) / (self.max_p - self.min_p)
			G = (G - self.min_g) / (self.max_g - self.min_g)
			C = (C - self.min_c) / (self.max_c - self.min_c)

		self.obj_grupales[index_group] = self.alpha * P + self.beta * G + self.gamma * C


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
			self.FruncionObjetivoGrupo(self.grupos[index], index)
			return True
		return False


	def UpdateGroup(self, index_grupo, index_data, new_obj=-1, p=0, g=0, c=0):
		if type(new_obj) == int:
			new_obj = np.append( p, g )
			new_obj = np.append( new_obj, c )

		if len(self.grupos[index_grupo]) == 0: self.grupos[index_grupo] = [ self.data[index_data].tolist().copy() ]
		else: 
			grupo = self.grupos[index_grupo].copy()
			grupo.append( self.data[index_data].tolist().copy() )
			self.grupos[index_grupo] = grupo.copy()

		self.obj_grupales[index_grupo] = new_obj
		self.X[index_data, index_grupo] = 1
		self.CheckGroup(index_grupo)


	def MENOR(self, primero, segundo):
		# Devuelve (primero < segundo).all()
		if (primero[:self.ejes_alpha] < segundo[:self.ejes_alpha]).all() and (primero[self.ejes_alpha] < segundo[self.ejes_alpha]).all() and (primero[self.ejes_alpha+1:] < segundo[self.ejes_alpha+1:]).all() :
			return True
		return False


	def diff(self, grupo):		
		# calcula que tanto se acerca el GRUPO al objetivo
		dif_p = np.zeros(self.ejes_alpha)
		dif_g = np.zeros(len(self.generos))
		dif_c = np.zeros(self.ejes_gamma)

		# Componente política
		for i in range(self.ejes_alpha):
			if len(grupo.shape) != 2: promedio = grupo[i]
			else: promedio = np.mean( grupo[:,i] )
			dif_p[i] = abs( promedio - self.ObjGenerales[i] ) ** self.exp
		
		# Componente de Género
		G = 0
		f  = 0 if ((grupo[:,self.ejes_alpha]==0).sum() == 0 ) else (grupo==0).sum()/len(grupo)
		nb = 0 if ((grupo[:,self.ejes_alpha]==0.5).sum() == 0 ) else (grupo==0.5).sum()/len(grupo)
		m  = 0 if ((grupo[:,self.ejes_alpha]==1).sum() == 0 ) else (grupo==1).sum()/len(grupo)
		generos = np.array([f, nb, m])
		dif_g = abs( self.generos - generos ) ** self.exp

		# Componente colaborativa
		e = 0
		for i in range( self.ejes - self.ejes_alpha+self.ejes_beta, self.ejes, 1 ):
			if len(grupo.shape) != 2: promedio = grupo[i]
			else: promedio = np.mean( grupo[:,i] )
			dif_c[e] = abs( promedio - self.ObjGenerales[i] )
			e += 1

		#if self.mm:
		#	P = (P - self.min_p) / (self.max_p - self.min_p)
		#	G = (G - self.min_g) / (self.max_g - self.min_g)
		#	C = (C - self.min_c) / (self.max_c - self.min_c)

		return dif_p, dif_g, dif_c



	def check_new(self, p, g, c):
		# Verifica que la resta en cada eje esté dentro de la tolerancia 
		# si está dentro => se agrega al nuevo grupo (return True)
		tol_p = np.ones( (self.ejes_alpha) ) * self.tol
		tol_g = np.ones( (len(self.generos)) ) * self.tol
		tol_c = np.ones( (self.ejes_gamma) ) * self.tol

		# Check componente política, en caso de que no esté dentro de la tolerancia
		if (p <= tol_p).any() and (p >= tol_p).any(): return False

		# Check componente género
		if (g <= tol_g).any() and (g >= tol_g).any(): return False

		# Check componente colaborativa
		if (c <= tol_c).any() and (c >= tol_c).any(): return False

		return True



	# --------- Usar algoritmo greedy --------- #
	def Greedy(self):

		for i in range(len(self.data)):
			#  Se guarda los promedios en todos los grupos, en caso que no cumpla la tolerancia
			mejor_p , mejor_g , mejor_c = -1, -1, -1

			indice_grupal = 0					# Guarda el índice al mejor grupo para agregar el dato i
			i_listo = False 					# Indica si el dato i se ingresó a un grupo o no

			for j in list(compress( list(range(self.L)), ~self.listos )):

				if len(self.grupos[j]) == 0:
					self.UpdateGroup(j, i, new_obj=self.data[i])
					i_listo = True
					break

				aux = self.grupos[j].copy()
				aux.append( self.data[i].tolist() )

				#nuevo_obj = self.CalcularObjetivo( np.array(aux, dtype=object) )
				nuevo_obj = 0
				p, g, c = self.diff( np.array(aux, dtype=float) )

				if self.check_new(p, g, c):
					self.UpdateGroup(j, i, p=p, g=g, c=c)
					i_listo = True
					break


				#if (nuevo_obj >= (self.ObjGenerales - self.tol)).all() and (nuevo_obj <= (self.ObjGenerales + self.tol)).all():
				if type(mejor_p) == int : 
					mejor_p , mejor_g , mejor_c = p, g, c
					indice_grupal = j
				elif (p < mejor_p).all() and (g< mejor_g).all() and (c < mejor_c).all() :
					mejor_p , mejor_g , mejor_c = p, g, c
					indice_grupal = j
					

			if not i_listo:
				self.UpdateGroup( indice_grupal, i, p=p, g=g, c=c )

		return self.grupos, self.X




# ----------------------------------------------------------- #
# --- Greedy con promedio como valor objetivo en cada eje --- #
# ----------------------------------------------------------- #

class PromedioGreedy(BaseGreedy):
	def __init__(self, data, M, L, num_ejes, objetivos=np.array([]), tol=0.05, exponente=1):
		BaseGreedy.__init__(self, data, M, L, num_ejes, objetivos, tol=0.05, exponente=exponente)
		#if not objetivos.all() or len(objetivos) < num_ejes:
		#	print("CAMBIANDO PROMEDIOS")
		#	self.ObjGenerales = self.PromedioEjes(self.data)


	def PromedioEjes(self, data):
		promedios = []
		if len(data.shape) == 1: return data
		for i in range(self.ejes_alpha):
			promedios.append( np.mean( data[:,i] ) )

		g = data[:,self.ejes_alpha]
		N = len(g)
		aux = []
		for i in self.generos:
			aux.append( 0 if ((g==i).sum()==0) else (g==i).sum()/N ) 
		promedios.append( np.array(aux) )

		for i in range(self.ejes_alpha + self.ejes_beta, self.ejes, 1 ):
			promedios.append( np.std(data[:,i]) )

		return np.array(promedios, dtype=float)


	def Usar(self):
		self.CalcularObjetivo = self.PromedioEjes
		BaseGreedy.Greedy(self)
		self.Objetivo, dims = self.FuncionObjetivo()
		return self.grupos, self.X, self.Objetivo, dims






# ------------------------------------------------------ #
# --- Greedy con Kurtosis por eje en vez de promedio --- #
# ------------------------------------------------------ #

class GreedyKurtosis(BaseGreedy):
	def __init__(self, data, M, L, num_ejes, tol=0.05, exponente=1):
		BaseGreedy.__init__(self, data, M, L, tol=0.05, exponente=exponente)
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

		self.Objetivo, self.dimensiones = self.FuncionObjetivo()

		return self.grupos, self.X





# ------------------------------------------------------ #
# ----------- "Epsilon-Greedy" con promedio ------------ #
# ------------------------------------------------------ #

class RandomGreedy(PromedioGreedy):
	def __init__(self, data, M, L, num_ejes, objetivos=np.array([]), tol=0.05, r=0.2, n=5, exponente=1):
		# r: porcentaje de aleatoreidad
		PromedioGreedy.__init__(self, data, M, L, num_ejes, objetivos, tol, exponente)
		self.r = r
		self.n = n


	def Usar(self, data=np.array([])):
		if not data.size: data = self.data.copy()
		self.CalcularObjetivo = self.PromedioEjes
		for i in range(len(data)):				# Se itera sobre todas las personas
			mejor_p, mejor_g, mejor_c = 0, 0, 0	# Guarda los promedios en todos los grupos, en caso que no cumpla la tolerancia
			indice_grupal = 0					# Guarda el índice al mejor grupo para agregar el dato i
			i_listo = False 					# Indica si el dato i se ingresó a un grupo o no
			lista = [] 							# Guarda los n mejores grupos para añadir el dato i

			for j in list(compress( list(range(self.L)), ~self.listos )):		# Se itera sobre todos los grupos que no estén llenos

				if len(self.grupos[j]) == 0:
					self.UpdateGroup(j, i, self.data[i])
					i_listo = True
					break

				aux = self.grupos[j].copy()
				aux.append( data[i].tolist() )

				p, g, c = self.diff( np.array(aux, dtype=float) )

				new_obj = np.append( p,g  )
				new_obj = np.append( new_obj, c )

				if len(lista) < self.n:
					lista.append( (new_obj, j) )
				else:
					lista.append( (new_obj, j) )
					lista.sort(key=lambda a: a[0].all())
					lista = lista[:self.n]

				if type(mejor_p) == int : 
					mejor_p , mejor_g , mejor_c = p, g, c
					indice_grupal = j
				elif (p < mejor_p).all() and (g< mejor_g).all() and (c < mejor_c).all() :
					mejor_p , mejor_g , mejor_c = p, g, c
					indice_grupal = j

			if i_listo: continue	
			numero_random = random.random()
			if numero_random < self.r:
				nuevo_grupo = random.choice(lista)
				self.UpdateGroup( nuevo_grupo[1], i, nuevo_grupo[0] )
			else: self.UpdateGroup( indice_grupal, i, p=mejor_p, g=mejor_g, c=mejor_c )

		self.Objetivo, self.dimensiones = self.FuncionObjetivo()

		return self.grupos, self.X, self.Objetivo, self.dimensiones



# --------------------------------------------- #
# --- Greedy semi-aleatorio con iteraciones --- #
# --------------------------------------------- #

class IteratedRandomGreedy (RandomGreedy):			
	def __init__(self, data, M, L, num_ejes, objetivos=np.array([]), tol=0.05, r=0.05, n=5, it=20, exponente=1):
		# r: porcentaje de aleatoreidad
		# n: número de mejores grupos que guarda para hacer el cambio aleatorio
		RandomGreedy.__init__(self, data, M, L, num_ejes, objetivos, tol, r, n, exponente)
		self.MejorSolucion = np.array([])				# Guarda el mejor X obtenido
		self.MejoresGrupos = np.array([])				# Guarda la solucion en forma de datos (para los gráficos)
		self.MejorMetrica  = float('inf') 
		self.it 		   = it


	def reset(self):
		self.X = np.zeros( (len(self.data), self.L), dtype=bool )
		self.grupos = [ [] for i in range(self.L) ] 
		self.listos = np.array([ False for i in range(self.L) ])
		self.obj_grupales = [ np.array([]) for i in range(self.L) ]


	def ActualizarMetrica(self, actual, grupos, X, dims):
		if actual < self.MejorMetrica:
			self.MejorMetrica  = actual
			self.MejoresGrupos = grupos
			self.MejorSolucion = X
			self.dimensiones   = dims
			return True
		return False


	def Usar(self):
		aux_obj_grupales = []

		for it in range(self.it):
			self.reset()
			grupos, X, _, _ = RandomGreedy.Usar(self)
			metrica, dims_metrica   = self.FuncionObjetivo()
			if self.ActualizarMetrica( metrica, grupos, X, dims_metrica ): aux_obj_grupales = self.obj_grupales

		self.obj_grupales = aux_obj_grupales
		self.Objetivo = self.MejorMetrica

		return self.MejoresGrupos, self.MejorSolucion, self.Objetivo, self.dimensiones





# ------------------------------------------------------------------------------------------------ #
# --- Greedy iterativo semi-aleatorio con destrucción parcial de la solución en cada iteración --- #
# ------------------------------------------------------------------------------------------------ #

class IteratedDestructiveGreedy (RandomGreedy):			
	def __init__(self, data, M, L, num_ejes, objetivos=np.array([]), tol=0.05, r=0.05, n=5, it=20, D=2, exponente=1):
		# r: porcentaje de aleatoreidad
		# n: número de mejores grupos que guarda para hacer el cambio aleatorio
		# D: número de grupos a destruir
		assert (D <= L), "ERROR!\nNO se pueden destruir {} grupos con {} grupos totales ... ".format(D, L)

		RandomGreedy.__init__(self, data, M, L, num_ejes, objetivos, tol, r, n, exponente)
		self.MejorSolucion = np.array([])				# Guarda el mejor X obtenido
		self.MejoresGrupos = np.array([])				# Guarda la solucion en forma de datos (para los gráficos)
		self.MejorMetrica  = float('inf') 
		self.it 		   = it							# Cantidad de repeticiones del algoritmo
		self.D 			   = D


	def reset(self):
		self.X = np.zeros( (len(self.data), self.L), dtype=bool )
		self.grupos = [ [] for i in range(self.L) ] 
		self.listos = np.array([ False for i in range(self.L) ])
		self.obj_grupales = [ np.array([]) for i in range(self.L) ]


	def DestruirGrupos(self):
		contador = 0			# Retorna la cantidad de personas que se quedan sin grupo
		index_del = []			# índices de los datos eliminados
		aux =  list( zip( self.obj_grupales, list(range(self.L)) ) )
		aux.sort(reverse=True)
		for i in range(self.D):
			j = aux[i][1]	# índice del grupo a eliminar
			contador += len(self.grupos[j])
			index_del += list(compress( list(range(len(self.data))), self.X[:,j] ) )
			self.X[:,j] = np.zeros(len(self.data)).copy()
			self.listos[j] = False
			self.grupos[j] = []

		"""
		for i in range( self.L-1, self.L-1-self.D, -1 ):	

			contador += len(self.grupos[i])
			self.listos[i] 	= False
			self.X[:,i] 	= np.zeros( len(self.data) ).copy()
			self.grupos[i] 	= []
		"""
		data = []
		for i in index_del:
			data.append( self.data[i,:] )
		
		data = np.array(data)

		return data


	def ActualizarMetrica(self, actual, grupos, X, dims):
		if actual < self.MejorMetrica:
			self.MejorMetrica  = actual
			self.MejoresGrupos = grupos
			self.MejorSolucion = X
			self.dimensiones   = dims
			return True
		return False


	def Usar(self):
		aux_obj_grupales = []

		for it in range(self.it):
			self.reset()
			grupos, X, _, _ = RandomGreedy.Usar(self)
			metrica, dims= self.FuncionObjetivo()
			if self.ActualizarMetrica( metrica, grupos, X, dims ): aux_obj_grupales = self.obj_grupales

			data = self.DestruirGrupos()
			grupos, X, obj, dims = RandomGreedy.Usar(self, data)
			if self.ActualizarMetrica( obj, grupos, X, dims ): aux_obj_grupales = self.obj_grupales



		self.obj_grupales = aux_obj_grupales
		self.Objetivo = self.MejorMetrica

		return self.MejoresGrupos, self.MejorSolucion, self.Objetivo, self.dimensiones