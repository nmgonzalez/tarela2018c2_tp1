
import math
import numpy as np
from numpy import linalg as LA

# CONSTANTES DEL PROBLEMA
# Numero de grupo
G = 3
# Largo de la viga
L = 1
# Producto entre el modulo de Young y el momento de Inercia
EI = 1
# Precision de punto flotante
FLOAT = np.float64

# ARCHIVOS DE EXPORTACION
FILE_W = "exports\\W"
FILE_SOR = "exports\\SOR"
FILE_MATRICIAL = "_MAT"
FILE_ANALITICO = "_ANAL"
FILE_EXTENSION = ".csv"

# Exporta una tabla con el muestreo de factores w y sus iteraciones con discretizacion 'n'
def exportarTablaW( file, info, n ):
	print( "\n* Exportando muestreo de factores w para n=" + repr(n) +":" )
	f = open( file, "w" )
	f.write( "n," + repr(n) + "\n" )
	f.write( "w,k\n" )
	for r in info:
		f.write( repr(r[0]) + "," + repr(r[1]) + "\n" )
	f.close()
	print( "\tArchivo '" + file + "' exportado." )

# Exporta una tabla con los calculos de cada iteracion del metodo SOR con discretizacion 'n' y el factor 'w'
def exportarTablaSOR( file, info, n, w ):
	print( "\n* Exportando calculos SOR para n=" + repr(n) + " w=" + repr(w) + ":" )
	d = info[0][3].shape[0] # dimension del resultado
	f = open( file, "w" )
	f.write( "n," + repr(n) + ",w optimo," + repr(w) + "\n" )
	f.write( "k,|Er|,log|Er|,p" )
	for i in range(0,d):
		f.write( ",x" + repr(i) )
	f.write( "\n" )
	for r in info:
		f.write( repr(r[0]) + "," + repr(r[1]) + "," + repr(math.log10(r[1])) + "," + repr(r[2]) )
		for i in range(0,d):
			f.write( "," + repr(r[3][i]) )
		f.write( "\n" )
	f.close()
	print( "\tArchivo '" + file + "' exportado." )

# Devuelve el valor de la carga de la viga en la coordenada 'x'
def carga( x ):
	return G + math.pow(G,2) * (x - math.pow(x,2))

# Genera la matriz 'A' y el vector 'b' para una discretizacion de 'n' segmentos
def generar( n ):
	# genero el vector 'b'
	print( "\tGenerando vector b dim=" + repr(n+1) + " ..", end='' )
	b = np.zeros(n+1, FLOAT)
	b[0] = 0
	for i in range(1,n):
		b[i] = carga(i*L/n) * math.pow(L/n,4) / EI
	b[n] = 0
	print( " Generado. " )
	
	# genero la matriz 'A'
	print( "\tGenerando matrix A dim=" + repr(n+1) + "x" + repr(n+1) + " ..", end='' )
	A = np.zeros((n+1,n+1), FLOAT)
	# fila: 0
	A[0][0] = 1
	# fila: 1
	A[1][0] = -4
	A[1][1] = 5
	A[1][2] = -4
	A[1][3] = 1
	# filas: 1 < i < n-1
	for j in range(2,n-1):
		 offset = j-2
		 A[j][offset+0] = 1
		 A[j][offset+1] = -4
		 A[j][offset+2] = 6
		 A[j][offset+3] = -4
		 A[j][offset+4] = 1
	# fila: n-1
	A[n-1][n-3] = 1
	A[n-1][n-2] = -4
	A[n-1][n-1] = 5
	A[n-1][n] = -4
	# fila: n
	A[n][n] = 1
	print( " Generada. " )
	
	return A, b

# Genera una semilla arbitraria de 'n' elementos
def generarSemilla( n ):
	s = np.zeros(n, FLOAT)
	for i in range(0, s.shape[0]):
		s[i] = 0.1 - 0.1 * math.pow(2*i/(n-1)-1, 2)
	return s
	
# Calcula el resultado de una iteracion del metodo SOR de forma matricial para 'Tsor' 'Csor' y 'x'
def calcularIteracionSORMatricial( Tsor, Csor, x ):
	return np.matmul(Tsor, x) + Csor

# Calcula el resultado de una iteracion del metodo SOR de forma analitica 'x', 'b' y el factor 'w'
def calcularIteracionSORAnalitico( x, b, w ):
	d = b.shape[0]
	r = np.copy(x)
	r[0] = 0
	r[1] = w * (b[1] + 4*r[0] + 4*r[2] - r[3] ) / 5 + (1-w) * r[1]
	for i in range(2,d-2):
		r[i] = w * (b[i] - r[i-2] + 4*r[i-1] + 4*r[i+1] - r[i+2] ) / 6 + (1-w) * r[i]
	r[d-2] = w * (b[d-2] + 4*r[d-1] + 4*r[d-3] - r[d-4] ) / 5 + (1-w) * r[d-2]
	r[d-1] = 0
	return r

# Resuelve por el metodo SOR la ecuacion 'A x = b' a partir de la semilla 'x', el factor 'w' y la tolerancia relativa 'rtol' y devuelve el resultado
def calcularSOR( A, x, b, w, rtol, matricial=False ):
	# datos de arranque: k=0, |Er|, p, x
	datos = [ [0,1,1,np.copy(x)] ]
	
	# dimension
	d = b.shape[0]
	print( "\tCalculando SOR " + ("matricial" if matricial else "analitico") + " dim=" + repr(d) + " w=" + format(w,".2f") + " rtol=" + repr(rtol) + ".", end='' )
	
	# matriz diagonal
	D = np.zeros((d,d), FLOAT)
	for i in range(0,d):
		D[i][i] = A[i][i]
	
	# matriz superior
	U = np.zeros((d,d), FLOAT)
	for j in range(0,d-1):
		for i in range(j+1,d):
			U[j][i] = -1 * A[j][i]
		
	# matriz inferior
	L = np.zeros((d,d), FLOAT)
	for j in range(1,d):
		for i in range(0,j):
			L[j][i] = -1 * A[j][i]
	
	# inversa(D-wL)
	DwLi = LA.inv( D - w*L )
	
	# Tsor = inversa(D-wL)((1-w)D+wU)
	Tsor = np.matmul( DwLi, ((1-w)*D + w*U) )
	# Csor = w * inversa(D-wL) * b
	Csor = np.matmul( w*DwLi, b )
	
	# calculo SOR
	k = 0 # iteraciones
	e = 0 # delta x actual
	e1 = 0 # delta x anterior
	e2 = 0 # delta x anterior al anterior
	while( k<999999 ): # limite de iteraciones para prevenir loops infinitos en caso de divergencias
		k += 1 # cuento la iteracion
		x_ = np.copy( x ) # copio solucion previa
		
		# calculo el x
		if ( matricial ):
			x = calcularIteracionSORMatricial( Tsor, Csor, x )
		else:
			x = calcularIteracionSORAnalitico( x, b, w )
		
		# errores previos
		e2 = e1
		e1 = e
		e = LA.norm(x-x_)
		Er = e / LA.norm(x) # error relativo
		# calculo de p
		p = (math.log(e/e1) / math.log(e1/e2)) if (k>3) else 1
		# agrego datos de la iteracion actual
		datos.append( [k, Er, p, np.copy(x)] )
		# pruebo la tolerancia
		if ( Er <= rtol ):
			break
	
	print( " k=" + repr(k) + "." )
	return datos

# Hace un muestreo de las cantidad de iteraciones para factores w en el intervalo ('wMin', 'wMax'] con incrementos 'inc' para resolver por SOR la ecuacion 'A x = b' a partir de una semilla 'x' con una tolerancia relativa 'rtol'
def samplearW( A, x, b, wMin, wMax, inc, rtol, matricial=False ):
	print( "Sampleando W .." )
	# resuelvo para cada w
	c = math.floor( (wMax - wMin)/inc )
	info = []
	kOpt = np.inf
	wOpt = wMin
	for i in range(0,c):
		w = wMin+i*inc
		datos = calcularSOR( A, x, b, w, rtol, matricial )
		k = len( datos ) - 1
		info.append( [w, k] )
		if ( k<kOpt ):
			kOpt = k
			wOpt = w
	print( "Sampleado." )
	return wOpt, info

# Estima el factor w optimo con incrementos 'inc' para resolver por SOR una discretizacion 'n' del problema con una tolerancia relativa 'rtol'
def estimarWOptimo( n, inc, rtol, matricial=False ):
	print( "Estimando W optimo para n=" + repr(n) + ":" )
	A, b = generar( n )
	x = generarSemilla( n+1 )
	w, info = samplearW( A, x, b, 1, 2, inc, rtol, matricial )
	print( "Factor optimo estimado. w=" + repr(w) )
	exportarTablaW( FILE_W + repr(n) + (FILE_MATRICIAL if matricial else FILE_ANALITICO) + FILE_EXTENSION, info, n )
	return w

# Resuelvo el problema por SOR con el factor 'w', discretizacion 'n' y tolerancia 'rtol'
def resolver( n, w, rtol, matricial=False ):
	print( "Resolviendo para n=" + repr(n) + " w=" + repr(w) + ":" )
	A, b = generar( n )
	x = generarSemilla( n+1 )
	info = calcularSOR( A, x, b, w, rtol, matricial )
	print( "Resuelto." )
	exportarTablaSOR( FILE_SOR + repr(n) + (FILE_MATRICIAL if matricial else FILE_ANALITICO) + FILE_EXTENSION, info, n, w )

# Hace una resolucion del trabajo practico de forma 'matricial' o analitica
def resolverTP( matricial ):
	# Estimo los w optimos para cada discretizacion
	print( "--------------------------------" )
	w5 = estimarWOptimo( 5, 0.05, 0.01, matricial )
	print( "--------------------------------" )
	w10 = estimarWOptimo( 10, 0.05, 0.01, matricial )
	print( "--------------------------------" )
	w100 = estimarWOptimo( 100, 0.05, 0.01, matricial )

	# Resuelvo el problema para cada discretizacion
	print( "--------------------------------" )
	resolver( 5, w5, 0.0001, matricial )
	print( "--------------------------------" )
	resolver( 10, w10, 0.0001, matricial )
	print( "--------------------------------" )
	resolver( 100, w100, 0.0001, matricial )

# Resuelvo el TP de manera matricial
resolverTP( True )

# Resuelvo el TP de manera analitica
resolverTP( False )