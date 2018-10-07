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
FILE_INDICIAL = "_IND"
FILE_EXTENSION = ".csv"


# Exporta una tabla con el muestreo de factores w y sus iteraciones con discretizacion 'n'
def exportarTablaW(file, info, n):
    print("\n* Exportando muestreo de factores w para n=" + repr(n) + ":")
    f = open(file, "w")
    f.write("n," + repr(n) + "\n")
    f.write("w,k\n")
    for r in info:
        f.write(repr(r[0]) + "," + repr(r[1]) + "\n")
    f.close()
    print("\tArchivo '" + file + "' exportado.")


# Exporta una tabla con los calculos de cada iteracion del metodo SOR con discretizacion 'n' y el factor 'w' y orden de convergencia 'p'
def exportarTablaSOR(file, info, n, w, p):
    print("\n* Exportando calculos SOR para n=" + repr(n) + " w=" + format(w,".2f") + " p=" + format(p,".2f") + ":")
    d = info[0][2].shape[0]  # dimension del resultado
    f = open(file, "w")
    f.write("n," + repr(n) + ",w optimo," + repr(w) + ",p," + repr(p) + "\n")
    f.write("k,|Er|,log|Er|")
    for i in range(0, d):
        f.write(",v[" + repr(i) + "]")
    f.write("\n")
    for r in info:
        f.write(repr(r[0]) + "," + repr(r[1]) + "," + repr(math.log10(r[1])) )
        for i in range(0, d):
            f.write("," + repr(r[2][i]))
        f.write("\n")
    f.close()
    print("\tArchivo '" + file + "' exportado.")


# Devuelve el valor de la carga de la viga en la coordenada 'x'
def calculoDeCarga(x):
    return G + math.pow(G, 2) * (x - math.pow(x, 2))


# Genera la matriz 'K' y el vector 'f' para una discretizacion de 'n' segmentos
def generar(n):
    # genero el vector 'f'
    print("\tGenerando vector f dim=" + repr(n + 1) + " ..", end='')
    f = np.zeros(n + 1, FLOAT)
    f[0] = 0
    for i in range(1, n):
        f[i] = calculoDeCarga(i * L / n) * math.pow(L / n, 4) / EI
    f[n] = 0
    print(" Generado. ")

    # genero la matriz 'K'
    print("\tGenerando matrix K dim=" + repr(n + 1) + "x" + repr(n + 1) + " ..", end='')
    K = np.zeros((n + 1, n + 1), FLOAT)
    # fila: 0
    K[0][0] = 1
    # fila: 1
    K[1][0] = -4
    K[1][1] = 5
    K[1][2] = -4
    K[1][3] = 1
    # filas: 1 < i < n-1
    for j in range(2, n - 1):
        K[j][j-2] = 1
        K[j][j-1] = -4
        K[j][j+0] = 6
        K[j][j+1] = -4
        K[j][j+2] = 1
    # fila: n-1
    K[n-1][n-3] = 1
    K[n-1][n-2] = -4
    K[n-1][n-1] = 5
    K[n-1][n] = -4
    # fila: n
    K[n][n] = 1
    print(" Generada. ")

    return K, f


# Genera una semilla arbitraria de 'n' elementos
def generarSemilla(n):
    s = np.zeros(n, FLOAT)
    for i in range(0, s.shape[0]):
        s[i] = 0
    return s


# Calcula el resultado de una iteracion del metodo SOR de forma matricial para 'Tsor' 'Csor' y 'v'
def calcularIteracionSORMatricial(Tsor, Csor, v):
    return np.matmul(Tsor, v) + Csor


# Calcula el resultado de una iteracion del metodo SOR de forma indicial para 'v', 'f' y el factor 'w'
def calcularIteracionSORIndicial(v, f, w):
    d = f.shape[0]
    r = np.copy(v)
    r[0] = 0
    r[1] = w * (f[1] + 4 * r[0] + 4 * r[2] - r[3]) / 5 + (1 - w) * r[1]
    for i in range(2, d - 2):
        r[i] = w * (f[i] - r[i - 2] + 4 * r[i - 1] + 4 * r[i + 1] - r[i + 2]) / 6 + (1 - w) * r[i]
    r[d - 2] = w * (f[d - 2] + 4 * r[d - 1] + 4 * r[d - 3] - r[d - 4]) / 5 + (1 - w) * r[d - 2]
    r[d - 1] = 0
    return r


# Resuelve por el metodo SOR la ecuacion 'K v = f' a partir de la semilla 'v', el factor 'w' y la tolerancia relativa 'rtol' y devuelve el resultado
def calcularSOR(K, v, f, w, rtol, matricial=False):
    # datos de arranque: k=0, |Er|, x
    datos = [[0, 1, np.copy(v)]]

    # dimension
    d = f.shape[0]
    print("\tCalculando SOR " + ("matricial" if matricial else "indicial") + " dim=" + repr(d) + " w=" + format(w,".2f") + " rtol=" + repr(rtol), end='')

    # matriz diagonal
    D = np.zeros((d, d), FLOAT)
    for i in range(0, d):
        D[i][i] = K[i][i]

    # matriz superior
    U = np.zeros((d, d), FLOAT)
    for j in range(0, d - 1):
        for i in range(j + 1, d):
            U[j][i] = -1 * K[j][i]

    # matriz inferior
    L = np.zeros((d, d), FLOAT)
    for j in range(1, d):
        for i in range(0, j):
            L[j][i] = -1 * K[j][i]

    # inversa(D-wL)
    DwLi = LA.inv(D - w * L)

    # Tsor = inversa(D-wL)((1-w)D+wU)
    Tsor = np.matmul(DwLi, ((1 - w) * D + w * U))
    # Csor = w * inversa(D-wL) * b
    Csor = np.matmul(w * DwLi, f)

    # calculo SOR
    k = 0 # iteraciones
    e = 0 # delta x actual
    e1 = 0 # delta x anterior
    e2 = 0 # delta x anterior al anterior
    Er = 1 # error relativo
    while (Er > rtol and k < 999999):  # pruebo la tolerancia y un limite de iteraciones para prevenir loops infinitos en caso de divergencias
        k += 1  # cuento la iteracion
        v_ = np.copy(v)  # copio solucion previa

        # calculo el x
        if (matricial):
            v = calcularIteracionSORMatricial(Tsor, Csor, v)
        else:
            v = calcularIteracionSORIndicial(v, f, w)

        # errores previos
        e2 = e1
        e1 = e
        e = LA.norm(v - v_)
        Er = e / LA.norm(v)  # error relativo
        # agrego datos de la iteracion actual
        datos.append([k, Er, np.copy(v)])

    # calculo el orden de convergencia p
    p = (math.log(e / e1) / math.log(e1 / e2)) if (k > 3) else -1

    print(" k=" + repr(k) + " p=" + format(p,".2f") + ".")
    return datos, k, p


# Hace un muestreo de las cantidad de iteraciones para factores w en el intervalo ('wMin', 'wMax'] con incrementos 'inc' para resolver por SOR la ecuacion 'K v = f' a partir de una semilla 'v' con una tolerancia relativa 'rtol'
def samplearW(K, v, f, wMin, wMax, inc, rtol, matricial=False):
    print("Sampleando W ..")
    # resuelvo para cada w
    c = math.floor((wMax - wMin) / inc)
    info = []
    kOpt = np.inf
    wOpt = wMin
    for i in range(0, c):
        w = wMin + i * inc
        datos, k, p = calcularSOR(K, v, f, w, rtol, matricial)
        #k = len(datos) - 1
        info.append([w, k])
        if (k < kOpt):
            kOpt = k
            wOpt = w
    print("Sampleado.")
    return wOpt, info


# Estima el factor w optimo con incrementos 'inc' para resolver por SOR una discretizacion 'n' del problema con una tolerancia relativa 'rtol'
def estimarWOptimo(n, inc, rtol, matricial=False):
    print("Estimando W optimo para n=" + repr(n) + ":")
    K, f = generar(n)
    v = generarSemilla(n + 1)
    w, info = samplearW(K, v, f, 1, 2, inc, rtol, matricial)
    print("Factor optimo estimado. w=" + format(w,".2f"))
    exportarTablaW(FILE_W + repr(n) + (FILE_MATRICIAL if matricial else FILE_INDICIAL) + FILE_EXTENSION, info, n)
    return w


# Resuelve el problema por SOR con el factor 'w', discretizacion 'n' y tolerancia 'rtol'
def resolver(n, w, rtol, matricial=False):
    print("Resolviendo para n=" + repr(n) + " w=" + format(w,".2f") + ":")
    K, f = generar(n)
    v = generarSemilla(n + 1)
    info, k, p = calcularSOR(K, v, f, w, rtol, matricial)
    print("Resuelto.")
    exportarTablaSOR(FILE_SOR + repr(n) + (FILE_MATRICIAL if matricial else FILE_INDICIAL) + FILE_EXTENSION, info, n, w, p)


# Hace una resolucion del trabajo practico de forma 'matricial' o 'indicial'
def resolverTP(matricial):
    # Estimo los w optimos para cada discretizacion
    print("\n--------------------------------")
    w5 = estimarWOptimo(5, 0.05, 0.01, matricial)
    print("\n--------------------------------")
    w10 = estimarWOptimo(10, 0.05, 0.01, matricial)
    print("\n--------------------------------")
    w100 = estimarWOptimo(100, 0.05, 0.00001, matricial)

    # Resuelvo el problema para cada discretizacion
    print("\n--------------------------------")
    resolver(5, w5, 0.0001, matricial)
    print("\n--------------------------------")
    resolver(10, w10, 0.0001, matricial)
    print("\n--------------------------------")
    resolver(100, w100, 0.0000001, matricial)


# Resuelvo el TP de manera matricial
resolverTP(True)
# Resuelvo el TP de manera indicial
resolverTP(False)
