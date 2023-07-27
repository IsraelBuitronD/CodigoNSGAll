import functools
import math
import matplotlib.pyplot as plt
import numpy as np

#----------Variables Globales-----------

n_poblacion = 50  # Tamaño de la población
n_generacion = 600  # Número de generaciones a ejecutar
tasa_mutacion = 0.05  # Tasa de mutación para la evolución
tasa_cruza = 1.0  # Tasa de cruce para la evolución

d_min = -5  # Valor mínimo del rango de búsqueda
d_max = 5  # Valor máximo del rango de búsqueda
n = 3  # Número de variables de la función objetivo
m = 3  # Número de funciónes objetivo

#---------------------------------------

def graficar_2d(pt):
    """
    Grafica los puntos en 2D.

    Args:
        pt (numpy.ndarray): Arreglo con los puntos a graficar.

    """
    plt.plot(pt[:n_poblacion, 0], pt[:n_poblacion, 1], ".")
    plt.pause(0.001)
    plt.draw()
    plt.cla()

def graficar(pt, ax):
    """
    Grafica los puntos en 3D.

    Args:
        pt (numpy.ndarray): Arreglo con los puntos a graficar.
        ax (mpl_toolkits.mplot3d.axes3d.Axes3D): Objeto Axes3D de matplotlib.

    Returns:
        None
    """
    xg = pt[:n_poblacion, 0]
    yg = pt[:n_poblacion, 1]
    zg = pt[:n_poblacion, 2]

    ax.plot(xg, yg, zg, ".")
    plt.pause(0.001)
    plt.draw()
    plt.cla()

def dominancia(a, b):
    """
    Verifica si el punto 'a' domina al punto 'b' en términos de pareto.

    Args:
        a (numpy.ndarray): Arreglo con las coordenadas del punto 'a'.
        b (numpy.ndarray): Arreglo con las coordenadas del punto 'b'.

    Returns:
        bool: True si el punto 'a' domina a 'b', False de lo contrario.
    """

    #Se define la primer condicion de acuerdo a la dominancia de Pareto
    #Donde cada elemento de 'a' en menor o igual a cada elemnto de 'b'
    condicion_1 = a <=

    #Se define la segunda condicion
    #Donde algun elemento de a debe ser estrictamente menor a los elementos de b
    condicion_2 = a < b

    #Evaluaremos las condiciones, en caso de que se cumplan retornaremos True
    #En caso contrario retornaremos False
    if np.all(condicion_1) and np.any(condicion_2):
        return True
    return False

def matriz_dominancia(datos):
    """
    Calcula la matriz de dominancia de Pareto para un conjunto de datos.

    Args:
        datos (numpy.ndarray): Arreglo con los datos a analizar. Cada fila es un punto.

    Returns:
        numpy.ndarray: Matriz booleana que indica si cada punto domina o no a los demás puntos.
    """
    p_t = datos.shape[0]
    n = datos.shape[1]

    x = np.zeros([p_t, p_t, n])
    y = np.zeros([p_t, p_t, n])

    for i in range(p_t):
        x[i,:,:] = datos[i]
        y[:,i,:] = datos[i]

    condicion_1 = x <= y
    condicion_2 = x < y

    return np.logical_and(np.all(condicion_1, axis=2), np.any(condicion_2, axis=2))

def frentes_pareto(datos):
    """
    Calcula los frentes de Pareto para un conjunto de datos.

    Args:
        datos (numpy.ndarray): Arreglo con los datos a analizar. Cada fila es un punto.

    Returns:
        List[numpy.ndarray]: Lista de arreglos con los índices de los puntos en cada frente de Pareto.
    """

    # Creamos una lista vacía para ir guardando los conjuntos de dominancia de cada punto
    conjunto_d = []
    # Creamos una lista vacía para ir contando cuántos puntos dominan a cada uno
    cuenta = []

    # Calculamos la matriz de dominancia para los datos de entrada
    matriz = matriz_dominancia(datos)
    # Obtenemos el número de puntos
    pop_size = datos.shape[0]

    # Recorremos todos los puntos y calculamos su conjunto de dominancia y el número de puntos que domina
    for i in range(pop_size):
        dominante_actual = set()
        cuenta.append(0)
        for j in range(pop_size):
            if matriz[i,j]:
                dominante_actual.add(j)
            elif matriz[j,i]:
                cuenta[-1] += 1
        conjunto_d.append(dominante_actual)

    # Convertimos la lista de cuentas a un arreglo de numpy para poder realizar operaciones matemáticas con él
    cuenta = np.array(cuenta)
    # Creamos una lista vacía para ir guardando los frentes de Pareto
    frentes = []
    # Iteramos hasta que no queden más puntos en los frentes de Pareto
    while True:
        # Buscamos los puntos que no son dominados por ningún otro
        frente_actual = np.where(cuenta == 0)[0]
        # Si no hay puntos que no sean dominados, salimos del loop
        if len(frente_actual) == 0:
            break
        # Añadimos los puntos del frente actual a la lista de frentes de Pareto
        frentes.append(frente_actual)

        # Marcamos los puntos del frente actual como dominados (-1 en la lista de cuentas)
        for individual in frente_actual:
            cuenta[individual] = -1
            # Recorremos los puntos que dominaba el punto actual y reducimos su cuenta en 1
            dominado_actual_c = conjunto_d[individual]
            for dominado_aux in dominado_actual_c:
                cuenta[dominado_aux] -= 1
    return frentes

def crowding(datos, fronts):
    """
    Calcula la medida de crowding de cada individuo en una población

    Args:
    datos (np.array): Matriz de datos, donde cada fila representa un individuo y cada columna una variable
    fronts (list): Lista de frentes de Pareto, donde cada frente es una lista de índices de individuos en datos

    Returns:
    crowding_medida (np.array): Arreglo de medida de crowding para cada individuo en datos
    """

    columnas = datos.shape[1]  # número de columnas, es decir, el número de funciones objetivo
    filas = datos.shape[0]  # número de filas, es decir, el número de individuos en la población

    # Se calcula la distancia de crowding de cada individuo para cada función objetivo
    crowding_distance = np.zeros_like(datos)
    for n in range(columnas):
        f_min = np.min(datos[:, n])  # valor mínimo de la n-ésima columna (función objetivo)
        f_max = np.max(datos[:, n])  # valor máximo de la n-ésima columna (función objetivo)
        v = f_max - f_min  # rango de la n-ésima función objetivo
        crowding_distance[:, n] = (datos[:, n] - f_min) / v  # distancia de crowding de cada individuo para la n-ésima función objetivo

    datos = crowding_distance  # se reasignan las distancias de crowding a la variable 'datos'
    crowding_medida = np.zeros(filas)  # inicialización del arreglo de medida de crowding para cada individuo

    # Se calcula la medida de crowding de cada individuo en cada frente de Pareto
    for front in fronts:
        for n in range(columnas):
            frente_ordenado = sorted(front, key=lambda x: datos[x, n])  # ordenamiento del frente de Pareto por la distancia de crowding para la n-ésima función objetivo
            crowding_medida[frente_ordenado[0]] = np.inf  # se pone en infinito la medida de crowding del primer individuo del frente de Pareto (boundary solution)
            crowding_medida[frente_ordenado[-1]] = np.inf  # se pone en infinito la medida de crowding del último individuo del frente de Pareto (boundary solution)
            if len(frente_ordenado) > 2:
                # se calcula la medida de crowding de los individuos que no son boundary solutions
                for i in range(1, len(frente_ordenado) - 1):
                    crowding_medida[frente_ordenado[i]] += datos[frente_ordenado[i + 1], n] - datos[frente_ordenado[i - 1], n]

    return crowding_medida

def rank(fronts):
    """
    Devuelve un diccionario que contiene el índice de rango para cada individuo en los frentes de Pareto dados.

    Args:
    - fronts (list): Una lista de listas de índices de individuos que pertenecen a cada frente de Pareto.

    Returns:
    - rank_indice (dict): Un diccionario que contiene el índice de rango para cada individuo. 
    El índice de rango es un número entero que indica el nivel del frente de Pareto al que pertenece el individuo 
    (los individuos en el primer frente tienen un índice de rango de 0, los del segundo frente tienen un índice de rango de 1, etc.).
    """

    rank_indice = {}  # Diccionario para almacenar el índice de rango de cada individuo
    for i, front in enumerate(fronts):  # Recorrer cada frente de Pareto y asignar el índice de rango a cada individuo
        for x in front:
            rank_indice[x] = i

    return rank_indice

def ordenamiento_no_dominado(rank_indice, crowding):
    """
    Función que realiza el ordenamiento no dominado de un conjunto de soluciones utilizando el método de crowding distance.

    Args:
        rank_indice (list): Lista con los índices de ranking de las soluciones.
        crowding (list): Lista con los valores de crowding distance de las soluciones.

    Returns:
        list: Lista con los índices de las soluciones ordenadas por orden no dominado, del mejor al "peor".
    """
    num_filas = len(crowding)
    indices = list(range(num_filas))

    def rank_no_dominado(a, b):
        if rank_indice[a] > rank_indice[b]:
            return -1  # -1 si bdom a y b es el menos aglomerado
        elif rank_indice[a] < rank_indice[b]:
            return 1  # si a domina b y a es menos aglomerado
        else:
            if crowding[a] < crowding[b]:
                return -1
            elif crowding[a] > crowding[b]:
                return 1
            else:
                return 0  # si son iguales en todo sentido

    # indices del ordenamiento no dominado, del mejor al "peor"
    return sorted(indices, key=functools.cmp_to_key(rank_no_dominado), reverse=True)

def seleccion(individuos, q_t):
    """
    Selecciona q_t individuos de manera aleatoria de una población de tamaño 'individuos'.

    Args:
    - individuos (int): tamaño de la población.
    - q_t (int): cantidad de individuos a seleccionar.

    Returns:
    - elegidos (List[int]): lista de índices de los individuos seleccionados.
    """
    elegidos = []  # Inicializa la lista de índices de individuos seleccionados
    for _ in range(q_t):  # Repite el proceso q_t veces
        index = np.random.randint(0,individuos)  # Escoge un índice aleatorio entre 0 e individuos (no inclusive)
        padre = index  # Elige el índice como el padre
        elegidos.append(padre)  # Agrega el índice del padre a la lista de seleccionados

    return elegidos  # Retorna la lista de índices seleccionados

def cruza(individuos, tasa):
    """
    Realiza la cruza de los individuos según una tasa de cruza.

    Args:
        individuos (numpy.ndarray): Array bidimensional con los individuos a cruzar.
        tasa (float): Tasa de cruza entre 0 y 1.

    Returns:
        numpy.ndarray: Array bidimensional con la descendencia generada por la cruza.
    """
    descendencia = individuos.copy()  # se copian los individuos para no modificar el array original
    for i in range(int(tasa/2)):  # se realiza la cruza para la mitad de la tasa
        x = np.random.randint(0, individuos.shape[0])  # se elige aleatoriamente un primer individuo
        y = np.random.randint(0, individuos.shape[0])  # se elige aleatoriamente un segundo individuo
        while x == y:  # se verifica que los índices no sean iguales para que no se cruze un individuo consigo mismo
            x = np.random.randint(0, individuos.shape[0])
            y = np.random.randint(0, individuos.shape[0])
        punto_cruza = np.random.randint(1, individuos.shape[1])  # se elige aleatoriamente el punto de cruza
        descendencia[2*i, 0:punto_cruza] = individuos[x, 0:punto_cruza]  # se crea el hijo 1 con la primera parte del primer padre y la segunda del segundo padre
        descendencia[2*i, punto_cruza:] = individuos[y, punto_cruza:]
        descendencia[2*i+1, 0:punto_cruza] = individuos[y, 0:punto_cruza]  # se crea el hijo 2 con la primera parte del segundo padre y la segunda del primer padre
        descendencia[2*i+1, punto_cruza:] = individuos[x, punto_cruza:] 
    return descendencia

def mutacion(individuo, min_val, max_val, tasa_m):
    """
    Realiza la mutación de un individuo mediante la adición de ruido gaussiano.

    Args:
    - individuo (numpy.ndarray): El individuo a mutar.
    - min_val (float): El valor mínimo posible para los elementos del individuo.
    - max_val (float): El valor máximo posible para los elementos del individuo.
    - tasa_m (float): La tasa de mutación, un número entre 0 y 1 que indica la probabilidad de mutación.

    Returns:
    - descendencia (numpy.ndarray): El individuo mutado.
    """
    mutacion_p = (max_val - min_val) * tasa_m # calcula la probabilidad de mutación
    descendencia = individuo.copy() # copia el individuo
    # genera ruido gaussiano para la mutación
    descendencia += np.random.normal(0, mutacion_p, size=descendencia.shape)
    # limita los valores del individuo a los valores máximos y mínimos permitidos
    descendencia = np.clip(descendencia, min_val, max_val)
    return descendencia

def NSGA_II(p_t, valores, g):
    """
    Implementación del algoritmo NSGA-II para optimización multiobjetivo.

    Args:
        p_t (np.array): Población actual.
        valores (np.array): Valores objetivos correspondientes a la población actual.
        g (int): Número de generación actual.

    Returns:
        np.array: Población de la siguiente generación.
    """

    # Imprimimos información de la generación actual
    print(f"NSGA-II en la generación {g+1}")

    # Obtenemos los frentes de Pareto de la población actual
    fronts = frentes_pareto(valores)

    # Obtenemos la posición de cada individuo en su frente
    posicion_ranking = rank(fronts)

    # Calculamos la distancia de crowding de cada individuo
    crowding_d = crowding(valores, fronts)

    # Ordenamos los individuos por el ranking y la distancia de crowding
    indices_nd = ordenamiento_no_dominado(posicion_ranking, crowding_d)

    # Seleccionamos la porción de mejores sobrevivientes para la siguiente generación
    sobrevivientes = p_t[indices_nd[:n_poblacion]]

    # Seleccionamos los padres para la cruza
    seleccionados = seleccion(individuos=n_poblacion, q_t=n_poblacion)

    # Realizamos la cruza con los padres seleccionados
    cruza_t = cruza(seleccionados, tasa=tasa_cruza)

    # Realizamos la mutación a partir de los sobrevivientes obtenidos
    pt_next = np.array([mutacion(sobrevivientes[i], dmin, dmax, tasa_mutacion) for i in cruza_t])

    # Combinamos los sobrevivientes y los hijos mutados para formar la siguiente generación
    poblacion_sig = np.concatenate([sobrevivientes, pt_next])

    # Retornamos la población de la siguiente generación
    return poblacion_sig

def sumatoria_fon(x, n, f):
    """
    Calcula la suma de la función de Fonseca y Fleming.

    Args:
        x (np.array): Valores de entrada.
        n (int): Dimensión del problema.
        f (str): Nombre de la función (f1 o f2).

    Returns:
        float: Sumatoria de la función de Fonseca y Fleming.
    """
    s = 0
    for i in range(n):
        if f == "f1":
            s += pow((x[:, i] - (1 - np.sqrt(n))), 2)
        elif f == "f2":
            s += pow((x[:, i] + (1 - np.sqrt(n))), 2)
    return s

def fonseca(x, n):
    """
    Función de Fonseca y Fleming para optimización multiobjetivo.

    Args:
        x (np.array): Valores de entrada.
        n (int): Dimensión del problema.

    Returns:
        np.array: Valores objetivos correspondientes a los valores de entrada.
    """
    f1 = 1 - np.exp(-sumatoria_fon(x, n, "f1"))
    f2 = 1 - np.exp(-sumatoria_fon(x, n, "f2"))

    return np.stack([f1, f2], axis=1)

def seno(f):
    """
    Función que aplica la función seno a cada valor de una lista.

    Args:
        f (list): Lista de valores a los que se les aplicará la función seno.

    Returns:
        list: Lista de valores con la función seno aplicada a cada uno de ellos.
    """
    for i in range(f):
        f[i] = math.sin(10 * math.pi * f[i])
    return f

def g(x, n):
    """
    Función auxiliar para la función ZDT3.

    Args:
        x (np.array): Valores de entrada.
        n (int): Dimensión del problema.

    Returns:
        float: Valor de la función auxiliar.
    """
    g = 0
    for i in range(n):
        g += 1 + (9 / 29) * (x[:, i])
    return g

def h1(f, g):
    """
    Calcula el valor h1 en función de f y g.

    Args:
    f: valor de la variable f
    g: valor de la variable g

    Returns:
    Valor del valor auxiliara h1
    """
    h = 1 - np.sqrt(f/g)
    return h

def h2(f, g):
    """
    Calcula el valor h2 en función de f y g.

    Args:
    f: valor de la variable f
    g: valor de la variable g

    Returns:
    Valor del valor auxiliar h2
    """
    h = 1 - pow((f/g), 2)
    return h

def h3(f, g):
    """
    Calcula el valor h3 en función de f y g.

    Args:
    f: valor de la variable f
    g: valor de la variable g

    Returns:
    Valor del valor auxiliar h3
    """
    h = 1 - np.sqrt(f/g) - ((f/g) * np.sin(f))
    return h

def f(x):
    """
    Obtiene el valor de la variable f a partir de la matriz x.

    Args:
    x: matriz de valores
    
    Returns:
    Valor de la variable f
    """
    f = x[:, 1]
    return f

def zdt1(x, n):
    """
    Calcula la función de Zitzler-Deb-Thiele nº 1 en función de x y n.
    
    Args:
    x: matriz de valores de entrada
    n: número de variables de entrada
    
    Returns:
    Matriz con los valores de salida de la función
    """
    f1 = f(x)
    f2 = g(x, n) * h1(f(x), g(x, n))
    return np.stack([f1, f2], axis=1)

def zdt2(x, n):
    """
    Calcula la función de Zitzler-Deb-Thiele nº 2 en función de x y n.
    
    Args:
    x: matriz de valores de entrada
    n: número de variables de entrada
    
    Returns:
    Matriz con los valores de salida de la función
    """
    f1 = f(x)
    f2 = g(x, n) * h2(f(x), g(x, n))
    return np.stack([f1, f2], axis=1)

def zdt3(x, n):
    """
    Calcula la función de Zitzler-Deb-Thiele nº 3 en función de x y n.
    
    Args:
    x: matriz de valores de entrada
    n: número de variables de entrada
    
    Returns:
    Matriz con los valores de salida de la función
    """
    f1 = f(x)
    f2 = g(x, n) * h3(f(x), g(x, n))
    return np.stack([f1, f2], axis=1)
 
def omega(gx, xi):
    """
    Calcula la función omega en función de gx y xi.
    
    Args:
    gx: valor de gx
    xi: valor de xi
    
    Returns:
    Valor de la función omega
    """
    return np.pi / (4 * (1 + gx)) * (1 + 2 * gx * xi)

def dtlz1(x):
    """
    Calcula la función dtlz1 en función de x.
    
    Args:
    x: matriz de valores de entrada
    
    Returns:
    Matriz con los valores de salida de la función
    """
    print("DTLZ1")
    
    funciones = []

    gx = 100 * (np.sum(np.square(x[:, m:] - 0.5) - np.cos(20 * np.pi * (x[:, m:] - 0.5)), axis=1))
 
    for f in range(m): # M número de funciones
        if f == m-1:
            xi = (1 - x[:, 0]) #(1 - x1)
        else:
            xi = x[:, 0] # x1
            
            for i in range((m-1) - f - 1): # (... Xm-1)
                xi = xi * x[:, (i+1)] 
            
            if f == 0:
                xi = xi
            else:
                xi = xi * (1 - x[:, ((m-1) - f)]) # (1 - Xm-1)
            
        fi = 0.5 * (xi * (1 + gx)) # fm(x)

        funciones.append(fi)

    return np.stack(funciones, axis=1)

def dtlz2(x):
    """
    Calcula la función dtlz2 en función de x.

    Args:
    x: matriz de valores de entrada

    Returns:
    Matriz con los valores de salida de la función
    """
    print("DTLZ2")
    
    funciones = []
    
    gx = np.sum(np.square(x[:, m:] - 0.5), axis=1)
    
    for f in range(m):  # m numero de funciones
        if f == m - 1:
            xi = np.sin(x[:, 0] * (np.pi) / 2)  # sen(x1) pi/2
        else:
            xi = np.cos(x[:, 0] * (np.pi) / 2)  # x1
            
            for i in range((m - 1) - f - 1):  # (... Xm-1)
                xi = xi * (np.cos(x[:, i + 1] * (np.pi) / 2)) 
            
            if f == 0:
                xi = xi * (np.cos(x[:, (m - 2) - f] * (np.pi) / 2)) 
            else:
                xi = xi * (np.sin(x[:, (m - 1) - f] * (np.pi) / 2))  # (1 - Xm-1)
            
        fi = (1 + gx) * xi  # fm(x)

        funciones.append(fi)
    
    return np.stack(funciones, axis=1)

def dtlz3(x):
    """
    Calcula la función dtlz3 en función de x.

    Args:
    x: matriz de valores de entrada

    Returns:
    Matriz con los valores de salida de la función
    """
    print("DTLZ3")
    
    funciones = []
    
    gx = 100 * (np.sum(np.square(x[:, m:] - 0.5) - np.cos(20 * np.pi * (x[:, m:] - 0.5)), axis=1))
    
    for f in range(m):  # m numero de funciones
        if f == m - 1:
            xi = (np.sin(x[:, 0] * (np.pi) / 2))  # (sen(x1) pi/2)
        else:
            xi = np.cos(x[:, 0] * (np.pi) / 2)  # x1
            
            for i in range((m - 1) - f - 1):  # (... Xm-1)
                
                xi  = xi * (np.cos(x[:, i + 1] * (np.pi) / 2)) 
            
            if f == 0:
                xi = xi * (np.cos(x[:, (m - 2) - f] * (np.pi) / 2)) 
            else:
                xi = xi * (np.sin(x[:, (m - 1) - f] * (np.pi) / 2))  # (1 - Xm-1)
            
        fi = (1 + gx) * xi  # fm(x)

        funciones.append(fi)
    
    return np.stack(funciones, axis=1)

def dtlz4(x):
    """
    Calcula la función dtlz4 en función de x.

    Args:
    x: matriz de valores de entrada

    Returns:
    Matriz con los valores de salida de la función
    """
    print("DTLZ4")

    m = 3  # m numero de funciones

    gx = np.sum(np.square(x[:, m:] - 0.5), axis=1)

    funciones = []

    for f in range(m):
        if f == m - 1:
            xi = np.sin(np.power(x[:, 0], 10) * np.pi / 2)  # (sin(x1) pi/2)
        else:
            xi = np.cos(np.power(x[:, 0], 10) * np.pi / 2)  # x1
            for i in range((m - 1) - f - 1):  # (... Xm-1)
                xi = xi * np.cos(np.power(x[:, i + 1], 10) * np.pi / 2)
            if f == 0:
                xi = xi * np.cos(np.power(x[:, (m - 2) - f], 10) * np.pi / 2)
            else:
                xi = xi * np.sin(np.power(x[:, (m - 1) - f], 10) * np.pi / 2)  # (1 - Xm-1)
        fi = (1 + gx) * xi  # fm(x)
        funciones.append(fi)

    return np.stack(funciones, axis=1)

def dtlz5(x):
    """
    Calcula la función dtlz5 en función de x.

    Args:
    x: matriz de valores de entrada

    Returns:
    Matriz con los valores de salida de la función
    """
    print("DTLZ5")

    funciones = []

    gx = np.sum(np.square(x[:, m:] - 0.5), axis=1)

    for f in range(m):  # m numero de funciones
        if f == m - 1:
            xi = (np.sin(omega(gx, x[:, 0]) * (np.pi) / 2))  # ( sen(x1) pi/2 )
        else:
            xi = np.cos(omega(gx, x[:, 0]) * (np.pi) / 2)  # x1

            for i in range((m - 1) - f - 1):  # (... Xm-1)
                xi = xi * (np.cos(omega(gx, x[:, i + 1]) * (np.pi) / 2))

            if f == 0:
                xi = xi * (np.cos(omega(gx, x[:, ((m - 2) - f)]) * (np.pi) / 2))
            else:
                xi = xi * (np.sin(omega(gx, x[:, ((m - 1) - f)]) * (np.pi) / 2))  # (1 - Xm-1)

        fi = (1 + gx) * xi  # fm(x)

        funciones.append(fi)

    return np.stack(funciones, axis=1)

def dtlz6(x):
    """
    Calcula la función dtlz6 en función de x.

    Args:
    x: matriz de valores de entrada

    Returns:
    Matriz con los valores de salida de la función
    """
    print("DTLZ6")
    
    funciones = []

    gx = np.sum(np.power(x[:, m:], 0.1), axis=1)
    
    for f in range(m):  # m nuemro de funciones
        if f == m-1:
            xi = np.sin(omega(gx, x[:, 0]) * (np.pi)/2)  # sin(x1 * pi/2)
        else:
            xi = np.cos(omega(gx, x[:, 0]) * (np.pi)/2)  # x1
            
            for i in range((m-1)-f-1):  # (... Xm-1)
                xi *= np.cos(omega(gx, x[:, i+1]) * (np.pi)/2)
            
            if f == 0:
                xi *= np.cos(omega(gx, x[:, (m-2)-f]) * (np.pi)/2)
            else:
                xi *= np.sin(omega(gx, x[:, (m-1)-f]) * (np.pi)/2)  # (1 - Xm-1)
            
        fi = (1 + gx) * xi  # fm(x)

        funciones.append(fi)
    
    return np.stack(funciones, axis=1)

def kursawe(x):
    """Calcula los valores objetivos de la función Kursawe.

    Args:
        x (array): matriz de entrada con las variables de decisión.

    Returns:
        array: matriz con los valores objetivos calculados.
    """
    sq_x = x ** 2
    objetivo_1 = -10 * np.exp(-0.2 * np.sqrt(np.sum(sq_x[:, :2], axis=1))) - 10 * np.exp(-0.2 * np.sqrt(np.sum(sq_x[:, 1:], axis=1)))
    objetivo_2 = np.sum(np.power(np.abs(x), 0.8) + 5 * np.sin(x ** 3), axis=1)
    
    return np.stack([objetivo_1, objetivo_2], axis=1)

def main():
    """
    Función principal que inicia el programa.
    """
    print("Hola, iniciando...")

    # Creamos la población inicial p0
    x = np.random.uniform(d_min, d_max, (2 * n_poblacion, n))
    plt.ion()

    # ax = plt.axes(projection="3d")
    # ax.view_init(45, 45)

    for generation in range(n_generacion):
        pt = kursawe(x)
        x = NSGA_II(x, pt, generation)
        # graficar(pt,ax)
        graficar_2d(pt)

    plt.ioff()

    plt.plot(pt[:n_poblacion, 0], pt[:n_poblacion, 1], ".", color="r")
    plt.show()

    # Comentario de prueba

    """
    ax = plt.axes(projection="3d")
    ax.view_init(45, 45)
    xg = pt[:n_poblacion, 0]
    yg = pt[:n_poblacion, 1]
    zg = pt[:n_poblacion, 2]

    ax.plot(xg, yg, zg, ".")
    plt.show()
    """

main()
