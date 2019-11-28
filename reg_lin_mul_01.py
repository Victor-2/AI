# Regresion lineal con multiples variables

import numpy as np 
import matplotlib.pyplot as plt

def calcularCosto(X, y, theta):
    J = 0
    s = np.power((X.dot(theta) - np.transpose([y])), 2)
    J = (1.0 / (2 * m)) * s.sum(axis = 0)

    return J

def descensoGradienteMulti(X, y, theta, alpha, num_iteraciones):
    m = len(y) # numero de ejemplos de entrenamiento
    J_history = np.zeros((num_iteraciones, 1))

    for i in range(num_iteraciones):
        theta = theta - alpha * (1.0 / m) * np.transpose(X).dot(X.dot(theta) - np.transpose([y]))
        # Guardar el costo de J en cada iteracion 
        J_history[i] = calcularCosto(X, y, theta)
        # print(J_history[i])

    return theta, J_history

def normalizarCaracteristicas(X):
    # Se requiere establecer los valores de manera correcta
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))

    for i in range(X.shape[1]):
    	mu[:, i] = np.mean(X[:, i])
    	sigma[:, i] = np.std(X[:, i])
    	X_norm[:, i] = (X[:, i] - float(mu[:, i])) / float(sigma[:, i])

    return X_norm, mu, sigma

# ================ Cargar datos de entrenamiento ================
print('Cargando datos ...\n')
data = np.loadtxt('ex1data2.txt', delimiter=",")
X = data[:, :2]
y = data[:, 2]
m = len(y) # numero de ejemplos de entrenamiento

# Imprime algunos puntos de los datos
print('Primeros 10 ejemplos del dataset: \n')
for i in range(10):
    print ("x = [{:.0f} {:.0f}], y = {:.0f}".format(X[i,0], X[i,1], y[i]))

input('Programa en pausa. Prescione <enter> para continuar.\n')

# ================ Normalizacion de las caracteristicas ================
# Escalar las caracteristicas y establecer en cero
print('Normalizar caracteristicas...')

X_norm, mu, sigma = normalizarCaracteristicas(X)

# Añadir término de intercepción x0 a X
X_x0agregado = np.column_stack((np.ones((m, 1)), X_norm)) # Añadir una columna de unos a x

# ================ Descenso por el gradiente ================
print('Ejecutando descenso por el gradiente ...')
# Se establece un valor para alpha
alpha = 0.01
num_iteraciones = 400

# inicializa theta y ejecuta descenso por el gradiente
theta = np.zeros((3, 1)) 

theta, J_history = descensoGradienteMulti(X_x0agregado, y, theta, alpha, num_iteraciones)

# Graficar la convergencia de la grafica
plt.plot(range(J_history.size), J_history, "-b", linewidth = 2)
plt.xlabel('Numero de iteraciones')
plt.ylabel('Costo J')
plt.show(block = False)

# Mostrar los resultados del descenso por el gradiente
print('Theta calculado a partir del descenso por el gradiente: ')
print("{:f}, {:f}, {:f}".format(theta[0, 0], theta[1, 0], theta[2, 0]))
print("")

# Estimar el precio de una casa sobre 1650 pies cuadrados y br
# Recuerda que la primera columna de X es de unos, por este motivo no es 
# necesario normalizar esta columna

area_norm = (1650 - float(mu[:, 0])) / float(sigma[:, 0])
br_norm = (3 - float(mu[:, 1]))/float(sigma[:, 1])
house_norm_padded = np.array([1, area_norm, br_norm])

price = np.array(house_norm_padded).dot(theta)

# ============================================================

print("Precio pronosticado de una casa de 1650 pies cuadrados y 3 br (usando pendiente de gradiente):\n ${:,.2f}".format(price[0]))

input('Programa en pausa. Prescione enter para continuar.\n')
