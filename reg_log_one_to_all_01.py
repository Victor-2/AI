# Clasificacion multiclase 

import scipy.io
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import sys

from scipy.special import expit 
import numpy as np

def sigmoide(z):
    # SIGMOIDE Calcula la funcion sigmoidea
    # J = SIGMOIDE(z) calcula la sigmoide de z.

    g = np.zeros(z.shape)

    # g = 1/(1 + np.exp(-z))
    g = expit(z)

    return g


def funcionlrCosto(theta, X, y, lambda_reg, return_grad=False):
    # FUNCIONLRCOSTO Calcula el costo y el gradiente para la regresion logistica con regularizacion
    # J = FUNCIONLRCOSTO(theta, X, y, lambda_reg) calcula el costo utilizando
    #  theta como parametro para la regresion logistica regularizada
    #  y el gradiente de el costo w.r.t. para los parametros. 

    m = len(y) # numero de ejemplos para el entrenamiento

    J = 0
    grad = np.zeros(theta.shape)
    one = y * np.transpose(np.log( sigmoide( np.dot(X,theta) ) ))
    two = (1-y) * np.transpose(np.log( 1 - sigmoide( np.dot(X,theta) ) ))
    reg = ( float(lambda_reg) / (2*m)) * np.power(theta[1:theta.shape[0]],2).sum()
    J = -(1./m)*(one+two).sum() + reg

    grad = (1./m) * np.dot(sigmoide( np.dot(X,theta) ).T - y, X).T + ( float(lambda_reg) / m )*theta

    # en el caso de j = 0 (recuerde que grad es un vector n + 1)
    grad_no_regularization = (1./m) * np.dot(sigmoide( np.dot(X,theta) ).T - y, X).T

    # asigna solo el primer elemento de grad_no_regularization a grad
    grad[0] = grad_no_regularization[0]

    # mostrar el costo en cada iteracion
    sys.stdout.write("Cost: %f   \r" % (J) )
    sys.stdout.flush()

    if return_grad:
        return J, grad.flatten()
    else:
        return J

def mostrarDatos(X, example_width = None):
    # MOSTRARDATOS Mostrar datos en una grilla 2D
    # [h, display_array] = MOSTRARDATOS(X, example_width) muestra datos en 2D
    # almacenados en X en una grilla. Esta funcion retorna el puntero h, a la figura
    # y el arreglo visualizado que se requiere, cerrando la figura previamente abierta.
    # Evitando una advertencia despues de abrir varias figuras
    plt.close()
    
    # crea una nueva figura 
    plt.figure()
    
    # convierte el arreglo X 1D en 2D
    if X.ndim == 1:
        X = np.reshape(X, (-1, X.shape[0]))
    
    # Establecer example_width automáticamente si no se pasa
    if not example_width or not 'example_width' in locals():
        example_width = int(round(math.sqrt(X.shape[1])))
    
    # Imagen gris
    plt.set_cmap("gray")

	# Calcular filas, columnas

    m, n = X.shape
    example_height = int(n / example_width)
    # Calcular el numero de elementos para mostrar
    # #display_rows = int(math.floor(math.sqrt(m)))
	# display_cols = int(math.ceil(m / display_rows))
    display_rows = int(math.floor(math.sqrt(m)))
    display_cols = int(math.ceil(m / display_rows))

	# Espacio de relleno entre las imagenes
    pad = 1

	# Configurar pantalla en blanco

    display_array = -np.ones((pad + display_rows * (example_height + pad),  pad + display_cols * (example_width + pad)))

	# Copia cada ejemplo en un parche en la matriz de visualización
    curr_ex = 1
    
    for j in range(1,display_rows+1):
        for i in range (1,display_cols+1):
            if curr_ex > m:
                break
			# Copia el parche
			# Obtiene el valor máximo del parche para normalizar todos los ejemplos

            max_val = max(abs(X[curr_ex-1, :]))
            rows = pad + (j - 1) * (example_height + pad) + np.array(range(example_height))
            cols = pad + (i - 1) * (example_width  + pad) + np.array(range(example_width ))

			# Basic (vs. advanced) indexing/slicing is necessary so that we look can assign
			# values directly to display_array and not to a copy of its subarray.
			# from stackoverflow.com/a/7960811/583834 and 
			# bytes.com/topic/python/answers/759181-help-slicing-replacing-matrix-sections
			# Also notice the order="F" parameter on the reshape call - this is because python's 
			# default reshape function uses "C-like index order, with the last axis index 
			# changing fastest, back to the first axis index changing slowest" i.e. 
			# it first fills out the first row/the first index, then the second row, etc. 
			# matlab uses "Fortran-like index order, with the first index changing fastest, 
			# and the last index changing slowest" i.e. it first fills out the first column, 
			# then the second column, etc. This latter behaviour is what we want.
			# Alternatively, we can keep the deault order="C" and then transpose the result
			# from the reshape call.
            display_array[rows[0]:rows[-1]+1 , cols[0]:cols[-1]+1] = np.reshape(X[curr_ex-1, :], (example_height, example_width), order="F") / max_val
            curr_ex += 1

        if curr_ex > m:
            break


	# Visualizar imagen

    h = plt.imshow(display_array, vmin = -1, vmax = 1)

	# No se muestran los ejes

    plt.axis('off')
    plt.show(block=False)
    
    return h, display_array

def oneVsAll(X, y, num_labels, lambda_reg):
    # ONEVSALL entrena clasificadores de regresion logistica multiple y retorna todos
    # los clasificadores en una matriz all_theta, donde la i-esima fila de all_theta 
    # corresponde a el clasificador para la etiqueta i
    # [all_theta] = ONEVSALL(X, y, num_labels, lambda) entrena num_labels
    # un clasificador de regresion logistica and retorna cada uno de estos clasificadores
    # en una matriz all_theta, donde la i-esima fila de all_theta corresponde 
    # a el clasificador para la etiqueta i

    m, n = X.shape

    all_theta = np.zeros((num_labels, n + 1))

    # Agregar unos a la matriz de datos X
    X = np.column_stack((np.ones((m, 1)), X))

    for c in range(num_labels):
        # theta inicial para c/clases
        initial_theta = np.zeros((n + 1, 1))
        print("Entrenando {:d} fuera de las categorias{:d} ...".format(c + 1, num_labels))
               
        # funciones WITH gradient/jac parametros
        # from https://github.com/tansaku/py-coursera/issues/9#issuecomment-8801160
        myargs = (X, (y%10==c).astype(int), lambda_reg, True)
        theta = minimize(funcionlrCosto, x0 = initial_theta, args = myargs, options={'disp': True, 'maxiter':13}, method = "Newton-CG", jac = True)

        # Otros métodos que se pueden revisar posteriormente
        # theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="CG", jac=True)
        # theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="BFGS", jac=True)
        # theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="L-BFGS-B", jac=True)
        # theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="TNC", jac=True)
        # theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="SLSQP", jac=True)
        # theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="dogleg", jac=True)
        # theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="trust-ncg", jac=True)
        
        # funciones sin parametros gradient/jac
        # myargs = (X, (y%10==c).astype(int), lambda_reg)
        # theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="Nelder-Mead")
        # theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="Powell") #maybe
        # theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':10}, method="COBYLA")
        
        # Asigna la fila de all_theta correspondiente a la c/class actual
        all_theta[c,:] = theta["x"]

    return all_theta

def predecirOneVsAll(all_theta, X):
    # PREDICT Predict the label for a trained one-vs-all classifier. The labels 
    # are in the range 1..K, where K = size(all_theta, 1). 
    #  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
    #  for each example in the matrix X. Note that X contains the examples in
    #  rows. all_theta is a matrix where the i-th row is a trained logistic
    #  regression theta vector for the i-th class. You should set p to a vector
    #  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
    #  for 4 examples) 
    
    m = X.shape[0]
    num_labels = all_theta.shape[0]

    p = np.zeros((m, 1))

    # Agregar unos a la matriz de datos X
    X = np.column_stack((np.ones((m,1)), X))

    p = np.argmax(sigmoide( np.dot(X,all_theta.T) ), axis=1)

    return p


# Configurar los parametros que se utilizaran en el ejercicio
input_layer_size  = 400  # Entrada: Imagenes de digitos de 20x20
num_labels = 10          # Etiquetas del 1 al 10, se mapea la etiqueta 10 con el 0

# Parte 1: Cargar y visualizar datos =============
# Se inicia el ejercicio cargando y visualizando el dataset. 
# Se trabajara con un dataset que contiene digitos numericos escritos a mano

# Cargando datos de entrenaimento
print('Cargando y visualizando datos ...')

mat = scipy.io.loadmat('ex3data1.mat')

X = mat["X"]
y = mat["y"]

m = X.shape[0]

# Un paso crucial para lograr un buen rendimiento
# cambia la dimensión de (m, 1) a (m,) de lo contrario, la minimización no es muy efectiva ...

y = y.flatten() 

# Selecciona aleatoriamente 100 puntos de datos para mostrar
rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100], :]

mostrarDatos(sel)

input('Programa en pausa. Precione <enter> para continuar.\n')

# ============ Parte 2: Vectorizar Regresion Logistica ============
#  En esta parte del ejercicio, se reutilizará el código de regresión logística del último ejercicio.
#  la regresión logística regularizada se vectoriza. 
#  Luego se implementará la clasificación de uno contra todo para el conjunto de datos de dígitos
#  escritos a mano.

print('Entrenando regresion logistica One-vs-All ...')

lambda_reg = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_reg)

input('Programa en pausa. Precione <enter> para continuar.\n')

# ================ Parte 3: Predecir para One-Vs-All ================
#  Despues ...
pred = predecirOneVsAll(all_theta, X)

print('Conjunto de entrenamiento de precisión: {:f}'.format((np.mean(pred == y%10)*100)))
print('Conjunto de entrenamiento de precisión para 1:  {:f}'.format(np.mean(pred[500:1000]  == y.flatten()[500:1000]%10)  * 100))
print('Conjunto de entrenamiento de precisión para 2:  {:f}'.format(np.mean(pred[1000:1500] == y.flatten()[1000:1500]%10) * 100))
print('Conjunto de entrenamiento de precisión para 3:  {:f}'.format(np.mean(pred[1500:2000] == y.flatten()[1500:2000]%10) * 100))
print('Conjunto de entrenamiento de precisión para 4:  {:f}'.format(np.mean(pred[2000:2500] == y.flatten()[2000:2500]%10) * 100))
print('Conjunto de entrenamiento de precisión para 5:  {:f}'.format(np.mean(pred[2500:3000] == y.flatten()[2500:3000]%10) * 100))
print('Conjunto de entrenamiento de precisión para 6:  {:f}'.format(np.mean(pred[3000:3500] == y.flatten()[3000:3500]%10) * 100))
print('Conjunto de entrenamiento de precisión para 7:  {:f}'.format(np.mean(pred[3500:4000] == y.flatten()[3500:4000]%10) * 100))
print('Conjunto de entrenamiento de precisión para 8:  {:f}'.format(np.mean(pred[4000:4500] == y.flatten()[4000:4500]%10) * 100))
print('Conjunto de entrenamiento de precisión para 9:  {:f}'.format(np.mean(pred[4500:5000] == y.flatten()[4500:5000]%10) * 100))
print('Conjunto de entrenamiento de precisión para 10: {:f}'.format(np.mean(pred[0:500]     == y.flatten()[0:500]%10)     * 100))
