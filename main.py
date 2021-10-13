import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Adaline")

class Adaline(object):
    # Constructor de Adaline para inicializar la taza de aprendizaje, epochs y el generador de aleatoriedad
    def __init__(self, eta=0.1, n_iter=50, randState=1):
        self.eta = eta
        self.n_iter = n_iter
        self.randState = randState

    # Función de aprendizaje
    def fit(self, X, y):
        rgen = np.random.RandomState(self.randState)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1]) # Generacion números random para todas las x + x0
        self.cost_ = []
        for i in range(self.n_iter):
            net_input = self.netInput(X) # Suma pesada, wx + w0
            output = self.activation(net_input) # La salida relevante en el Adaline es simplemente la activación lineal
            errors = (y-output) # El error es el valor real menos el valor que sale de la suma ponderada
            self.w_[1:] += self.eta * X.T.dot(errors) # La actualizacion de los pesos se realiza con el dataset completo
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0 # Costo definido como la sumatoria de error cuadrático
            self.cost_.append(cost)
        return self

    # Función que realiza la suma ponderada
    def netInput(self, x):
        return np.dot(x, self.w_[1:]) +self.w_[0] # Suma ponderada como el producto escalar entre w y x + w0

    # Activación lineal es la identidad
    def activation(self,x): #
        return x

    # Función que realiza la predicción
    def predict(self, x):
        return np.where(self.activation((self.netInput(x)))>=0.0, -1, 1)


# Importando dataset Iris con pandas
df = pd.read_csv('https://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data',header=None, encoding='utf-8')
# Extrearemos solamente dos etiquetas (flores setosa y versicolor), son las 100 primeras muestras.
# Sólo trabajaremos con la  las características longitud sepal y longitudo del pétalo (1era y 3era columna)
y= df.iloc[0:100,4].values
# La clasificación consistirá en setosa -1 y versicolor 1
y=np.where(y=='Iris-setosa', -1,1)
x= df.iloc[0:100,[0, 2]].values # Coger sólo la primera y tercera columna

# Visualización errores con dos neuronas con distinta eta

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = Adaline(n_iter=10, eta=0.01).fit(x, y)
ax[0].plot(range(1, len(ada1.cost_) + 1),np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = Adaline(n_iter=10, eta=0.0001).fit(x, y)
ax[1].plot(range(1, len(ada2.cost_) + 1),
ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.savefig("errorsAdaline.png")
#plt.show()
plt.close()

# Aplicar estandarización para mejorar la optimización
# Se normalizan las propiedades aplicando la media y la desviacion típica
X_std = np.copy(x)
X_std[:,0] = (x[:,0] - x[:,0].mean()) / x[:,0].std()
X_std[:,1] = (x[:,1] - x[:,1].mean()) / x[:,1].std()

ada_gd = Adaline(n_iter=15, eta=0.01)
ada_gd.fit(X_std, y)
plt.scatter(X_std[:50,0], X_std[:50,1], color='red', marker='o', label='setosa (-1)') # Muestras de setosa con las dos columnas
plt.scatter(X_std[50:100,0], X_std[50:100,1], color='blue', marker='x', label='versicolor (1)') # Muestras de versicolor con las dos columnas
plt.xlabel('longitud sepal (cm) (normalizado)')
plt.ylabel('longitud de pétalo (cm) (normalizado)')
plt.legend(loc='upper left')
# Visualizacion de la clasificación
sepal_min, sepal_max = X_std[:, 0].min() - 1, X_std[:, 0].max() + 1 # Coger valores maximos y minimos de la longitud sepal
petalo_min, petalo_max = X_std[:, 1].min() - 1, X_std[:, 1].max() + 1  # Coger valores maximos y minimos de la longitud pétalo
# Se crea una matriz de puntos para ambas propiedades
valSepal, valPetalo = np.meshgrid(np.arange(sepal_min, sepal_max, 0.02),
                                  np.arange(petalo_min, petalo_max, 0.02))
# Se clasifican los valores (1 ó -1)
Z = ada_gd.predict(np.array([valSepal.ravel(), valPetalo.ravel()]).T)
Z = Z.reshape(valSepal.shape)

plt.contourf(valSepal, valPetalo, Z, alpha=0.3)
plt.xlim(valSepal.min(), valSepal.max())
plt.ylim(valPetalo.min(), valPetalo.max())
#plt.show()
plt.savefig("adalineClassificationStd")
plt.close()

plt.plot(range(1, len(ada_gd.cost_) + 1),ada_gd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.tight_layout()
plt.title("Errors with standardized Adaline", pad=-20)
plt.savefig("adalineErrorsStd")
plt.close()