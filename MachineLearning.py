import numpy as np
from tensorflow import keras  # Updated import for TensorFlow 2.x

# Permet de déclarer un reseau de neurone
model = keras.Sequential()

# Creation de la première couche, dite couche d'entrée
# Le nombre units correspond au nombre de neurones en entrée
# Input shape correspond à la donnée en entrée 
model.add(keras.layers.Dense(units=3, input_shape=[1]))  

# Creation des couches intermédaire, dite couches cachées
model.add(keras.layers.Dense(units=64))
model.add(keras.layers.Dense(units=64))
model.add(keras.layers.Dense(units=64))

# Creation de la couche finale, dite couche de sortie
model.add(keras.layers.Dense(units=1))


# Donnée aidant notre réseau de neurones à savoir pour quels
# entrée associé quels sorties
entree = np.array([1, 2, 3, 4, 5])
sortie = np.array([2, 4, 6, 8, 10])

# La fonction donnée à loss correspond à ce que le programme
# défini qu'une valeur de sorti est défini comme bonne
# lorsque le carré de écart à la moyenne est minimal
# Optimiser est une fonction servant à optimiser ^^
model.compile(loss='mean_squared_error', optimizer='adam')

# x correspond au entrée et y au sortie
# et le bomre epochs le nombre de cycle 
model.fit(x=entree, y=sortie, epochs=1000)


# Suite a affinage du modele on le test en fournissant
# un nombre et le programme prédict sont carré 
# comme les tableaux fournie en entré et sortie
while True:
    x = int(input('Nombre :'))
    print('Prédiction : ' + str(model.predict(np.array([x]))))