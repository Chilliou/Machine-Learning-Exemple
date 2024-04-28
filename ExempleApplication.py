import numpy as np
from tensorflow import keras  # Updated import for TensorFlow 2.x

model = keras.Sequential()

model.add(keras.layers.Dense(units=3, input_shape=[1]))  

model.add(keras.layers.Dense(units=64))
model.add(keras.layers.Dense(units=64))
model.add(keras.layers.Dense(units=64))

model.add(keras.layers.Dense(units=1))

entree = np.array([4+4, 5+5, 6+6, 7+7, 8+8, 9+9])
sortie = np.array([8, 15, 24, 35, 48, 56])


model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x=entree, y=sortie, epochs=1000)

print("Alors la solution de l'égnime pour 10+10 est : %s", str(model.predict(np.array([10+10]))))
