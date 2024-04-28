# Machine-Learning-Exemple
## Prérequis
- Python
- Tensorflow
- Keras
## Explications du programme
Voici une image de réseau représentatif du code programme principal pour faire le parallèle avec notre code  
```python
#input layer
model.add(keras.layers.Dense(units=3, input_shape=[1]))  

#hidden layer 1
model.add(keras.layers.Dense(units=64))
#hidden layer 2
model.add(keras.layers.Dense(units=64))

#output layer
model.add(keras.layers.Dense(units=1))
```
![Reseau de neurone](https://www.lebigdata.fr/wp-content/uploads/2019/04/reseau-de-neurones-fonctionnement.jpg)
Source de l'image: https://www.lebigdata.fr/reseau-de-neurones-artificiels-definition   
  
Nous allons en entrée fourir une suite de nomre allant de 1 à 5 et en sortie nous allons fournir leur addiction par eux-mêmes  

```python
entree = np.array([1, 2, 3, 4, 5])
sortie = np.array([2, 4, 6, 8, 10])
```
Suite a la compilation du réseau de neurones et a son entrainement nous seront capable de intérroger   
Tout cela afin qui nous fournisse une prédiction sur le chiffre que l'on lui donnes  
  
Example: 10000   
Le programme renvoie une aproximation tel que 20000.4  
## Cas utilisation
Nous pouvons avoir une exemple d'utilisation très simple commme trouver la solution a une égnime très dure
![egnime](https://github.com/Chilliou/Machine-Learning-Exemple/assets/25181715/37b723bf-10d7-4884-92de-6a73d5beda1b)
Source de l'image : https://tidudi.fr/enigme-suite-numerique/  
  
Nous donnons les donnée en entrée commme l'égnime avec les mêmes résultats
```python
entree = np.array([4+4, 5+5, 6+6, 7+7, 8+8])
sortie = np.array([8, 15, 24, 35, 48])
```
Puis nous demandons les résultats au programme
```python
print("Alors la solution de l'égnime pour 9+9 est : %s", str(model.predict(np.array([9+9]))))
```
Qui nous donnera le résultat 56 qui est le bon étant donnée que l'égnime consuite à faire 4x2 = 8, 5x3= 15, etc
## Importance des données
Pou répondre simplement à cette question nous allons reprendre l'égnime vu précédemment   
Lorsque on lui donne ceci
```python
entree = np.array([4+4, 5+5, 6+6, 7+7, 8+8])
sortie = np.array([8, 15, 24, 35, 48])
```
Notre réseau de neurones stagne très rapidement à un loss de 2.8 donc il n'est plus capable d'améliorer ses résultats  
Cepandant lorsque l'on lui fournie une valeur en plus 
```python
entree = np.array([4+4, 5+5, 6+6, 7+7, 8+8, 9+9])
sortie = np.array([8, 15, 24, 35, 48, 56])
```
Là notre programme arrive à s'améliorer encore plus en réduisant son champs d'erreurs avec une loss à 2.33  
Voici donc pourquoi il faut fournir un maximum de donnée et si possible précise afin d'avoir le réseau de neurones le plus performant possible

