import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

columns = ["Mosha", "Gjinia", "Lloji i dhimbjes në gjoks", "Presioni i gjakut në pushim", "Kolesteroli në serum",
           "Sheqeri në gjak pas agjërimit", "Rezultatet elektrografike në pushim (Restecg)", "Rrahjet maksimale të zemrës",
           "Angina e shkaktuar nga ushtrimet fizike (Exang)", "Oldpeak - ST", "Shkalla", "Numri i enëve kryesore",
           "Lloji i defektit (Thal)", "Target"]

data_path = 'dataset_pacientet.csv'
logger.info("Duke ngarkuar datasetin nga %s", data_path)
df = pd.read_csv(data_path, header=None, names=columns)

df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric)
df.fillna(df.mean(), inplace=True)

df['Target'] = df['Target'].apply(lambda x: 0 if x == 0 else 1)

X = df.drop(columns=['Target'])
y = df['Target']

scaler = StandardScaler()
X = scaler.fit_transform(X)

def create_ann(input_dim, layers, neurons, activation='relu', dropout_rate=0.2, output_activation='sigmoid'):
    logger.info("Duke krijuar model ANN me %d shtresa dhe %d neurone për shtresë", layers, neurons)
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(neurons, activation=activation))
    model.add(Dropout(dropout_rate))
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation=output_activation))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

POPULATION_SIZE = 20 
GENERATIONS = 10
CXPB, MUTPB = 0.7, 0.3

def create_individual():
    return [random.randint(1, 2), 
            random.randint(5, 30)]  

def evaluate_individual(individual):
    layers, neurons = individual
    logger.info("Duke vlerësuar individin me %d shtresa dhe %d neurone", layers, neurons)
    model = create_ann(X.shape[1], layers, neurons)

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracy_scores = []
    
    for train_idx, val_idx in kfold.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train, epochs=5, batch_size=10, verbose=0) 
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        accuracy_scores.append(accuracy)
    
    avg_accuracy = np.mean(accuracy_scores)
    logger.info("Vlerësimi i individit përfundoi me saktësi mesatare: %f", avg_accuracy)
    return avg_accuracy,

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[1, 5], up=[3, 50], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_ga():
    logger.info("Duke filluar algoritmin gjenetik")
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("mesatarja", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=GENERATIONS, 
                                       stats=stats, halloffame=hof, verbose=True)
    logger.info("Algoritmi gjenetik përfundoi")
    return pop, logbook, hof

pop, logbook, hof = run_ga()

gen = logbook.select("gen")
fit_max = logbook.select("max")
fit_avg = logbook.select("avg")

plt.figure(figsize=(10, 6))
plt.plot(gen, fit_max, label='Fitness Maksimal')
plt.plot(gen, fit_avg, label='Fitness Mesatar')
plt.xlabel('Gjenerata')
plt.ylabel('Fitness')
plt.title('Progresi i GA')
plt.legend()
plt.grid(True)
plt.show()

best_individual = hof[0]
best_layers, best_neurons = best_individual
logger.info("Individi më i mirë u gjet: %d shtresa dhe %d neurone", best_layers, best_neurons)

logger.info("Duke trajnuar modelin përfundimtar me individin më të mirë")
final_model = create_ann(X.shape[1], best_layers, best_neurons)
history = final_model.fit(X, y, epochs=20, batch_size=10, validation_split=0.2, verbose=1)

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Saktësia e Trajnimit')
plt.plot(history.history['val_accuracy'], label='Saktësia e Validimit')
plt.title('Saktesia e Modelit')
plt.ylabel('Saktesia')
plt.xlabel('Epoka')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Humbja e Trajnimit')
plt.plot(history.history['val_loss'], label='Humbja e Validimit')
plt.title('Humbja e Modelit')
plt.ylabel('Humbja')
plt.xlabel('Epoka')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

loss, accuracy = final_model.evaluate(X_test, y_test, verbose=1)
logger.info("Saktesia e testit te modelit perfundimtar: %f", accuracy)

from tensorflow.keras.utils import plot_model
plot_model(final_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

img = plt.imread('model_plot.png')
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.show()
