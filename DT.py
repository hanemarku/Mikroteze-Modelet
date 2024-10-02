import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import plot_tree

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

columns = ["Mosha", "Gjinia", "Lloji i dhimbjes në gjoks", "Presioni i gjakut në pushim", "Kolesteroli në serum",
           "Sheqeri në gjak pas agjërimit", "Rezultatet elektrografike në pushim (Restecg)", "Rrahjet maksimale të zemrës",
           "Angina e shkaktuar nga ushtrimet fizike (Exang)", "Oldpeak - ST", "Shkalla", "Numri i enëve kryesore",
           "Lloji i defektit (Thal)", "Target"]

data_path = 'pacientet.csv'
df = pd.read_csv(data_path, header=None, names=columns)

df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric)  
df.fillna(df.mean(), inplace=True)

df['Target'] = df['Target'].apply(lambda x: 0 if x == 0 else 1)

X = df.drop(columns=['Target'])
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
logger.info("Veçoritë u standardizuan")

logger.info("Duke balancuar datasetin duke përdorur RandomOverSampler")
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

logger.info("Duke filluar optimizimin e hipërparametrave me Grid Search")
rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=1, scoring='accuracy')
grid_search.fit(X_train_bal, y_train_bal)

best_params = grid_search.best_params_
logger.info("Parametrat më të mirë të gjetur: %s", best_params)
best_model = grid_search.best_estimator_

logger.info("Duke trajnuar modelin përfundimtar Random Forest me parametrat më të mirë")
best_model.fit(X_train_bal, y_train_bal)

logger.info("Duke vlerësuar modelin")
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
logger.info("Saktësia: %f", accuracy)

print("Raporti i klasifikimit:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Matrica e konfuzionit")
plt.ylabel('Aktual')
plt.xlabel('Parashikuar')
plt.show()
plt.close()
logger.info("Duke vizualizuar një nga pemët e vendimeve")

logger.info("Duke vizualizuar një nga pemët e vendimeve")

plt.figure(figsize=(20,10))
plot_tree(
    best_model.estimators_[0], 
    filled=True, 
    feature_names=columns[:-1], 
    class_names=['0', '1'], 
    rounded=True,
    proportion=False  
)

plt.title("Pema e Vendimeve nga Random Forest")
plt.show()
import joblib
joblib.dump(best_model, 'random_forest_model.pkl')
logger.info("Modeli përfundimtar u ruajt")
