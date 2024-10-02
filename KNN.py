import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.combine import SMOTEENN  # Hybrid sampling technique
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

columns = ["Mosha", "Gjinia", "Lloji i dhimbjes në gjoks", "Presioni i gjakut në pushim", "Kolesteroli në serum",
           "Sheqeri në gjak pas agjërimit", "Rezultatet elektrografike në pushim (Restecg)", "Rrahjet maksimale të zemrës",
           "Angina e shkaktuar nga ushtrimet fizike (Exang)", "Oldpeak - ST", "Shkalla", "Numri i enëve kryesore",
           "Lloji i defektit (Thal)", "Target"]

data_path = 'dataset_pacientet.csv'
df = pd.read_csv(data_path, header=None, names=columns)

df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric)
df.fillna(df.mean(), inplace=True)

df['Target'] = df['Target'].apply(lambda x: 0 if x == 0 else 1)

X = df.drop(columns=['Target'])
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
logger.info("Të dhënat u ndanë në set trajnimi dhe testimi")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
logger.info("Veçoritë u standardizuan")

logger.info("Balancimi i të dhënave duke përdorur SMOTEENN")
smoteenn = SMOTEENN(random_state=42)
X_train_bal, y_train_bal = smoteenn.fit_resample(X_train, y_train)

param_grid_knn = {
    'n_neighbors': list(range(1, 21)),  # Range of n_neighbors
    'weights': ['uniform', 'distance'],  # Weighting strategies
    'metric': ['euclidean', 'manhattan']  # Distance metrics
}

skf = StratifiedKFold(n_splits=5)
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, refit=True, cv=skf, n_jobs=-1, verbose=0)

logger.info("Trajnimi i modelit KNN")
grid_knn.fit(X_train_bal, y_train_bal)

best_params = grid_knn.best_params_
logger.info(f"Parametrat më të mirë të gjetur: {best_params}")

y_pred = grid_knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
logger.info(f"Saktësia e modelit: {accuracy:.2f}")
logger.info(f"F1-Score: {f1:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Matrica e konfuzionit")
plt.ylabel('Aktual')
plt.xlabel('Parashikuar')
plt.show()

classification_rep = classification_report(y_test, y_pred)
logger.info("Raporti i klasifikimit:\n%s", classification_rep)

cv_scores = cross_val_score(grid_knn, X_train_bal, y_train_bal, cv=skf, n_jobs=-1)
logger.info(f"Saktësia e Cross-Validation: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")

logger.info("Përfundoi trajnimi dhe vlerësimi")

joblib.dump(grid_knn, 'knn_model_improved.pkl')
logger.info("Modeli përfundimtar u ruajt")


