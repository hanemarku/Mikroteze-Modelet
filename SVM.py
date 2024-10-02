import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

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

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'poly']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

# Parametrat më të mirë
best_params = grid.best_params_
logger.info("Parametrat më të mirë u gjetën: %s", best_params)

logger.info("Duke trajnuar modelin SVM me parametrat më të mirë")
svm_model = SVC(**best_params)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
logger.info("Saktësia e modelit: %.2f%%", accuracy * 100)

print("Raporti i Klasifikimit:")
print(classification_report(y_test, y_pred))

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
    plt.xlabel('Komponenti 1')
    plt.ylabel('Komponenti 2')
    plt.title('Vija e Vendimeve SVM me veçoritë e PCA-Reduced')
    plt.show()

plot_decision_boundary(svm_model, X_test, y_test)
