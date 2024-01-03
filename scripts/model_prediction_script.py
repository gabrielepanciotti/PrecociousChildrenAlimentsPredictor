
from tkinter import Y
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from IPython.display import display

# Caricamento dei dati
data = pd.read_csv('../data/new_dataset.csv', sep=';')  # Aggiorna con il tuo percorso

# Definizione delle colonne
peso_columns = ['Peso'] + [f'Peso_d-{i}' for i in range(1, 11)]
variazione_columns = [f'daily_weight_diff_d-{i}' for i in range(1, 11)]
nutrizione_columns = ['VolumeNE', 'ne_calkg', 'ne_chokg', 'ne_protkg', 'ne_lipkg', 'VolumeNP', 'np_calkg', 'np_chokg', 'np_protkg', 'np_lipkg'] 
nutrizione_storico_columns = [f'{col}_d-{i}' for col in nutrizione_columns for i in range(1, 11)]

selected_columns = ['age', 'Etagest'] + peso_columns + variazione_columns + nutrizione_columns + nutrizione_storico_columns
decimal_cols = ['Etagest', 'daily_weight_diff'] + peso_columns + variazione_columns + nutrizione_columns + nutrizione_storico_columns

# Conversione delle colonne in float
for col in decimal_cols:
    if data[col].dtype == 'object':
        data[col] = data[col].str.replace(',', '.').astype(float)
    else:
        data[col] = data[col].astype(float)

# Selezione delle colonne rilevanti e divisione dei dati

X = data[selected_columns]
display(X)
y = data['daily_weight_diff']
display(Y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizzazione dei dati
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Applicazione della PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Definizione della funzione per valutare i modelli
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mse, mae, r2

# Inizializzazione dei modelli
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Valutazione dei modelli con e senza PCA
results = []
predicted_results = []
for name, model in models.items():
    y_pred, mse, mae, r2 = evaluate_model(model, X_train, y_train, X_test, y_test)
    results.append((name, 'No PCA', mse, mae, r2))
    predicted_results.append(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Model': name}))
    y_pred, mse, mae, r2 = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    results.append((name, 'Scaler', mse, mae, r2))
    y_pred, mse_pca, mae_pca, r2_pca = evaluate_model(model, X_train_pca, y_train, X_test_pca, y_test)
    results.append((name, 'With PCA', mse_pca, mae_pca, r2_pca))

# Creazione della tabella riassuntiva
results_df = pd.DataFrame(results, columns=['Model', 'PCA', 'MSE', 'MAE', 'R^2'])
results_sorted = results_df.sort_values(by='MSE')
print(results_sorted)
results_sorted.to_csv('result.csv')

# Concatenazione dei risultati predetti in un unico DataFrame
predicted_df = pd.concat(predicted_results)

# Creazione della tabella di confronto tra peso previsto e reale
comparison_table = predicted_df[['Actual', 'Predicted', 'Model']]
print(comparison_table)
comparison_table.to_csv('prediction.csv')