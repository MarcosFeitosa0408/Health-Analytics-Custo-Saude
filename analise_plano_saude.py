import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Carregar dados
df = pd.read_excel("dados/base_plano_de_saude.xlsx")

# Codificação de variáveis categóricas
le = LabelEncoder()
df["Sexo"] = le.fit_transform(df["Sexo"])
df["Fumante"] = le.fit_transform(df["Fumante"])
df["Região"] = le.fit_transform(df["Região"])

# Separar variáveis
X = df.drop("Custo_Saude", axis=1)
y = df["Custo_Saude"]

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regressão Linear
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Métricas
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R² Score: {r2}")
print(f"MSE: {mse}")

# Gráfico
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Valor Real")
plt.ylabel("Valor Previsto")
plt.title("Regressão Linear: Valor Real vs Previsto")
plt.grid(True)
plt.tight_layout()
plt.savefig("imagens/regressao_custo_saude.png")
