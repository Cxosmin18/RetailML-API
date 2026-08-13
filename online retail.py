import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================
# Calea catre fisierul excel
df = pd.read_csv("C:\\Python\\OnlineRetail\\online_retail_II.csv")

print("=" * 60)
print("INFORMATII GENERALE DESPRE DATASET")
print("=" * 60)
print(f"Numar randuri:   {df.shape[0]:,}")
print(f"Numar coloane:   {df.shape[1]}")
print(f"\nColoane si tipuri:")
print(df.dtypes)


# ============================================================
# 2.2 EXPLORAREA DATELOR
# ============================================================

# ------------------------------------------------------------
# GRAFIC 1: Valori lipsa
# ------------------------------------------------------------
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Valori lipsa": missing, "Procent (%)": missing_pct})
missing_df = missing_df[missing_df["Valori lipsa"] > 0]

print("\n" + "=" * 60)
print("VALORI LIPSA")
print("=" * 60)
print(missing_df)

plt.figure(figsize=(8, 4))
bars = plt.bar(missing_df.index, missing_df["Procent (%)"], color=["#E74C3C", "#E67E22"])
plt.title("Procentul valorilor lipsă pe atribute", fontsize=13, fontweight="bold")
plt.ylabel("Procent valori lipsă (%)")
plt.xlabel("Atribut")
for bar, val in zip(bars, missing_df["Procent (%)"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f"{val}%", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig("grafic1_valori_lipsa.png", dpi=150)
plt.show()
print(">> Salvat: grafic1_valori_lipsa.png")


# ------------------------------------------------------------
# GRAFIC 2: Statistici descriptive (Quantity si Price)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("STATISTICI DESCRIPTIVE")
print("=" * 60)
print(df[["Quantity", "Price"]].describe().round(2))


# ------------------------------------------------------------
# GRAFIC 3: Distributia Quantity
# ------------------------------------------------------------
q_filtrat = df[(df["Quantity"] > 0) & (df["Quantity"] < 100)]

plt.figure(figsize=(8, 4))
plt.hist(q_filtrat["Quantity"], bins=50, color="#3498DB", edgecolor="white")
plt.title("Distribuția cantității comandate (Quantity)", fontsize=13, fontweight="bold")
plt.xlabel("Quantity (cantitate)")
plt.ylabel("Frecvență")
plt.tight_layout()
plt.savefig("grafic2_distributie_quantity.png", dpi=150)
plt.show()
print(">> Salvat: grafic2_distributie_quantity.png")


# ------------------------------------------------------------
# GRAFIC 4: Distributia Price
# ------------------------------------------------------------
p_filtrat = df[(df["Price"] > 0) & (df["Price"] < 20)]

plt.figure(figsize=(8, 4))
plt.hist(p_filtrat["Price"], bins=50, color="#2ECC71", edgecolor="white")
plt.title("Distribuția prețului unitar (Price)", fontsize=13, fontweight="bold")
plt.xlabel("Price (£)")
plt.ylabel("Frecvență")
plt.tight_layout()
plt.savefig("grafic3_distributie_price.png", dpi=150)
plt.show()
print(">> Salvat: grafic3_distributie_price.png")


# ------------------------------------------------------------
# GRAFIC 5: Top 10 tari dupa numar de tranzactii
# ------------------------------------------------------------
top_tari = df["Country"].value_counts().head(10)

plt.figure(figsize=(10, 5))
bars = plt.barh(top_tari.index[::-1], top_tari.values[::-1], color="#9B59B6")
plt.title("Top 10 țări după numărul de tranzacții", fontsize=13, fontweight="bold")
plt.xlabel("Număr tranzacții")
for bar, val in zip(bars, top_tari.values[::-1]):
    plt.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
             f"{val:,}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("grafic4_top_tari.png", dpi=150)
plt.show()
print(">> Salvat: grafic4_top_tari.png")


# ------------------------------------------------------------
# GRAFIC 6: Vanzari lunare (serii de timp - preview)
# ------------------------------------------------------------
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df_valid = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()
df_valid["TotalValue"] = df_valid["Quantity"] * df_valid["Price"]
df_valid["YearMonth"] = df_valid["InvoiceDate"].dt.to_period("M")

vanzari_lunare = df_valid.groupby("YearMonth")["TotalValue"].sum()

plt.figure(figsize=(12, 5))
plt.plot(vanzari_lunare.index.astype(str), vanzari_lunare.values,
         marker="o", color="#E74C3C", linewidth=2)
plt.title("Evoluția vânzărilor lunare (valoare totală £)", fontsize=13, fontweight="bold")
plt.xlabel("Luna")
plt.ylabel("Valoare totală (£)")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("grafic5_vanzari_lunare.png", dpi=150)
plt.show()
print(">> Salvat: grafic5_vanzari_lunare.png")


# ------------------------------------------------------------
# GRAFIC 7: Heatmap corelatii (pe datele numerice)
# ------------------------------------------------------------
df_valid["TotalValue"] = df_valid["Quantity"] * df_valid["Price"]
corr_matrix = df_valid[["Quantity", "Price", "TotalValue"]].corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues",
            linewidths=0.5, square=True)
plt.title("Heatmap corelații între variabilele numerice", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("grafic6_heatmap_corelatii.png", dpi=150)
plt.show()
print(">> Salvat: grafic6_heatmap_corelatii.png")


# ------------------------------------------------------------
# GRAFIC 8: Boxplot Quantity si Price (outlieri)
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

df_valid.boxplot(column="Quantity", ax=axes[0])
axes[0].set_title("Boxplot - Quantity", fontweight="bold")
axes[0].set_ylabel("Cantitate")

df_valid.boxplot(column="Price", ax=axes[1])
axes[1].set_title("Boxplot - Price", fontweight="bold")
axes[1].set_ylabel("Preț (£)")

plt.suptitle("Distribuția și outlierii variabilelor numerice", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("grafic7_boxplot.png", dpi=150)
plt.show()
print(">> Salvat: grafic7_boxplot.png")


# ============================================================
# 2.3 PREPROCESARE
# ============================================================
print("\n" + "=" * 60)
print("PREPROCESARE DATE")
print("=" * 60)

# Pasul 1: Eliminam valorile lipsa
df_clean = df.dropna(subset=["Customer ID", "Description"])
print(f"Dupa eliminarea valorilor lipsa: {df_clean.shape[0]:,} randuri")

# Pasul 2: Eliminam tranzactiile anulate (Invoice incepe cu 'C')
df_clean = df_clean[~df_clean["Invoice"].astype(str).str.startswith("C")]
print(f"Dupa eliminarea anularilor: {df_clean.shape[0]:,} randuri")

# Pasul 3: Eliminam Quantity si Price negative sau zero
df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["Price"] > 0)]
print(f"Dupa eliminarea valorilor negative: {df_clean.shape[0]:,} randuri")

# Pasul 4: Cream coloana TotalValue
df_clean = df_clean.copy()
df_clean["TotalValue"] = df_clean["Quantity"] * df_clean["Price"]

# Pasul 5: Cream coloana IsHighValue (clasificare) - 1 daca TotalValue > mediana
mediana = df_clean["TotalValue"].median()
df_clean["IsHighValue"] = (df_clean["TotalValue"] > mediana).astype(int)
print(f"\nMediana TotalValue: £{mediana:.2f}")
print(f"Tranzactii IsHighValue=1: {df_clean['IsHighValue'].sum():,}")
print(f"Tranzactii IsHighValue=0: {(df_clean['IsHighValue']==0).sum():,}")

# Pasul 6: Encodare variabile categoriale (Country)
top_countries = df_clean["Country"].value_counts().head(10).index
df_clean["Country_encoded"] = df_clean["Country"].apply(
    lambda x: x if x in top_countries else "Other"
)
df_clean = pd.get_dummies(df_clean, columns=["Country_encoded"], drop_first=True)
print(f"\nDupa encodare Country: {df_clean.shape[1]} coloane totale")

# Pasul 7: Definim x si y pentru modele
feature_cols = ["Quantity", "Price"] + [c for c in df_clean.columns if c.startswith("Country_encoded_")]

X = df_clean[feature_cols]
y_regresie = df_clean["TotalValue"]
y_clasificare = df_clean["IsHighValue"]

# Pasul 8: Standardizare
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\nFeatures pentru modele: {feature_cols}")

# Pasul 9: Train/Test split (80/20)
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X_scaled, y_regresie, test_size=0.2, random_state=42)

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_scaled, y_clasificare, test_size=0.2, random_state=42)

print(f"\nTrain size: {X_train.shape[0]:,} randuri")
print(f"Test size:  {X_test.shape[0]:,} randuri")

# Salvam datele curatate pentru capitolele urmatoare
df_clean.to_csv("retail_clean.csv", index=False)
print("\n>> Date curatate salvate in: retail_clean.csv")
print("\n>> EDA si preprocesare finalizate!")


# ============================================================
# 3.1 REGRESIE - IMPLEMENTARE ȘI EVALUARE
# ============================================================
print("\n" + "=" * 60)
print("CAPITOLUL 3.1: REGRESIE (PREZICERE TOTAL VALUE)")
print("=" * 60)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Din cauza volumului mare de date (~644k rânduri), Random Forest poate fi extrem de lent.
# Pentru eficienta in faza de proiect, optimizam hiperparametrii (n_estimators=50, max_depth=10, n_jobs=-1 ca sa foloseasca toate nucleele procesorului)
print("--> Antrenare Regresie Liniară...")
# Definirea si antrenarea modelelor
lr_model = LinearRegression()
lr_model.fit(X_train, y_train_reg)

rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train_reg)

# Predictii pe setul de test
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Functie pentru calcularea si afisarea metricilor
def evalueaza_regresie(nume_model, y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, y_pred)
    print(f"\nMetrici pentru {nume_model}:")
    print(f"  - MAE:  {mae:.4f} £")
    print(f"  - MSE:  {mse:.4f}")
    print(f"  - RMSE: {rmse:.4f} £")
    print(f"  - R²:   {r2:.4f}")
    return [mae, mse, rmse, r2]

metrics_lr = evalueaza_regresie("Regresie Liniara", y_test_reg, y_pred_lr)
metrics_rf = evalueaza_regresie("Random Forest Regressor", y_test_reg, y_pred_rf)

# ------------------------------------------------------------
# GRAFIC 3.1.4: Valori Reale vs. Predicții (Random Forest)
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))
# Luam un eaantion de 500 de puncte pentru ca graficul sa fie lizibil
plt.scatter(y_test_reg.iloc[:500], y_pred_rf[:500], alpha=0.6, color='#2ECC71', label='Predicții RF')
plt.plot([y_test_reg.iloc[:500].min(), y_test_reg.iloc[:500].max()], 
         [y_test_reg.iloc[:500].min(), y_test_reg.iloc[:500].max()], 
         'r--', lw=2, label='Linia Perfectă (Ideal)')
plt.title("Valori Reale vs. Valori Prezise (Random Forest)", fontsize=13, fontweight='bold')
plt.xlabel("Valori Reale (TotalValue £)")
plt.ylabel("Valori Prezise (TotalValue £)")
plt.legend()
plt.tight_layout()
plt.savefig("regresie_reale_vs_prezise.png", dpi=150)
plt.show()
print("\n>> Salvat: regresie_reale_vs_prezise.png")

# ------------------------------------------------------------
# GRAFIC 3.1.4 (F): Feature Importance (Random Forest)
# ------------------------------------------------------------
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features_ordonat = [feature_cols[i] for i in indices]

plt.figure(figsize=(10, 5))
plt.bar(range(X_train.shape[1]), importances[indices], color='#34495E', align='center')
plt.xticks(range(X_train.shape[1]), features_ordonat, rotation=45, ha='right')
plt.title("Importanța Caracteristicilor în Modelul Random Forest", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("regresie_feature_importance.png", dpi=150)
plt.show()
print(">> Salvat: regresie_feature_importance.png")


# ============================================================
# 3.2 CLASIFICARE - IMPLEMENTARE SI EVALUARE (ANULATE VS NORMALE)
# ============================================================
print("\n" + "=" * 60)
print("CAPITOLUL 3.2: CLASIFICARE (PREZICERE TRANZACȚII ANULATE)")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# 1. Pregatim datele special pentru problema de detectare a anularilor
df_cls = df.dropna(subset=["Customer ID", "Description"]).copy()

# Generam eticheta: 1 daca factura incepe cu 'C' (Anulata), 0 în caz contrar (Normala)
df_cls["IsCanceled"] = df_cls["Invoice"].astype(str).str.startswith("C").astype(int)

# Pentru a nu trisa modelul, scoatem Quantity si Price din formatul lor negativ 
# si folosim valorile absolute, deoarece in productie (REST API) utilizatorul va introduce valori pozitive
df_cls["Quantity"] = df_cls["Quantity"].abs()
df_cls["Price"] = df_cls["Price"].abs()

# Encodam tara la fel ca inainte
df_cls["Country_encoded"] = df_cls["Country"].apply(lambda x: x if x in top_countries else "Other")
df_cls = pd.get_dummies(df_cls, columns=["Country_encoded"], drop_first=True)

# Definim x si y pentru clasificare
feature_cols_cls = ["Quantity", "Price"] + [c for c in df_cls.columns if c.startswith("Country_encoded_")]
X_c = df_cls[feature_cols_cls]
y_c = df_cls["IsCanceled"]

# Impartire Train/Test 80/20
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

# Standardizare
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

# 2. Instantierea si configurarea modelelor
log_reg = LogisticRegression(max_iter=1000, random_state=42)
dt_clf = DecisionTreeClassifier(max_depth=6, random_state=42)

# Antrenarea modelelor pe datele de test
log_reg.fit(X_train_c_scaled, y_train_c)
dt_clf.fit(X_train_c_scaled, y_train_c)

# Generarea predictiilor pe date noi
y_pred_log = log_reg.predict(X_test_c_scaled)
y_pred_dt = dt_clf.predict(X_test_c_scaled)

# 3. Functie de evaluare performante
def evalueaza_clasificare(nume_model, y_real, y_pred):
    acc = accuracy_score(y_real, y_pred)
    prec = precision_score(y_real, y_pred, zero_division=0)
    rec = recall_score(y_real, y_pred, zero_division=0)
    f1 = f1_score(y_real, y_pred, zero_division=0)
    
    print(f"\nMetrici pentru {nume_model}:")
    print(f"  - Accuracy:  {acc:.4f}")
    print(f"  - Precision: {prec:.4f}")
    print(f"  - Recall:    {rec:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")
    return confusion_matrix(y_real, y_pred)

cm_log = evalueaza_clasificare("Regresie Logistică", y_test_c, y_pred_log)
cm_dt = evalueaza_clasificare("Decision Tree Classifier", y_test_c, y_pred_dt)

# ------------------------------------------------------------
# GRAFIC 3.2.4: Matricea de Confuzie (Decision Tree)
# ------------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Purples", cbar=False,
            xticklabels=["Normale (0)", "Anulate (1)"],
            yticklabels=["Normale (0)", "Anulate (1)"])
plt.title("Matricea de Confuzie - Decision Tree", fontsize=13, fontweight='bold')
plt.ylabel("Clasa Reală")
plt.xlabel("Clasa Prezisă")
plt.tight_layout()
plt.savefig("clasificare_matrice_confuzie.png", dpi=150)
plt.show()
print("\n>> Salvat: clasificare_matrice_confuzie.png")

# ------------------------------------------------------------
# GRAFIC 3.2.4 (F): Feature Importance (Decision Tree)
# ------------------------------------------------------------
importances_c = dt_clf.feature_importances_
indices_c = np.argsort(importances_c)[::-1]
features_ordonat_c = [feature_cols_cls[i] for i in indices_c]

plt.figure(figsize=(10, 5))
plt.bar(range(X_train_c.shape[1]), importances_c[indices_c], color='#8E44AD', align='center')
plt.xticks(range(X_train_c.shape[1]), features_ordonat_c, rotation=45, ha='right')
plt.title("Importanța Caracteristicilor în Modelul Decision Tree", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("clasificare_feature_importance.png", dpi=150)
plt.show()
print(">> Salvat: clasificare_feature_importance.png")


# ============================================================
# 3.3 SERII DE TIMP - PROGNOZA VANZARILOR
# ============================================================
print("\n" + "=" * 60)
print("CAPITOLUL 3.3: SERII DE TIMP (PROGNOZA VÂNZĂRILOR ZILNICE)")
print("=" * 60)

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Pregatim datele: adunam vanzarile pe fiecare zi
df_ts = df.dropna(subset=["Customer ID"]).copy()
df_ts["InvoiceDate"] = pd.to_datetime(df_ts["InvoiceDate"])
df_ts = df_ts[(df_ts["Quantity"] > 0) & (df_ts["Price"] > 0)]
df_ts["TotalValue"] = df_ts["Quantity"] * df_ts["Price"]

# Grupam dupa data (an-luna-zi) si calculam suma vanzarilor
vanzari_zilnice = df_ts.groupby(df_ts["InvoiceDate"].dt.date)["TotalValue"].sum()

# Impartim datele: lasam ultimele 30 de zile pentru test, restul pentru antrenare
train_ts = vanzari_zilnice.iloc[:-30]
test_ts = vanzari_zilnice.iloc[-30:]

# 2. Antrenam modelul ARIMA
print("--> Antrenare model ARIMA...")
model_arima = ARIMA(train_ts, order=(5, 1, 0)) # p, d, q parametrii standard
model_arima_fit = model_arima.fit()

# Facem prognoza pentru cele 30 de zile de test
prognoza = model_arima_fit.forecast(steps=30)
prognoza.index = test_ts.index

# 3. Calculam erorile
mae_ts = mean_absolute_error(test_ts, prognoza)
rmse_ts = np.sqrt(mean_squared_error(test_ts, prognoza))

print(f"\nMetrici Serii de Timp (ARIMA):")
print(f"  - MAE (Eroarea medie zilnică): {mae_ts:.2f} £")
print(f"  - RMSE: {rmse_ts:.2f} £")

# ------------------------------------------------------------
# GRAFIC 3.3.3: Vanzari Reale vs. Prognoza
# ------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_ts.index[-60:], train_ts.values[-60:], label="Istoric ultimele 60 zile", color="#34495E")
plt.plot(test_ts.index, test_ts.values, label="Valori Reale (Test)", color="#2ECC71", linewidth=2)
plt.plot(prognoza.index, prognoza.values, label="Prognoză ARIMA", color="#E74C3C", linestyle="--", linewidth=2)
plt.title("Prognoza Vânzărilor Zilnice cu ARIMA", fontsize=13, fontweight='bold')
plt.xlabel("Dată")
plt.ylabel("Vânzări (£)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("timeseries_prognoza.png", dpi=150)
plt.show()
print("\n>> Salvat: timeseries_prognoza.png")


