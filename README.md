# RetailML-API

## Descriere:
RetailML API este o platformă integrată de algoritmi Machine Learning aplicați pe setul de date Online Retail II (peste 1.000.000 de tranzacții ale unui retailer online din Regatul Unit), expusă printr-un REST API construit în Flask. Platforma acoperă patru tipuri de analiză: regresie, clasificare, serii de timp și clusterizare, oferind predicții instant sub formă de JSON.

## Module ML disponibile:
- Regresie - predicția valorii totale a unei tranzacții (Quantity × Price), folosind Linear Regression și Random Forest Regressor
- Clasificare - detectarea tranzacțiilor anulate/retururi, folosind Regresie Logistică și Decision Tree
- Serii de timp - prognoza vânzărilor zilnice pe următoarele 30 de zile, folosind ARIMA
- Clusterizare - segmentarea clienților prin analiza RFM (Recency, Frequency, Monetary) cu K-Means și DBSCAN

## Caracteristici:
- Pipeline complet de preprocesare (curățare date, feature engineering, standardizare, train/test split)
- Modele antrenate și salvate (.pkl) pentru reutilizare fără reantrenare
- REST API în Flask cu endpoint-uri dedicate pentru fiecare tip de predicție
- Răspunsuri structurate în format JSON
- Evaluare completă a performanței (R², MAE, RMSE, precizie, recall, matrice de confuzie)

## Tehnologii folosite:
- Python
- scikit-learn
- statsmodels (ARIMA)
- Flask (REST API)
- Pandas / NumPy
- Spyder
