# Otomatik Hiperparametre Optimizasyonu

Kaba kuvvet arama (`GridSearchCV`) ile modern Tree-structured Parzen Estimator (TPE) Bayesyen optimizasyon çerçevesi (`Optuna`) arasındaki verimlilik, arama uzayı ve başarım kıyaslaması.

## İlgili Notebook
- [hyperparameter-tuning-gridsearch-optuna.ipynb](hyperparameter-tuning-gridsearch-optuna.ipynb)

## Temel Kütüphaneler
- `sklearn.model_selection.GridSearchCV`
- `optuna.create_study`, `TPESampler`
- `cross_val_score`
