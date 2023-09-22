Car accidents - model and dashboard.

First of all, install the dependencies in your environment. In your terminal, run pip install -r requirements.txt or in your notebook (Jupyter) !pip install -r requirements.txt. Python version: 3.10.11

Now, you can run the dashboard using: streamlit run dashboard_car_accident.py

And you can run the page to load the model and predict a specific datapoint using: streamlit run predictor_dash.py

My prediction approach is based on the idea of MLP ( Multi-Layer Perceptron ) implemented using Keras Sequential.

I trained and deployed two models: one using CHI feature selection (and reducing the features for 3) and the other one using Logistic Regresseion Features Importance.

This way We can predict the number of accidents absed on:

1. State, Region, Time of Day, Year and Type of accident.
2. State, Year and Highway.

We also have:

--- Notebooks to deploy our models: model_regression.ipynb and model_less_feature.ipynb.
--- The models in pkl and h5: mlp_model and mlp_model_features_reduced.
--- A notebook of data exploration: It's a draft and It's pretty messy. : )
--- A notebook with some statistical inferences: inferences.ipynb.
--- Files with our dummy datasets structure (columns): colunas and colunas_reduzidas.
--- The processed data (combination of dataset from 2007 to 2023) is (as csv) in a zip file, but also in a pickle (data_dashboard_pkl)
--- The dependencies: requirements.txt
--- Our streamlit pages: predictor_dash.py and dashboard_car_accident.py

    Hope you enjoy!!

    Érica Ferreira.
