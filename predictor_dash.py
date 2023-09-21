import streamlit as st
import joblib
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
tf.config.run_functions_eagerly(True)

# Load your model
model = joblib.load('mlp_model.pkl')
model_red = joblib.load('mlp_model_features_reduced.pkl')

# Define options for multiselects
# Replace with your actual options
tipo_acidente_options = ['Sada de leito carrovel', 'Atropelamento de Pedestre',
                         'Tombamento', 'Coliso frontal', 'Coliso transversal', 'Incndio',
                         'Coliso lateral mesmo sentido', 'Coliso traseira', 'Engavetamento',
                         'Coliso lateral sentido oposto', 'Derramamento de carga',
                         'Coliso com objeto', 'Eventos atpicos', 'Capotamento',
                         'Queda de ocupante de veculo', 'Atropelamento de Animal',
                         'Coliso lateral', 'Coliso com objeto esttico', 'Danos eventuais',
                         'Coliso com objeto em movimento', 'Unknown',
                         'Coliso com objeto fixo', 'Coliso com bicicleta',
                         'Queda de motocicleta / bicicleta / veculo', 'Coliso Transversal',
                         'Sada de Pista', 'Atropelamento de pessoa',
                         'Coliso com objeto mvel', 'Atropelamento de animal',
                         'Derramamento de Carga', 'Danos Eventuais']

uso_solo_options = ['Urbano', 'Rural', 'Unknown']
state_options = ['ES', 'SP', 'MT', 'PR', 'MG', 'BA', 'RJ', 'RS', 'SC', 'PI', 'GO',
                 'PE', 'PA', 'MS', 'MA', 'CE', 'AP', 'PB', 'SE', 'RO', 'RN', 'TO',
                 'RR', 'DF', 'AL', 'AC', 'AM', 'Unknown']

classificacao_acidente_options = [
    'Com Vtimas Feridas', 'Com Vtimas Fatais', 'Sem Vtimas', 'Unknown',
    'Ignorado']

col1, col2 = st.columns(2)

with col1:
    # Define a form layout
    st.subheader("Predict # of Accidents based on: ")
    st.markdown("<style>color: skyblue; text-align: center; font-size: 10px;</style>",
                unsafe_allow_html=True)

    with st.form("accident_prediction_form"):
        # BR (Number stored as string)
        br = st.text_input('BR (Number)', '')

        # Tipo Acidente (Multiselect)
        tipo_acidente = st.multiselect('Tipo Acidente', tipo_acidente_options)
        tipo_acidente = tipo_acidente[0] if tipo_acidente else None

        # Uso Solo (Multiselect)
        uso_solo = st.multiselect('Uso Solo', uso_solo_options)
        uso_solo = uso_solo[0] if uso_solo else None

        # State (Multiselect)
        state = st.multiselect('State', state_options)
        regional = "SPRF-" + "-".join(state) if state else None

        # Classificação do Acidente (Multiselect)
        classificacao_acidente = st.multiselect(
            'Classificação do Acidente', classificacao_acidente_options)
        classificacao_acidente = classificacao_acidente[0] if classificacao_acidente else None

        # Ano (Integer)
        ano = st.number_input('Ano', min_value=2000)

        # Submit button
        submitted = st.form_submit_button("Predict")

    # Button to trigger prediction
    if submitted:
        # Make a prediction using your model
        input_data = {'br': br,
                      'tipo_acidente': tipo_acidente,
                      'uso_solo': uso_solo,
                      'regional': regional,
                      'classificacao_acidente': classificacao_acidente,
                      'ano': ano}

        data_pre = joblib.load('colunas')

        input_df = pd.DataFrame([input_data])

        scaler = StandardScaler()
        anos = [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013,
                2012, 2011, 2010, 2009, 2008, 2007, input_df['ano'].values[0]]
        scaled_ano = scaler.fit_transform(np.array(anos).reshape(-1, 1))

        input_df['ano'] = float(scaled_ano[len(anos)-1])

        st.write(f'Predicted Output: {input_df["ano"]}')

        for column in input_df.columns:
            if input_df[column].dtype == 'O':
                input_df[column] = input_df[column].str.lower()
                input_df[column] = input_df[column].replace(
                    '(null)', 'Unknown')
                input_df[column] = input_df[column].str.encode(
                    'ISO-8859-1').str.decode('utf-8', 'ignore')
            else:
                input_df[column] = input_df[column].fillna(-99999999)

        st.write(f'Df Output: {input_df}')
        valores = list()
        for i in input_df.columns:
            if i != 'ano':
                valores.append(input_df[i].iloc[0])

        st.write(f'Valores: {valores}')
        nova_linha = pd.DataFrame(
            [[0]*len(data_pre.columns)], columns=data_pre.columns)

        data_pre = pd.concat([data_pre, nova_linha], ignore_index=True)

        for valor in valores:
            for i in data_pre.columns:
                if valor in i:
                    data_pre[i].iloc[0] = 1

        # input_df = pd.get_dummies(input_df, columns=['br', 'tipo_acidente', 'classificacao_acidente',
        #                                              'uso_solo', 'regional'], drop_first=True)
        # st.write(f'Df Output: {input_df}')

        # for i in input_df.columns:
        #     if i != 'ano':
        #         input_df[i] = [0 if x == False else 1 for x in input_df[i]]

        st.write(f'Df Output: {data_pre}')

        # data_pre = pd.DataFrame(columns=data_pre.columns)
        for i in data_pre.columns:
            if i in input_df.columns and i != 'ano':
                data_pre[i] = 1
            else:
                data_pre[i] = 0

        data_pre['ano'] = input_df['ano']

        # model.compile(optimizer='adam',
        # loss='mean_squared_error')

        prediction = model.predict(data_pre)

        sensitivity = 0.2
        denormalized_value = (prediction * (2508 - 1) * sensitivity) + 1

        # denormalized_value = (
        #     prediction * (250 - 9)) + 9

        # Display the prediction
        st.write(f'Predicted Output: {denormalized_value}')


with col2:

    st.subheader("Predict # of Accidents based on: ")
    st.markdown("<style>color: skyblue; text-align: center; font-size: 10px;</style>",
                unsafe_allow_html=True)

    with st.form("accident_prediction_form+red"):
        # BR (Number stored as string)
        br = st.text_input('BR (Number)', '')

        # State (Multiselect)
        state = st.multiselect('State', state_options)
        regional = "SPRF-" + "-".join(state) if state else None

        # Ano (Integer)
        ano = st.number_input('Ano', min_value=2000)

        # Submit button
        submitted = st.form_submit_button("Predict")
     # Button to trigger prediction

    if submitted:
        # Make a prediction using your model
        input_data_ = {'br': br,
                       'regional': regional,
                       'ano': ano}

        data_red = joblib.load('colunas_reduzidas')

        input_red = pd.DataFrame([input_data_])

        scaler = StandardScaler()
        anos = [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013,
                2012, 2011, 2010, 2009, 2008, 2007, input_red['ano'].values[0]]
        scaled_ano = scaler.fit_transform(np.array(anos).reshape(-1, 1))

        input_red['ano'] = float(scaled_ano[len(anos)-1])

        for column in input_red.columns:
            if input_red[column].dtype == 'O':
                input_red[column] = input_red[column].str.lower()
                input_red[column] = input_red[column].replace(
                    '(null)', 'Unknown')
                input_red[column] = input_red[column].str.encode(
                    'ISO-8859-1').str.decode('utf-8', 'ignore')
            else:
                input_red[column] = input_red[column].fillna(-99999999)

        valores = list()
        for i in input_red.columns:
            if i != 'ano':
                valores.append(input_red[i].iloc[0])

        st.write(f'Valores: {valores}')
        nova_linha = pd.DataFrame(
            [[0]*len(data_red.columns)], columns=data_red.columns)

        data_red = pd.concat([data_red, nova_linha], ignore_index=True)

        for valor in valores:
            for i in data_red.columns:
                if valor in i:
                    data_red[i].iloc[0] = 1

        # input_red = pd.get_dummies(input_red, columns=['br', 'tipo_acidente', 'classificacao_acidente',
        #                                              'uso_solo', 'regional'], drop_first=True)
        # st.write(f'Df Output: {input_red}')

        # for i in input_red.columns:
        #     if i != 'ano':
        #         input_red[i] = [0 if x == False else 1 for x in input_red[i]]

        st.write(f'Df Output: {data_red}')

        # data_red = pd.DataFrame(columns=data_red.columns)
        for i in data_red.columns:
            if i in input_red.columns and i != 'ano':
                data_red[i] = 1
            else:
                data_red[i] = 0

        data_red['ano'] = input_red['ano']

        model_red.compile(optimizer='adam',
                          loss='mean_squared_error')

        prediction = model_red.predict(data_red)

        sensitivity = 0.2
        # denormalized_value = (prediction * (192326 - 1) * sensitivity) + 1

        denormalized_value = (
            prediction * (192326 - 1)) + 1

        # Display the prediction
        st.write(f'Predicted Output: {prediction}')
