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
# ['uf', 'tipo_acidente', 'fase_dia', 'regional', 'ano']

st.title(":blue_car: Car Accidents in Brazil :car:")
st.subheader("Predictor")
st.markdown(
    '<style>div.block-container{padding-top:1rem;text-align:center;}</style>', unsafe_allow_html=True)

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

fase_dia_options = ['full night', 'full day', 'dusk', 'dawn', 'unknown']
state_options = ['ES', 'SP', 'MT', 'PR', 'MG', 'BA', 'RJ', 'RS', 'SC', 'PI', 'GO',
                 'PE', 'PA', 'MS', 'MA', 'CE', 'AP', 'PB', 'SE', 'RO', 'RN', 'TO',
                 'RR', 'DF', 'AL', 'AC', 'AM', 'Unknown']

classificacao_acidente_options = [
    'Com Vtimas Feridas', 'Com Vtimas Fatais', 'Sem Vtimas', 'Unknown',
    'Ignorado']

col1, col2 = st.columns(2)


with col1:
    st.subheader("Features 1")
    st.markdown("<style>color: skyblue; text-align: center; font-size: 10px;</style>",
                unsafe_allow_html=True)

    with st.form("accident_prediction_form"):
        # BR (Number stored as string)
        br = st.text_input('Highway (Number)', '')

        # Tipo Acidente (Multiselect)
        tipo_acidente = st.multiselect(
            'Accident Type', sorted(tipo_acidente_options))
        tipo_acidente = tipo_acidente[0] if tipo_acidente else None

        # Uso Solo (Multiselect)
        fase_dia = st.multiselect('Time of Day', sorted(fase_dia_options))
        fase_dia = fase_dia[0] if fase_dia else None

        # State (Multiselect)
        state = st.multiselect('State', sorted(state_options))
        regional = "SPRF-" + "-".join(state) if state else None

        # # Classificação do Acidente (Multiselect)
        # classificacao_acidente = st.multiselect(
        #     'Classificação do Acidente', classificacao_acidente_options)
        # classificacao_acidente = classificacao_acidente[0] if classificacao_acidente else None

        # Ano (Integer)
        ano = st.number_input('Year', min_value=2000)

        # Submit button
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = {'br': br,
                      'tipo_acidente': tipo_acidente,
                      'fase_dia': fase_dia,
                      'regional': regional,
                      'ano': ano}

        data_pre = joblib.load('colunas')

        input_df = pd.DataFrame([input_data])

        scaler = StandardScaler()
        anos = [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013,
                2012, 2011, 2010, 2009, 2008, 2007, input_df['ano'].values[0]]
        scaled_ano = scaler.fit_transform(np.array(anos).reshape(-1, 1))

        input_df['ano'] = float(scaled_ano[len(anos)-1])

        for column in input_df.columns:
            if input_df[column].dtype == 'O':
                input_df[column] = input_df[column].str.lower()
                input_df[column] = input_df[column].replace(
                    '(null)', 'Unknown')
                input_df[column] = input_df[column].str.encode(
                    'ISO-8859-1').str.decode('utf-8', 'ignore')
            else:
                input_df[column] = input_df[column].fillna(-99999999)

        valores = list()
        for i in input_df.columns:
            if i != 'ano':
                valores.append(input_df[i].iloc[0])

        nova_linha = pd.DataFrame(
            [[0]*len(data_pre.columns)], columns=data_pre.columns)

        data_pre = pd.concat([data_pre, nova_linha], ignore_index=True)

        for valor in valores:
            for i in data_pre.columns:
                if valor in i:
                    data_pre[i].iloc[0] = 1

        # data_pre = pd.DataFrame(columns=data_pre.columns)
        for i in data_pre.columns:
            if i in input_df.columns and i != 'ano':
                data_pre[i] = 1
            else:
                data_pre[i] = 0

        data_pre['ano'] = input_df['ano']

        prediction = model.predict(data_pre)

        denormalized_value = (prediction * (4681 - 1)) + 1

        st.subheader(f"""{round(denormalized_value[0][0], 0)} accidents""")
        st.markdown("<style>color: skyblue; text-align: center; font-size: 10px;</style>",
                    unsafe_allow_html=True)

with col2:

    st.subheader("Features 2")
    st.markdown("<style>color: skyblue; text-align: center; font-size: 10px;</style>",
                unsafe_allow_html=True)

    with st.form("accident_prediction_form+red"):
        br = st.text_input('Highway (Number)', '')

        state = st.multiselect('State', state_options)
        regional = "SPRF-" + "-".join(state) if state else None

        ano = st.number_input('Year', min_value=2000)

        submitted = st.form_submit_button("Predict")

    if submitted:
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

        nova_linha = pd.DataFrame(
            [[0]*len(data_red.columns)], columns=data_red.columns)

        data_red = pd.concat([data_red, nova_linha], ignore_index=True)

        for valor in valores:
            for i in data_red.columns:
                if valor in i:
                    data_red[i].iloc[0] = 1

        for i in data_red.columns:
            if i in input_red.columns and i != 'ano':
                data_red[i] = 1
            else:
                data_red[i] = 0

        data_red['ano'] = input_red['ano']

        model_red.compile(optimizer='adam',
                          loss='mean_squared_error')

        prediction = model_red.predict(data_red)

        sensitivity = 0.5
        denormalized_value_ = (prediction * (192326 - 1) * sensitivity) + 1

        st.subheader(f"""{round(denormalized_value_[0][0], 0)} accidents""")
        st.markdown("<style>color: skyblue; text-align: center; font-size: 10px;</style>",
                    unsafe_allow_html=True)

st.divider()


st.subheader(f"""Érica ferreira""")
st.markdown("<style>color: skyblue; text-align: center; font-size: 5px;</style>",
            unsafe_allow_html=True)
st.subheader(f"""Data Scientist""")
st.markdown("<style>color: skyblue; text-align: center; font-size: 5px;</style>",
            unsafe_allow_html=True)

st.write(
    "[My github](https://github.com/dsericaferreira)")


st.write(
    "[My Linkedin](https://www.linkedin.com/in/ericacarneiro-ds/)")
