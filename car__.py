import streamlit as st
import pandas as pd
import os
from datetime import datetime
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import folium
import plotly.express as px
import pydeck as pdk
import numpy as np
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Car Accidents in Brazil.",
                   page_icon=":car:", layout="wide")

st.title(":car: :blue_car: Car Accidents in Brazil.")
st.markdown(
    '<style>div.block-container{padding-top:1rem;text-align:center;}</style>', unsafe_allow_html=True)

# Caching the data loading function


@st.cache_data
def load_data():
    df = pd.read_csv('preprocess_data.csv', sep=',',
                     encoding='latin1', header=0)
    df = df.reset_index()
    for column in df:
        if df[column].dtype == 'O':
            df[column] = df[column].str.replace('(null)', 'Unknown').str.encode(
                'latin1').str.decode('utf-8', 'ignore')
    df["data_inversa"] = pd.to_datetime(df["data_inversa"])
    df['horario'] = pd.to_datetime(df['horario'], format='%H:%M:%S').dt.time
    return df


df = load_data()

col1, col2 = st.columns(2)
# Getting the min and max date
startDate = pd.to_datetime(df["data_inversa"]).min()
endDate = pd.to_datetime(df["data_inversa"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))
    start_time = st.time_input(
        "Start Time", value=pd.to_datetime('08:00:00').time())

with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))
    end_time = st.time_input(
        "End Time", value=pd.to_datetime('18:00:00').time())

# Filtering the data
df = df[(df["data_inversa"] >= date1) & (df["data_inversa"] <= date2)].copy()
df = df[(df['horario'] >= start_time) & (df['horario'] <= end_time)]

st.sidebar.header("Filters: ")

dia_semana = st.sidebar.multiselect(
    "Day of the week: ", df["dia_semana"].unique())
if dia_semana:
    df = df[df["dia_semana"].isin(dia_semana)]

uf = st.sidebar.multiselect("State (UF): ", df["uf"].unique())
if uf:
    df = df[df["uf"].isin(uf)]

municipio = st.sidebar.multiselect("City: ", df["municipio"].unique())
if municipio:
    df = df[df["municipio"].isin(municipio)]

fase_dia = st.sidebar.multiselect("Time of Day: ", df["fase_dia"].unique())
if fase_dia:
    df = df[df["fase_dia"].isin(fase_dia)]

condicao_metereologica = st.sidebar.multiselect(
    "Weather: ", df["condicao_metereologica"].unique())
if condicao_metereologica:
    df = df[df["condicao_metereologica"].isin(condicao_metereologica)]

mortos_filter = st.sidebar.radio("Dead?", ["All", 'Yes', 'No'])
if mortos_filter == 'Yes':
    df = df[df['mortos'] != 0]
elif mortos_filter == 'No':
    df = df[df['mortos'] == 0]

injured_filter = st.sidebar.radio("Injured?", ["All", 'Yes', 'No'])
if injured_filter == 'Yes':
    df = df[df['feridos'] != 0]
elif injured_filter == 'No':
    df = df[df['feridos'] == 0]


accidents_per_state = df['uf'].value_counts().reset_index()
accidents_per_city = df['municipio'].value_counts().head(20).reset_index()
# Rename the columns for clarity
accidents_per_state.columns = ['State', 'Accidents']
accidents_per_city.columns = ['City', 'Accidents']

tipos_acidentes = df['tipo_acidente'].value_counts()
regionais = df['regional'].value_counts()
vias = df['tracado_via'].value_counts()
pistas = df['tipo_pista'].value_counts()
delegacia = df['delegacia'].value_counts()


# Calculate KPIs
total_accidents = len(df)
total_fatalities = df['mortos'].sum()
total_injuries = df['feridos_leves'].sum() + df['feridos_graves'].sum()
fatalities_per_accident = str(
    round((total_fatalities / total_accidents), 2)*100) + "%"
injuries_per_accident = str(
    round((total_injuries / total_accidents), 2)*100)+"%"

kp1, kp2, kp3, kp4, kp5 = st.columns(5)
# Define the KPIs
# Define KPIs
kpi_values = {
    "Total Accidents": total_accidents,
    "Total Fatalities": total_fatalities,
    "Total Injuries": total_injuries,
    "Fatalities per Accident": fatalities_per_accident,
    "Injuries per Accident": injuries_per_accident


}

# Define a custom CSS style for the KPIs
kpi_style = """
     .kpiBox {
    display: inline-block;
    width: 10rem;
    height: 5rem;
    background-color: #0E1218;
    border-left: 10px solid black;
    padding-top: 1rem;
    padding-left: 1.5rem;
    font-family: 'open sans';
    }
    .kpi-value {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .kpi-label {
        font-size: 30px;
        color: #555;
    }
"""
colunas = [kp1, kp2, kp3, kp4, kp5]
start = 0
# Iterate over KPIs and display them with custom style
for kpi_name, kpi_value in kpi_values.items():

    with colunas[start]:
        st.markdown(
            f"<div class='kpi-container'>"
            f"<div class='kpi-value'>{kpi_value}</div>"
            f"<div class='kpi-label'>{kpi_name}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    start = start + 1

st.divider()


st.subheader("Accidents per State")
st.markdown("<style>color: skyblue; text-align: center; font-size: 10px;</style>",
            unsafe_allow_html=True)

accidents_per_state_sorted = accidents_per_state.sort_values('Accidents')

bar_chart = alt.Chart(accidents_per_state_sorted).mark_bar().encode(
    y=alt.Y('Accidents:Q', title=None),
    x=alt.X('State:N', title=None, sort='-y'),
    tooltip=['State:N', 'Accidents:Q']
)
bar_chart = bar_chart.configure_axis(
    grid=False  # Set grid to False to remove grid lines
)

# Show the bar chart using Streamlit
st.altair_chart(bar_chart, use_container_width=True)


st.subheader("Top 20 Cities with the Highest Number of Accidents")
st.markdown("<style>color: skyblue; text-align: center; font-size: 10px;</style>",
            unsafe_allow_html=True)

accidents_per_city_sorted = accidents_per_city.sort_values('Accidents')

city_chart = alt.Chart(accidents_per_city_sorted).mark_bar().encode(
    y=alt.Y('Accidents:Q', title=None),
    # Set labelAngle to tilt the labels
    x=alt.X('City:N', title=None, sort='-y',
            axis=alt.Axis(labelAngle=-60)),
    tooltip=['City:N', 'Accidents:Q']
)

city_chart = city_chart.configure_axis(
    grid=False  # Set grid to False to remove grid lines
)
# Show the bar chart using Streamlit
st.altair_chart(city_chart, use_container_width=True)

st.subheader(":partly_sunny_rain: Weather Impact :partly_sunny_rain:")
st.markdown("<style>color: skyblue; text-align: center; font-size: 12px;</style>",
            unsafe_allow_html=True)

chart1, chart2 = st.columns(2)

with chart1:

    dead_por_condicao = df.groupby('condicao_metereologica')[
        'mortos'].sum().reset_index()
    dead_por_condicao.columns = ['Weather', 'Fatalities']
    chart_dead = dead_por_condicao[['Weather', 'Fatalities']]

    st.area_chart(
        chart_dead,
        x='Weather',
        y='Fatalities',
        use_container_width=True,
        color=['#FF0000']
    )
with chart2:
    total_por_condicao = df['condicao_metereologica'].value_counts(
    ).reset_index()
    total_por_condicao.columns = ['Weather', 'Accidents']
    chart_data = total_por_condicao[['Weather', 'Accidents']]

    st.area_chart(
        chart_data,
        x='Weather',
        y='Accidents',
        use_container_width=True,
        color=['#FFA500']
    )

chart3, chart4 = st.columns(2)

with chart3:

    st.subheader("Accidents per Year")
    st.markdown("<style>color: skyblue; text-align: center; font-size: 16px;</style>",
                unsafe_allow_html=True)
    acidentes_por_ano = df.groupby(df['ano'])['horario'].count().reset_index()
    acidentes_por_ano.columns = ['Year', 'Accidents']
    # Create a line chart for the trend of accidents by year
    line_chart = alt.Chart(acidentes_por_ano).mark_line().encode(
        x=alt.X('Year', title=None),  # Remove x-axis title
        y=alt.Y('Accidents', title=None),  # Remove y-axis title
        tooltip=['Year', 'Accidents']
    ).properties(
        width=700,
        height=400
    )

    # Add regression line as dots
    regression_dots = line_chart.transform_regression(
        'Year', 'Accidents', method='poly', order=1
    ).mark_point(color='red', filled=True)

    # Combine the two charts
    combined_chart = (
        line_chart + regression_dots).configure_legend(orient="bottom")

    # Configure axis to remove grids
    combined_chart = combined_chart.configure_axis(
        grid=False  # Remove as linhas de grade
    )
    # Show the combined chart
    st.altair_chart(combined_chart, use_container_width=True)

# with chart4:

# st.write(top_tipos_acidentes)

with chart4:
    # Group by day of the week and count accidents
    st.subheader("Accidents per Week Day")
    st.markdown("<style>color: skyblue; text-align: center; font-size: 16px;</style>",
                unsafe_allow_html=True)
    acidentes_por_dia = df['dia_semana'].value_counts().reset_index()
    acidentes_por_dia.columns = ['day_of_week', 'Accidents']

    # Define the custom sorting order for the days of the week
    custom_sort_order = ['monday', 'tuesday', 'wednesday',
                         'thursday', 'friday', 'saturday', 'sunday']

    # Convert 'day_of_week' to categorical with the custom sort order
    acidentes_por_dia['day_of_week'] = pd.Categorical(
        acidentes_por_dia['day_of_week'], categories=custom_sort_order, ordered=True)

    # Sort the DataFrame by the custom order
    acidentes_por_dia = acidentes_por_dia.sort_values('day_of_week')

    # Create the bar chart with explicit encoding for sorting
    bar_chart = alt.Chart(acidentes_por_dia).mark_bar().encode(
        x=alt.X('day_of_week:N', title=None,
                sort=custom_sort_order),  # Explicitly set the sorting order
        y=alt.Y('Accidents:Q', title=None),
        tooltip=['day_of_week:N', 'Accidents']
    ).properties(
        width=700,
        height=400
    )

    bar_chart = bar_chart.configure_axis(
        grid=False  # Set grid to False to remove grid lines
    )

    # Show the bar chart
    st.altair_chart(bar_chart, use_container_width=True)

st.divider()

chart6, chart7, chart8, chart9 = st.columns(4)

with chart6:
    st.subheader("Accidents per Type")
    st.markdown("<style>color: skyblue; text-align: center; font-size: 10px;</style>",
                unsafe_allow_html=True)
    top_tipos_acidentes_ = pd.DataFrame(columns=['Accident Type', 'Sum'])
    top_tipos_acidentes_['Accident Type'] = tipos_acidentes.keys(
    ).str.encode('latin1').str.decode('utf-8', 'ignore')
    top_tipos_acidentes_['Sum'] = tipos_acidentes.values

    st.write(top_tipos_acidentes_)
with chart7:
    st.subheader("Accidents per Road")
    st.markdown("<style>color: skyblue; text-align: center; font-size: 10px;</style>",
                unsafe_allow_html=True)
    vias_ = pd.DataFrame(columns=['Accident Road', 'Sum'])
    vias_['Accident Road'] = vias.keys()

    vias_['Accident Road'] = vias_['Accident Road'].str.encode(
        'latin1').str.decode('utf-8', 'ignore')
    vias_['Sum'] = vias.values

    st.write(vias_)


with chart8:
    st.subheader("Accidents per Police Station")
    st.markdown("<style>color: skyblue; text-align: center; font-size: 10px;</style>",
                unsafe_allow_html=True)
    delegacia_ = pd.DataFrame(columns=['Police Station', 'Sum'])
    delegacia_['Police Station'] = delegacia.keys().str.encode(
        'latin1').str.decode('utf-8', 'ignore')
    delegacia_['Sum'] = delegacia.values

    st.write(delegacia_)

with chart9:
    st.subheader("Accidents per Regional")
    st.markdown("<style>color: skyblue; text-align: center; font-size: 10px;</style>",
                unsafe_allow_html=True)
    regionais_ = pd.DataFrame(columns=['Accident Regional', 'Sum'])
    regionais_['Accident Regional'] = regionais.keys().str.encode(
        'latin1').str.decode('utf-8', 'ignore')
    regionais_['Sum'] = regionais.values

    st.write(regionais_)

coluna1, coluna2 = st.columns(2)

st.divider()

df_brasil = df[['latitude', 'longitude']].dropna()

st.subheader("Map of Accidents through Brazil.")
st.markdown("<style>color: skyblue; text-align: center; font-size: 10px;</style>",
            unsafe_allow_html=True)
# Criar o mapa com PyDeck
view_state = pdk.ViewState(
    latitude=-14.235,
    longitude=-51.9253,
    zoom=3,
    bearing=0,
    pitch=0,
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_brasil,
    get_position=["longitude", "latitude"],
    get_radius=10000,  # Ajuste conforme necessário
    get_fill_color=[255, 0, 0],
    pickable=True,
)

# Renderizar o mapa
map_ = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
)

st.pydeck_chart(map_)
