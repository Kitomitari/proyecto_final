# Importar librerías
#nota tener instaladas esta librerias
import dash
from dash import Dash, html, dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import numpy as np

# Importación de la data
df = pd.read_csv("/home/muresaki/Documents/proyecto_final/datos/dataset.csv")

# Convertir "Fecha" a tipo datetime
df["Fecha"] = pd.to_datetime(df["Fecha"])
df.set_index("Fecha", inplace=True)  # Convertir "Fecha" en índice

# Renombrar columnas
rename = {
    df.columns[0]: "Plástico",
    df.columns[1]: "Madera",
    df.columns[2]: "Vidrio",
    df.columns[3]: "Sargazo"
}
df.rename(columns=rename, inplace=True)

# Lista de materiales para que funcione el selector
materiales = list(df.columns)

# Iniciar la app
app = Dash(__name__)

# Layout con el dropdown para seleccionar material y las gráficas
app.layout = html.Div([
    html.H1("Sistema Predictivo para la Estimación en la Demanda de Materiales Reciclados", style={"text-align": "center"}),

    # Contenedor con el dropdown para seleccionar el material
    html.Div([
        # Dropdown o boton para seleccionar material 
        dcc.Dropdown(
            id="material-selector",
            options=[{"label": mat, "value": mat} for mat in materiales],
            value="Plástico",  # Valor inicial
            clearable=False,
            style={"width": "200px"}
        )
    ], style={"text-align": "center", "margin-bottom": "20px"}),

    # Contenedor con las dos primeras gráficas (barras y tendencia) en formato horizontal
    html.Div([
        dcc.Graph(id="grafica-total", style={"flex": "1", "height": "400px"}),  # Barras
        dcc.Graph(id="grafica-tendencia", style={"flex": "1", "height": "400px"})  # Tendencia mensual
    ], style={"display": "flex", "gap": "20px", "justify-content": "center"}),  # Primer contenedor de gráficas

    # Contenedor para la distribución de residuos y autocorrelación, en formato horizontal
    html.Div([
        # Gráfica de residuos
        html.Div([
            dcc.Graph(id="grafica-residuos", style={"height": "400px"})  # Distribución de residuos
        ], style={"flex": "1", "margin-right": "10px"}),  # Residuos a la izquierda

        # Gráfica de autocorrelación y autocorrelación parcial
        html.Div([
            dcc.Graph(id="acf-pacf", style={"height": "400px"})  # Autocorrelación y autocorrelación parcial
        ], style={"flex": "1", "margin-left": "10px"})  # Autocorrelación a la derecha

    ], style={"display": "flex", "gap": "20px", "justify-content": "center", "margin-top": "30px"})  # Segundo contenedor de gráficas

], style={"background-color": "#f5f5f5", "padding": "20px"})  # Fondo gris claro para el layout

# Callback para actualizar las gráficas de total, tendencia mensual, distribución de residuos y autocorrelación
@app.callback(
    [Output("grafica-total", "figure"), 
     Output("grafica-tendencia", "figure"), 
     Output("grafica-residuos", "figure"),
     Output("acf-pacf", "figure")],
    [Input("material-selector", "value")]
)
def actualizar_graficas(material):
    print(f"Actualizando gráfico para el material: {material}")
    
    # Validar si los datos existen
    df_agg = df.resample("Y").sum()
    if material not in df_agg.columns:
        print(f"Error: El material '{material}' no se encuentra en los datos.")
    
    df_agg.index = df_agg.index.year

    #  Gráfico de Barras (Total por Año) 
    fig_barras = px.bar(
        df_agg, x=df_agg.index, y=material,
        title=f"Total de {material} recolectado por año",
        labels={"index": "Año", material: "Cantidad (kg)"},
        color_discrete_sequence=["#002D62"]  
    )
    fig_barras.update_layout(
        xaxis_title="Año", 
        yaxis_title="Cantidad (kg)", 
        plot_bgcolor="#FFFFFF",  # Fondo 
        paper_bgcolor="#FFFFFF"
    )

    # Gráfico de Tendencia (Evolución Mensual) 
    df_trend = df.resample("M").sum()
    fig_tendencia = px.line(
        df_trend, x=df_trend.index, y=material,
        title=f"Tendencia de {material} recolectado (mensual)",
        labels={"Fecha": "Fecha", material: "Cantidad (kg)"},
        line_shape="spline",
        markers=True,
        color_discrete_sequence=["#D50032"]  
    )
    fig_tendencia.update_layout(
        xaxis=dict(
            tickangle=-45,
            dtick="M6" 
        ),
        xaxis_title="Fecha",
        yaxis_title="Cantidad (kg)",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )

    # Gráfico de distribución de residuos 
    residuos = df[material] - df[material].rolling(window=10).mean()
    fig_residuos = px.histogram(
        residuos, 
        title=f"Distribución de los residuos de {material} (en kg)",
        labels={"value": "Residuos (kg)"},
        nbins=20,  
        color_discrete_sequence=["#D50032"]  
    )
    fig_residuos.update_layout(
        xaxis_title="Residuos (kg)",
        yaxis_title="Frecuencia",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )

    # Gráfico de autocorrelación (ACF) 
    lags = np.arange(1, 13)
    acf_values = [df[material].autocorr(lag) for lag in lags]
    print(f"ACF values: {acf_values}")

    fig_acf = go.Figure()
    fig_acf.add_trace(go.Scatter(
        x=lags, y=acf_values, 
        mode="markers+lines", 
        name="ACF", 
        line=dict(color="#002D62")  
    ))
    fig_acf.update_layout(
        title=f"Autocorrelación de {material}",
        xaxis_title="Lags",
        yaxis_title="ACF",
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )

    # Gráfico de autocorrelación parcial (PACF) - Rojo ITLA
    from statsmodels.tsa.stattools import pacf
    pacf_values = pacf(df[material], nlags=12)

    fig_pacf = go.Figure()
    fig_pacf.add_trace(go.Scatter(
        x=lags, y=pacf_values[1:], 
        mode="markers+lines", 
        name="PACF", 
        line=dict(color="#D50032")  
    ))
    fig_pacf.update_layout(
        title=f"Autocorrelación Parcial de {material}",
        xaxis_title="Lags",
        yaxis_title="PACF",
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )

    # Retornar las gráficas
    return fig_barras, fig_tendencia, fig_residuos, fig_acf


# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(debug=True)