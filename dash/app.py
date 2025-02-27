# Importar librerías
from dash import Dash, html, dcc
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output

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

# Lista de materiales
materiales = list(df.columns)

# Iniciar la app
app = Dash(__name__)

# Layout con dos gráficos
app.layout = html.Div([
    html.H1("Sistema Predictivo para la Estimación en la Demanda de Materiales Reciclados", style={"text-align": "center"}),

    # Contenedor con el dropdown y las gráficas
    html.Div([
        # Dropdown para seleccionar material
        dcc.Dropdown(
            id="material-selector",
            options=[{"label": mat, "value": mat} for mat in materiales],
            value="Plástico",
            clearable=False,
            style={"width": "200px"}
        )
    ], style={"text-align": "center", "margin-bottom": "20px"}),

    # Contenedor con las dos gráficas en formato horizontal
    html.Div([
        dcc.Graph(id="grafica-total", style={"flex": "1", "height": "400px"}),  # Barras
        dcc.Graph(id="grafica-tendencia", style={"flex": "1", "height": "400px"})  # Tendencia
    ], style={"display": "flex", "gap": "20px", "justify-content": "center"}),

])

# Callback para actualizar las gráficas
@app.callback(
    [Output("grafica-total", "figure"), Output("grafica-tendencia", "figure")],
    Input("material-selector", "value")
)
def actualizar_graficas(material):
    df_agg = df.resample("Y").sum()
    df_agg.index = df_agg.index.year

    fig_barras = px.bar(
        df_agg, x=df_agg.index, y=material,
        title=f"Total de {material} recolectado por año",
        labels={"index": "Año", material: "Cantidad (kg)"},
        color_discrete_sequence=["green"]
    )
    fig_barras.update_layout(xaxis_title="Año", yaxis_title="Cantidad (kg)")


    df_trend = df.resample("M").sum()
    fig_tendencia = px.line(
        df_trend, x=df_trend.index, y=material,
        title=f"Tendencia de {material} recolectado (mensual)",
        labels={"Fecha": "Fecha", material: "Cantidad (kg)"},
        line_shape="spline",
        markers=True
    )
    fig_tendencia.update_layout(
        xaxis=dict(
            tickangle=-45,
            dtick="M6"  # Mostrar etiqueta cada 6 meses
        ),
        xaxis_title="Fecha",
        yaxis_title="Cantidad (kg)"
    )

    return fig_barras, fig_tendencia

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(debug=True)

