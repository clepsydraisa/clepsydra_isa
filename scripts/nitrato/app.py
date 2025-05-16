import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Ler o arquivo CSV e ajustar os valores de nitrato
file_path = "/workspaces/clepsydra_isa/scripts/nitrato/bd_nitrato.csv" 
data = pd.read_csv(file_path, parse_dates=['data'])

# Converter os valores de nitrato, tratando os prefixos (<) e (e<)
def clean_nitrate_value(value):
    try:
        # Se o valor for uma string, remover os prefixos (<) ou (e<)
        if isinstance(value, str):
            value = value.replace('(e<)', '').replace('(<)', '')
        return float(value)
    except ValueError:
        return 0  # Para outros casos de erro, mantém como 0

data['nitrato'] = data['nitrato'].apply(clean_nitrate_value)

# Inicializar o aplicativo Dash
app = dash.Dash(__name__)

# Layout do Dash com um dropdown e um gráfico
app.layout = html.Div([
    html.H1("Visualização de Nitratos"),
    html.Label("Selecione os códigos das localizações:"),
    dcc.Dropdown(
        id='codigo-dropdown',
        options=[
            {'label': f"{row['localizacao']} - {row['freguesia']}", 'value': row['localizacao']} 
            for _, row in data[['localizacao', 'freguesia']].drop_duplicates().iterrows()
        ],
        value=[sorted(data['localizacao'].unique())[0]],  # Valor inicial como uma lista
        multi=True  # Permitir múltiplas seleções
    ),
    dcc.Graph(id='graph-output')
])

# Callback para atualizar o gráfico com base nos códigos selecionados
@app.callback(
    Output('graph-output', 'figure'),
    [Input('codigo-dropdown', 'value')]
)
def update_graph(selected_codigos):
    # Filtrar os dados com base nos códigos selecionados
    filtered_data = data[data['localizacao'].isin(selected_codigos)]

    # Inicializar a figura
    fig = px.line()

    # Adicionar uma linha conectada com pontos visíveis para cada localização
    for codigo in selected_codigos:
        subset = filtered_data[filtered_data['localizacao'] == codigo]
        if not subset.empty:
            # Ordenar os dados por data em ordem crescente
            subset = subset.sort_values(by='data')
            # Obter a freguesia correspondente (assumindo consistência)
            freguesia = subset['freguesia'].iloc[0] if pd.notna(subset['freguesia'].iloc[0]) else "Desconhecida"
            fig.add_scatter(
                x=subset['data'], 
                y=subset['nitrato'], 
                mode='lines+markers',  # Linhas conectadas com marcadores nos vértices
                name=f'{codigo} - {freguesia}'  # Legenda: código + freguesia
            )
    
    # Configurar o layout da figura
    fig.update_layout(
        title="Gráfico de Nitratos (linhas conectadas com marcadores)",
        xaxis_title="Data",
        yaxis_title="Concentração NO3 (mg/l)",
        legend_title="Código - Freguesia"
    )
    
    return fig

# Executar o servidor em localhost
if __name__ == '__main__':
    app.run_server(debug=True)