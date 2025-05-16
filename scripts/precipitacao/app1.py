import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Ler o arquivo CSV
file_path = "/Users/diogopinto/Documents/Pessoal/path_4med/snirh_scrape/bd_CHARTS/precipitacao/bd_precipitacao.csv"  # Substitua pelo caminho real do seu CSV
data = pd.read_csv(file_path, parse_dates=['data'])

# Garantir que a coluna de precipitação está no tipo numérico
data['precipitacao_dia_mm'] = pd.to_numeric(data['precipitacao_dia_mm'], errors='coerce')

# Inicializar o aplicativo Dash
app = dash.Dash(__name__)

# Layout do Dash com dois dropdowns e um gráfico
app.layout = html.Div([
    html.H1("Visualização de Precipitação"),
    html.Label("Selecione as localizações:"),
    dcc.Dropdown(
        id='localizacao-dropdown',
        options=[
            {'label': loc, 'value': loc} for loc in sorted(data['nome'].unique())
        ],
        value=[sorted(data['nome'].unique())[0]],  # Valor inicial como uma lista
        multi=True  # Permitir múltiplas seleções
    ),
    html.Label("Selecione as variáveis:"),
    dcc.Dropdown(
        id='variavel-dropdown',
        options=[
            {'label': 'Precipitação (mm)', 'value': 'precipitacao_dia_mm'}
        ],
        value=['precipitacao_dia_mm'],  # Valor inicial como uma lista
        multi=True  # Permitir múltiplas seleções
    ),
    dcc.Graph(id='graph-output')
])

# Callback para atualizar o gráfico com base nos filtros selecionados
@app.callback(
    Output('graph-output', 'figure'),
    [Input('localizacao-dropdown', 'value'),
     Input('variavel-dropdown', 'value')]
)
def update_graph(selected_localizacoes, selected_variaveis):
    # Filtrar os dados com base nos inputs
    filtered_data = data[data['nome'].isin(selected_localizacoes)]

    # Inicializar a figura
    fig = px.scatter()

    # Adicionar os pontos para cada combinação de variável e localização selecionada
    for var in selected_variaveis:
        for loc in selected_localizacoes:
            subset = filtered_data[filtered_data['nome'] == loc]
            if not subset.empty:
                fig.add_scatter(
                    x=subset['data'], 
                    y=subset[var], 
                    mode='markers',  # Apenas pontos
                    name=f'{var} - {loc}'
                )
    
    # Configurar o layout da figura
    fig.update_layout(
        title="Gráfico de Precipitação",
        xaxis_title="Data",
        yaxis_title="Valores (mm)",
    )
    
    return fig

# Executar o servidor em localhost
if __name__ == '__main__':
    app.run_server(debug=True)
