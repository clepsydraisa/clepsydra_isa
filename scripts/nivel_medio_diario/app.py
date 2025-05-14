import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import re

# Ler o arquivo CSV
file_path = "/workspaces/clepsydra_isa/scripts/nivel_medio_diario/nivel_medio_diario.csv"  # Substitua pelo caminho real do seu CSV
data = pd.read_csv(file_path, parse_dates=['data'])

# Limpar valores na coluna 'nivel_medio_diario', removendo prefixos não numéricos
def clean_numeric_value(value):
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(match.group()) if match else None

data['nivel_medio_diario'] = data['nivel_medio_diario'].apply(clean_numeric_value)

# Inicializar o aplicativo Dash
app = dash.Dash(__name__)

# Layout do Dash com dois dropdowns e um gráfico
app.layout = html.Div([
    html.H1("Visualização de Dados"),
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
            {'label': 'Nível Médio Diário', 'value': 'nivel_medio_diario'}
        ],
        value=['nivel_medio_diario'],  # Valor inicial como uma lista
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
    fig = px.line()

    # Adicionar uma linha com marcadores para cada combinação de variável e localização selecionada
    for var in selected_variaveis:
        for loc in selected_localizacoes:
            subset = filtered_data[filtered_data['nome'] == loc]
            if not subset.empty:
                fig.add_scatter(
                    x=subset['data'], 
                    y=subset[var], 
                    mode='lines+markers',  # Linhas conectadas com pontos visíveis
                    name=f'{var} - {loc}'
                )
    
    # Configurar o layout da figura
    fig.update_layout(
        title="Gráfico com múltiplas localizações e variáveis",
        xaxis_title="Data",
        yaxis_title="Valores",
    )
    
    return fig

# Executar o servidor em localhost
if __name__ == '__main__':
    app.run_server(debug=True)