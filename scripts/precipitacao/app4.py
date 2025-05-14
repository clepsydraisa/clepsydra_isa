import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Ler o arquivo CSV
file_path = "/workspaces/clepsydra_isa/scripts/precipitacao/bd_precipitacao.csv"  # Substitua pelo caminho real do seu CSV
data = pd.read_csv(file_path, parse_dates=['data'])

# Garantir que a coluna de precipitação está no tipo numérico
data['precipitacao_dia_mm'] = pd.to_numeric(data['precipitacao_dia_mm'], errors='coerce')

# Definir valores fixos para o eixo Y com base no conjunto completo de dados
Y_MIN = data['precipitacao_dia_mm'].min() - 5  # Margem abaixo do mínimo global
Y_MAX = data['precipitacao_dia_mm'].max() + 5  # Margem acima do máximo global

# Inicializar o aplicativo Dash
app = dash.Dash(__name__)

# Layout do Dash com dois dropdowns e um gráfico
app.layout = html.Div([
    html.H1("Visualização de Precipitação"),
    html.Label("Selecione os códigos das localizações:"),
    dcc.Dropdown(
        id='codigo-dropdown',
        options=[
            {'label': f"{row['codigo']} - {row['nome']}", 'value': row['codigo']} 
            for _, row in data[['codigo', 'nome']].drop_duplicates().iterrows()
        ],
        value=[data['codigo'].unique()[0]],  # Valor inicial como uma lista
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
    [Input('codigo-dropdown', 'value'),
     Input('variavel-dropdown', 'value')]
)
def update_graph(selected_codigos, selected_variaveis):
    # Filtrar os dados com base nos códigos selecionados
    filtered_data = data[data['codigo'].isin(selected_codigos)]

    # Criar uma coluna para o título combinando código e nome
    filtered_data['titulo'] = filtered_data['codigo'] + " - " + filtered_data['nome']

    # Criar o gráfico de linhas
    fig = px.line(
        filtered_data,
        x='data',
        y=selected_variaveis[0],  # Usar a primeira variável selecionada
        color='titulo',  # Diferenciar linhas pelo título (código + nome)
        line_shape='linear'  # Forma da linha
    )

    # Atualizar o estilo das linhas para preto
    fig.update_traces(
        line=dict(color='black', width=2)  # Linhas pretas com espessura 2
    )

    # Configurar o layout da figura com escala fixa no eixo Y
    fig.update_layout(
        title="Gráfico de Precipitação",
        xaxis_title="Data",
        yaxis_title="Precipitação (mm)",
        legend_title="Código - Localização",
        yaxis=dict(range=[Y_MIN, Y_MAX])  # Escala fixa baseada nos valores globais
    )
    
    return fig

# Executar o servidor em localhost
if __name__ == '__main__':
    app.run_server(debug=True)