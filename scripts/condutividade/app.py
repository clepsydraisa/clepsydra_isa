import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Ler o arquivo CSV
file_path = "/workspaces/clepsydra_isa/scripts/condutividade/bd_condut.csv"
data = pd.read_csv(file_path, parse_dates=['data'])

# Garantir que as colunas de condutividade estão no tipo numérico
data['condutividade'] = pd.to_numeric(data['condutividade'], errors='coerce')
data['condcamp20c'] = pd.to_numeric(data['condcamp20c'], errors='coerce')

# Ordenar os dados por data em ordem crescente
data = data.sort_values('data')

# Definir uma paleta de cores para as localizações (apenas para consistência visual)
colors = px.colors.qualitative.Plotly
unique_localizacoes = sorted(data['localizacao'].unique())
color_map = {loc: colors[i % len(colors)] for i, loc in enumerate(unique_localizacoes)}

# Inicializar o aplicativo Dash
app = dash.Dash(__name__)

# Layout do Dash com dropdowns para localizações e variáveis
app.layout = html.Div([
    html.H1("Visualização de Condutividade"),
    html.Label("Selecione as localizações:"),
    dcc.Dropdown(
        id='localizacao-dropdown',
        options=[
            {'label': f"{row['localizacao']} - {row['freguesia']}", 'value': row['localizacao']}
            for _, row in data[['localizacao', 'freguesia']].drop_duplicates().iterrows()
        ],
        value=[sorted(data['localizacao'].unique())[0]],  # Valor inicial
        multi=True
    ),
    html.Label("Selecione as variáveis:"),
    dcc.Dropdown(
        id='variavel-dropdown',
        options=[
            {'label': 'Condutividade', 'value': 'condutividade'},
            {'label': 'Condutividade a 20ºC', 'value': 'condcamp20c'}
        ],
        value=['condutividade'],  # Valor inicial
        multi=True
    ),
    dcc.Graph(id='graph-output')
])

# Callback para atualizar o gráfico
@app.callback(
    Output('graph-output', 'figure'),
    [Input('localizacao-dropdown', 'value'),
     Input('variavel-dropdown', 'value')]
)
def update_graph(selected_localizacoes, selected_variaveis):
    # Filtrar os dados com base nas localizações selecionadas
    filtered_data = data[data['localizacao'].isin(selected_localizacoes)]

    # Inicializar a figura
    fig = px.line()

    # Adicionar linhas para cada localização e variável selecionada
    for loc in selected_localizacoes:
        subset = filtered_data[filtered_data['localizacao'] == loc]
        if not subset.empty:
            freguesia = subset['freguesia'].iloc[0]  # Obter freguesia correspondente
            # Para cada variável selecionada
            for var in selected_variaveis:
                if var == 'condutividade':
                    fig.add_scatter(
                        x=subset['data'],
                        y=subset['condutividade'],
                        mode='lines+markers',
                        name=f'{loc} - {freguesia} (Condutividade)',
                        line=dict(color='red', width=2, dash='solid')
                    )
                elif var == 'condcamp20c':
                    fig.add_scatter(
                        x=subset['data'],
                        y=subset['condcamp20c'],
                        mode='lines+markers',
                        name=f'{loc} - {freguesia} (Cond. 20ºC)',
                        line=dict(color='blue', width=2, dash='dash')
                    )

    # Configurar o layout da figura
    fig.update_layout(
        title="Gráfico de Condutividade",
        xaxis_title="Data",
        yaxis_title="Condutividade (µS/cm)",
        legend_title="Localização - Variável",
    )

    return fig

# Executar o servidor em localhost
if __name__ == '__main__':
    app.run_server(debug=True)