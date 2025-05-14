import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Ler o arquivo CSV
file_path = "/workspaces/clepsydra_isa/scripts/piezo/bd_piezo.csv"  # Substitua pelo caminho real do seu CSV
data = pd.read_csv(file_path, parse_dates=['data'])

# Garantir que a coluna de profundidade está no tipo numérico
data['profundidade_nivel_agua'] = pd.to_numeric(data['profundidade_nivel_agua'], errors='coerce')

# Definir uma paleta de cores para as localizações
colors = px.colors.qualitative.Plotly  # Lista de cores: ['#636EFA', '#EF553B', '#00CC96', ...]
unique_localizacoes = sorted(data['localizacao'].unique())
color_map = {loc: colors[i % len(colors)] for i, loc in enumerate(unique_localizacoes)}

# Inicializar o aplicativo Dash
app = dash.Dash(__name__)

# Layout do Dash com um dropdown e um gráfico
app.layout = html.Div([
    html.H1("Visualização de Profundidade"),
    html.Label("Selecione as localizações:"),
    dcc.Dropdown(
        id='localizacao-dropdown',
        options=[
            {'label': f"{row['localizacao']} - {row['freguesia']}", 'value': row['localizacao']} 
            for _, row in data[['localizacao', 'freguesia']].drop_duplicates().iterrows()
        ],
        value=[sorted(data['localizacao'].unique())[0]],  # Valor inicial como uma lista
        multi=True  # Permitir múltiplas seleções
    ),
    dcc.Graph(id='graph-output')
])

# Callback para atualizar o gráfico com base nos filtros selecionados
@app.callback(
    Output('graph-output', 'figure'),
    [Input('localizacao-dropdown', 'value')]
)
def update_graph(selected_localizacoes):
    # Filtrar os dados com base nas localizações selecionadas
    filtered_data = data[data['localizacao'].isin(selected_localizacoes)]

    # Inicializar a figura
    fig = px.line()

    # Adicionar uma linha com marcadores para cada localização selecionada
    for loc in selected_localizacoes:
        subset = filtered_data[filtered_data['localizacao'] == loc]
        if not subset.empty:
            # Obter a freguesia correspondente (assumindo consistência)
            freguesia = subset['freguesia'].iloc[0]
            # Usar a cor associada à localização
            line_color = color_map[loc]
            fig.add_scatter(
                x=subset['data'], 
                y=subset['profundidade_nivel_agua'], 
                mode='lines+markers',  # Linhas conectadas com pontos visíveis
                name=f'{loc} - {freguesia}',  # Legenda: localização + freguesia
                line=dict(color=line_color, width=2, dash='solid')
            )
    
    # Configurar o layout da figura com eixo Y invertido
    fig.update_layout(
        title="Gráfico de Profundidade",
        xaxis_title="Data",
        yaxis_title="Profundidade (mm)",
        legend_title="Localização - Freguesia",
        yaxis=dict(autorange='reversed')  # Inverter a ordem do eixo Y
    )
    
    return fig

# Executar o servidor em localhost
if __name__ == '__main__':
    app.run_server(debug=True)