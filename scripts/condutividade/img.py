import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Ler o arquivo CSV
file_path = "/Users/diogopinto/Documents/Pessoal/path_4med/snirh_scrape/bd_CHARTS/condutividade/bd_condut.csv"
data = pd.read_csv(file_path, parse_dates=['data'])

# Garantir que as colunas numéricas estão no tipo float
data['condutividade'] = pd.to_numeric(data['condutividade'], errors='coerce')
data['condcamp20c'] = pd.to_numeric(data['condcamp20c'], errors='coerce')

# Ordenar os dados por data em ordem crescente
data = data.sort_values('data')

# Definir o diretório de saída
output_dir = "/Users/diogopinto/Documents/Pessoal/path_4med/img_condutividade"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Obter todas as localizações únicas
localizacoes = sorted(data['localizacao'].unique())

# Loop para gerar e salvar um gráfico para cada localização
for loc in localizacoes:
    # Filtrar os dados para a localização atual
    filtered_data = data[data['localizacao'] == loc]

    # Obter a freguesia correspondente (assumindo consistência)
    freguesia = filtered_data['freguesia'].iloc[0] if pd.notna(filtered_data['freguesia'].iloc[0]) else "Desconhecida"
    label = f"{loc} - {freguesia}"

    # Criar a figura com plotly.graph_objects para adicionar múltiplas linhas
    fig = go.Figure()

    # Adicionar linha para 'condutividade' (vermelha, sólida)
    if filtered_data['condutividade'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=filtered_data['data'],
                y=filtered_data['condutividade'],
                mode='lines+markers',
                name='Condutividade',
                line=dict(color='red', width=2, dash='solid')
            )
        )

    # Adicionar linha para 'condcamp20c' (azul, tracejada)
    if filtered_data['condcamp20c'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=filtered_data['data'],
                y=filtered_data['condcamp20c'],
                mode='lines+markers',
                name='Cond. a 20ºC',
                line=dict(color='blue', width=2, dash='dash')
            )
        )

    # Configurar o layout da figura
    fig.update_layout(
        title=f"Condutividade em {label}",
        xaxis_title="Data",
        yaxis_title="Condutividade (µS/cm)",
        legend_title="Variáveis",
        showlegend=True  # Mostrar legenda
    )

    # Sanitizar o nome do arquivo substituindo caracteres inválidos
    safe_loc = loc.replace('/', '_')  # Substitui '/' por '_'
    output_path = os.path.join(output_dir, f"condutividade_{safe_loc}.png")

    # Salvar o gráfico como PNG
    fig.write_image(output_path, format="png", width=1200, height=600)

    print(f"Gráfico salvo: {output_path}")

print("Todos os gráficos foram gerados e salvos!")