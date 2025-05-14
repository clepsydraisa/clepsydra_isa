import pandas as pd
import plotly.express as px
import os

# Ler o arquivo CSV
file_path = "/Users/diogopinto/Documents/Pessoal/path_4med/snirh_scrape/bd_CHARTS/piezo/bd_piezo.csv"  # Substitua pelo caminho real do seu CSV
data = pd.read_csv(file_path, parse_dates=['data'])

# Garantir que a coluna numérica está no tipo float
data['profundidade_nivel_agua'] = pd.to_numeric(data['profundidade_nivel_agua'], errors='coerce')

# Definir o diretório de saída
output_dir = "/Users/diogopinto/Documents/Pessoal/path_4med/img_piezo"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Obter todas as localizações únicas
localizacoes = sorted(data['localizacao'].unique())

# Variável a ser plotada
variavel = 'profundidade_nivel_agua'

# Loop para gerar e salvar um gráfico para cada localização
for loc in localizacoes:
    # Filtrar os dados para a localização atual
    filtered_data = data[data['localizacao'] == loc]

    # Obter a freguesia correspondente (assumindo consistência)
    freguesia = filtered_data['freguesia'].iloc[0] if pd.notna(filtered_data['freguesia'].iloc[0]) else "Desconhecida"
    label = f"{loc} - {freguesia}"

    # Criar o gráfico de linhas usando plotly.express
    fig = px.line(
        filtered_data,
        x='data',
        y=variavel,  # Apenas profundidade_nivel_agua
        title=f"Profundidade do Nível da Água em {label}"
    )

    # Atualizar o estilo das linhas
    fig.update_traces(
        line=dict(color='blue', width=2),  # Definir a cor como azul para todas as linhas
        mode='lines+markers'  # Linhas com marcadores
    )

    # Configurar o layout da figura
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Profundidade (mm)",  # Alterado para "Profundidade (mm)"
        showlegend=False,  # Não precisa de legenda, pois é uma única variável
        yaxis=dict(autorange='reversed')  # Inverter a ordem do eixo Y
    )

    # Sanitizar o nome do arquivo substituindo caracteres inválidos
    safe_loc = loc.replace('/', '_')  # Substitui '/' por '_'
    output_path = os.path.join(output_dir, f"profundidade_{safe_loc}.png")

    # Salvar o gráfico como PNG
    fig.write_image(output_path, format="png", width=1200, height=600)

    print(f"Gráfico salvo: {output_path}")

print("Todos os gráficos foram gerados e salvos!")