import pandas as pd
import plotly.express as px
import os
import re

# Ler o arquivo CSV
file_path = "/Users/diogopinto/Documents/Pessoal/path_4med/snirh_scrape/bd_CHARTS/caudal/caudal.csv"
data = pd.read_csv(file_path, parse_dates=['data'])

# Função para extrair apenas os números da string
def extract_numeric(value):
    if pd.isna(value):
        return float('nan')
    match = re.search(r'-?\d*\.?\d+', str(value))  # Procura por números (com ou sem decimais)
    if match:
        return float(match.group())
    return float('nan')  # Retorna NaN se não encontrar número

# Aplicar a função à coluna de caudal
data['caudal_médio_diário(m3/s)'] = data['caudal_médio_diário(m3/s)'].apply(extract_numeric)

# Ordenar os dados por data em ordem crescente
data = data.sort_values('data')

# Definir o diretório de saída
output_dir = "/Users/diogopinto/Documents/Pessoal/path_4med/img_caudal"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Obter todas as localizações únicas
localizacoes = sorted(data['localizacao'].unique())

# Variável a ser plotada
variavel = 'caudal_médio_diário(m3/s)'

# Loop para gerar e salvar um gráfico para cada localização
for loc in localizacoes:
    # Filtrar os dados para a localização atual
    filtered_data = data[data['localizacao'] == loc]

    # Criar o gráfico de linhas usando plotly.express
    fig = px.line(
        filtered_data,
        x='data',
        y=variavel,
        title=f"Caudal Médio Diário em {loc}"
    )

    # Atualizar o estilo das linhas
    fig.update_traces(
        line=dict(color='green', width=2),  # Definir a cor como verde para consistência
        mode='lines+markers'  # Linhas com marcadores
    )

    # Configurar o layout da figura
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Caudal Médio Diário (m³/s)",
        showlegend=False  # Não precisa de legenda, pois é uma única variável
    )

    # Sanitizar o nome do arquivo substituindo caracteres inválidos
    safe_loc = loc.replace('/', '_').replace(' ', '_')  # Substitui '/' e espaços por '_'
    output_path = os.path.join(output_dir, f"caudal_{safe_loc}.png")

    # Salvar o gráfico como PNG
    fig.write_image(output_path, format="png", width=1200, height=600)

    print(f"Gráfico salvo: {output_path}")

print("Todos os gráficos foram gerados e salvos!")