import pandas as pd
import plotly.express as px
import os

# Ler o arquivo CSV
file_path = "/Users/diogopinto/Documents/Pessoal/path_4med/snirh_scrape/bd_CHARTS/nitrato/bd_nitrato.csv"  # Caminho do CSV de nitratos
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

# Definir o diretório de saída
output_dir = "/Users/diogopinto/Documents/Pessoal/path_4med/img_nitrato"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Obter todas as localizações únicas
localizacoes = sorted(data['localizacao'].unique())

# Variável a ser plotada
variavel = 'nitrato'

# Loop para gerar e salvar um gráfico para cada localização
for loc in localizacoes:
    # Filtrar os dados para a localização atual
    filtered_data = data[data['localizacao'] == loc]

    # Ordenar os dados por data em ordem crescente
    filtered_data = filtered_data.sort_values(by='data')

    # Obter a freguesia correspondente (assumindo consistência)
    freguesia = filtered_data['freguesia'].iloc[0] if pd.notna(filtered_data['freguesia'].iloc[0]) else "Desconhecida"
    label = f"{loc} - {freguesia}"

    # Criar o gráfico de linhas usando plotly.express
    fig = px.line(
        filtered_data,
        x='data',
        y=variavel,  # Apenas nitrato
        title=f"Concentração de Nitratos em {label}"
    )

    # Atualizar o estilo das linhas
    fig.update_traces(
        line=dict(color='blue', width=2),  # Definir a cor como azul para todas as linhas
        mode='lines+markers'  # Linhas com marcadores
    )

    # Configurar o layout da figura
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Concentração NO3 (mg/l)",  # Título do eixo Y
        showlegend=False  # Não precisa de legenda, pois é uma única variável
    )

    # Sanitizar o nome do arquivo substituindo caracteres inválidos
    safe_loc = loc.replace('/', '_')  # Substitui '/' por '_'
    output_path = os.path.join(output_dir, f"nitrato_{safe_loc}.png")

    # Salvar o gráfico como PNG
    fig.write_image(output_path, format="png", width=1200, height=600)

    print(f"Gráfico salvo: {output_path}")

print("Todos os gráficos foram gerados e salvos!")