import pandas as pd
import plotly.express as px
import os

# Passo 1: Carregar o dataset
file_path = "/Users/diogopinto/Documents/Pessoal/path_4med/snirh_scrape/bd_CHARTS/nivel_medio_diario/nivel_medio_diario.csv"
data = pd.read_csv(file_path, parse_dates=['data'])

# Renomear as colunas para corresponder ao que o código espera
data = data.rename(columns={
    "data": "date",
    "nome": "location",
    "codigo": "station_code",
    "nivel_medio_diario": "nivel_medio_diario",
    "coord_x_m": "x_coord",
    "coord_y_m": "y_coord"
})

# Passo 2: Limpar a coluna nivel_medio_diario
def clean_nivel(value):
    if pd.isna(value) or value == "":  # Se o valor for vazio ou NaN
        return None
    try:
        return float(value.replace("(vc)", ""))  # Remove o (vc) e converte para float
    except (ValueError, TypeError):
        return None  # Se não for possível converter, retorna None

# Aplica a função de limpeza
data["nivel_medio_diario"] = data["nivel_medio_diario"].apply(clean_nivel)

# Passo 3: Definir o diretório de saída para os gráficos
output_dir = "/Users/diogopinto/Documents/Pessoal/path_4med/snirh_scrape/bd_CHARTS/nivel_medio_diario/img_nivel_medio_diario"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Passo 4: Obter todas as localizações únicas
localizacoes = sorted(data['location'].unique())

# Variável a ser plotada
variavel = 'nivel_medio_diario'

# Passo 5: Loop para gerar e salvar um gráfico para cada localização
for loc in localizacoes:
    # Filtrar os dados para a localização atual
    filtered_data = data[data['location'] == loc]

    # Ignorar se não houver dados de nivel_medio_diario (valores não nulos)
    if filtered_data[variavel].isna().all():
        print(f"Sem dados de {variavel} para {loc}, pulando o gráfico.")
        continue

    # Criar o gráfico de linhas usando plotly.express
    fig = px.line(
        filtered_data,
        x='date',
        y=variavel,
        title=f"Nível Médio Diário em {loc}"
    )

    # Atualizar o estilo das linhas
    fig.update_traces(
        line=dict(color='blue', width=2),  # Linha azul com espessura 2
        mode='lines+markers'  # Linhas com marcadores
    )

    # Configurar o layout da figura
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Nível Médio Diário (m)",
        showlegend=False  # Não precisa de legenda, pois é uma única variável
    )

    # Sanitizar o nome do arquivo substituindo caracteres inválidos
    safe_loc = loc.replace('/', '_').replace(' ', '_')  # Substitui '/' e espaços por '_'
    output_path = os.path.join(output_dir, f"nivel_medio_diario_{safe_loc}.png")

    # Salvar o gráfico como PNG
    fig.write_image(output_path, format="png", width=1200, height=600)

    print(f"Gráfico salvo: {output_path}")

print("Todos os gráficos foram gerados e salvos!")