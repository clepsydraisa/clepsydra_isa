import pandas as pd
import plotly.express as px
import os

# Ler o arquivo CSV
file_path = "/Users/diogopinto/Documents/Pessoal/path_4med/snirh_scrape/bd_CHARTS/precipitacao/bd_precipitacao.csv"  # Substitua pelo caminho real do seu CSV
data = pd.read_csv(file_path, parse_dates=['data'])

# Garantir que a coluna de precipitação está no tipo numérico
data['precipitacao_dia_mm'] = pd.to_numeric(data['precipitacao_dia_mm'], errors='coerce')

# Definir valores fixos para o eixo Y com base no conjunto completo de dados
Y_MIN = data['precipitacao_dia_mm'].min() - 5  # Margem abaixo do mínimo global
Y_MAX = data['precipitacao_dia_mm'].max() + 5  # Margem acima do máximo global

# Definir o diretório de saída
output_dir = "/Users/diogopinto/Documents/Pessoal/path_4med/img_prec"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Obter todos os códigos únicos
codigos = sorted(data['codigo'].unique())

# Variável a ser plotada
variavel = 'precipitacao_dia_mm'

# Loop para gerar e salvar um gráfico para cada código
for codigo in codigos:
    # Filtrar os dados para o código atual
    filtered_data = data[data['codigo'] == codigo]

    # Obter o nome correspondente ao código (assumindo que o nome é consistente para cada código)
    nome = filtered_data['nome'].iloc[0]  # Pega o primeiro nome associado ao código

    # Criar o gráfico de linhas
    fig = px.line(
        filtered_data,
        x='data',
        y=variavel,
        line_shape='linear',
        title=f"Precipitação em {codigo} - {nome}"  # Título com código e nome
    )

    # Atualizar o estilo das linhas para preto
    fig.update_traces(
        line=dict(color='black', width=2)  # Linhas pretas com espessura 2
    )

    # Configurar o layout da figura com escala fixa no eixo Y
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Precipitação (mm)",
        showlegend=False,  # Não precisa de legenda, já que é uma única linha
        yaxis=dict(range=[Y_MIN, Y_MAX])  # Escala fixa baseada nos valores globais
    )

    # Substituir '/' por '_' no código para evitar problemas no nome do arquivo
    codigo_safe = codigo.replace('/', '_')

    # Salvar o gráfico como PNG
    output_path = os.path.join(output_dir, f"precipitacao_{codigo_safe}.png")
    fig.write_image(output_path, format="png", width=1200, height=600)

    print(f"Gráfico salvo: {output_path}")

print("Todos os gráficos foram gerados e salvos!")