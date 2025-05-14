import pandas as pd
import plotly.express as px
import re  # Importando o módulo re
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Carregando o arquivo CSV do caminho especificado
df = pd.read_csv("/workspaces/clepsydra_isa/scripts/caudal/caudal.csv")

# Convertendo a coluna de data para datetime
df['data'] = pd.to_datetime(df['data'])

# Função para extrair apenas os números da string
def extract_numeric(value):
    if pd.isna(value):
        return float('nan')
    match = re.search(r'-?\d*\.?\d+', str(value))  # Procura por números (com ou sem decimais)
    if match:
        return float(match.group())
    return float('nan')  # Retorna NaN se não encontrar número

# Aplicando a função à coluna caudal_médio_diário(m3/s)
df['caudal_médio_diário(m3/s)'] = df['caudal_médio_diário(m3/s)'].apply(extract_numeric)

# Criando o gráfico interativo
fig = px.line(df, 
              x="data", 
              y="caudal_médio_diário(m3/s)", 
              color="localizacao",
              title="Caudal Médio Diário por Localização",
              labels={
                  "data": "Data",
                  "caudal_médio_diário(m3/s)": "Caudal Médio Diário (m³/s)",
                  "localizacao": "Localização"
              })

# Atualizando o layout para adicionar interatividade
fig.update_layout(
    updatemenus=[
        dict(
            type="dropdown",
            direction="down",
            active=0,
            x=1.1,
            y=1.1,
            buttons=list([
                dict(
                    args=[{"visible": [True for _ in df['localizacao'].unique()]}],
                    label="Todas as Localizações",
                    method="update"
                )
            ] + [
                dict(
                    args=[{"visible": [loc == l for loc in df['localizacao'].unique()]}],
                    label=l,
                    method="update"
                ) for l in df['localizacao'].unique()
            ])
        ),
    ],
    # Permitindo múltiplas seleções na legenda
    legend=dict(
        title="Localizações",
        tracegroupgap=0,
    ),
    # Adicionando opções de zoom e pan
    xaxis=dict(
        rangeslider=dict(visible=True),
        type="date"
    )
)

# Configurando para abrir em localhost
fig.show()