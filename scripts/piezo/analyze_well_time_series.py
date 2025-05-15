# Manuel Campagnolo, May 15, 2025

# root (working directory)
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# identificar pasta de trabalho
try:
    working_dir=Path(__file__).parent.parent # working directory from script location: scripts are in 'scripts' folder
except:
    working_dir=Path().absolute()

# Dados
codigo_poco='442/94'
codigo_poco='376/24'
fn= Path(working_dir) /  "resources" / "aquifer_depth_piezo.csv"
fn_out= Path("path4med") / "resources" / Path("aquifer_depth_piezo_"+codigo_poco.replace('/','_')+".csv")
EXPORT=False

# ler ficheiro piezometria
df=pd.read_csv(fn)

# filtar dados para o poço
df=df[df['codigo']==codigo_poco]

# construir serie temporal
df['date'] = pd.to_datetime(df['data'])
df = df.set_index('date')
df['nivel'] = -pd.to_numeric(df['profundidade_nivel_agua'], errors='coerce') # ou 'nivel_piezometrico' ? (valores negativos)
print(df.shape)
if EXPORT:
    df.to_csv(fn_out, index=True)

#df = df.asfreq('MS')  # Aligns to month-end; use 'MS' for month-start

if True :
    # Plot the 'nivel' column
    plt.scatter(df.index, df['nivel'], color='blue', alpha=0.7)
    # Add vertical lines at the beginning of each year (January 1st)
    years = df.index.year.unique()
    for year in years:
        plt.axvline(pd.Timestamp(f'{year}-01-01'), color='grey', linestyle='-', linewidth=1)
    # Add green vertical lines at the start of each trimester: Jan, Apr, Jul, Oct
    trimesters = [1, 4, 7, 10]
    for year in years:
        for month in trimesters:
            plt.axvline(pd.Timestamp(f'{year}-{month:02d}-01'), color='black', linestyle=':', linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Nivel')
    plt.title('Nivel over Time')
    plt.show()

