# root (working directory)
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose

# identificar pasta de trabalho
try:
    working_dir=Path(__file__).parent.parent # working directory from script location: scripts are in 'scripts' folder
except:
    working_dir=Path().absolute()

# Dados
codigos=['330/183', '331/15', '331/2', '341/17', '342/78', '342/97', '377/54', '377/59', '377/84', '377/86', '377/94', '390/208', '390/99', '391/243', '391/33', '391/437', '404/69', '405/17', '405/34', '418/15', '418/4']
codigo_poco='442/94'
codigo_poco='376/24'
fn= Path(working_dir) /  "resources" / "aquifer_depth_piezo.csv"

for codigo_poco in codigos:
    fn_out= Path("path4med") / "resources" / Path("aquifer_depth_piezo_"+codigo_poco.replace('/','_')+".csv")
    fn_out_fig= Path("path4med") / "figuras" / Path("aquifer_depth_piezo_"+codigo_poco.replace('/','_')+".jpg")
    # ler ficheiro piezometria
    df=pd.read_csv(fn)
    # filtar dados para o poço
    df=df[df['codigo']==codigo_poco]
    # construir serie temporal
    df['date'] = pd.to_datetime(df['data'])
    df = df.set_index('date')
    df['nivel'] = -pd.to_numeric(df['profundidade_nivel_agua'], errors='coerce') # ou 'nivel_piezometrico' ? (valores negativos)
    print(df.shape)
    df.to_csv(fn_out, index=True)
    #df = df.asfreq('MS')  # Aligns to month-end; use 'MS' for month-start
    if True :
        plt.figure(figsize=(10, 8))  # width=10 inches, height=8 inches
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
        plt.title('Nivel over Time'+codigo_poco)
        plt.savefig(fn_out_fig,dpi=300)
        plt.close()

if False:
    decompose = seasonal_decompose(df['nivel'], model='additive', period=365)
    decompose.plot()
    plt.show()
