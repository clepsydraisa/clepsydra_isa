Para forecasting em geral, ver livro (on-line) [Forecasting: Principles and Practice](https://otexts.com/fpp2/). Este livro usa o package *forecast* de R.

Para Python há duas possibilidades:

1. uma possibilidade é chamar as funções de *forecast* em Python com https://pypi.org/project/rpy2/. 

2. Usar packages alternativos em Python:
   - Análise descritiva de séries temporais: https://github.com/statsmodels/statsmodels/
   - Caso particular de *exponential smoothing* em Python:
      -  O package `tbats` em Python substitui *forecast* para modelos do tipo *exponential smoothing* -- ver https://github.com/intive-DataScience/tbats
      -  Para instalar o package para Python é necessário criar um *virtual environment* e instalar `tbats` com `pip install -r requirements_stable.txt` nesse *virtual environment*. O ficheiro foi adaptado de https://github.com/intive-DataScience/tbats/blob/master/requirements_stable.txt.

ver mais detalhes sobre métodos em (https://github.com/clepsydraisa/clepsydra_isa/tree/main/material_apoio/time_series)
