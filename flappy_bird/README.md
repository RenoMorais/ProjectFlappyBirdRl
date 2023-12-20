# Flappy Bird 

Esse diretório contém a implementação do algoritmo DQN para o Flappy Bird, usando o environment [flappy-bird-gym](https://github.com/Talendar/flappy-bird-gym). 


### Requirements
- Python 3.10.3

### Como executar

- #### Usando [pyenv](https://github.com/pyenv/pyenv) (ou no Windows [pyenv-win](https://github.com/pyenv-win/pyenv-win)):

1. Instalação do python 3.10.3:

    ```pyenv install 3.10.3```

2. Seleção do python 3.10.3 instalado anteriormente para o diretório atual:

    ```pyenv local 3.10.3```

3. Criação de um ambiente virtual (venv):

    ```python -m venv <NOME_DO_VENV>```

    Ex.:
    ```python -m venv venv```

4. Ativação do venv criado no passo 3:

    - No Linux ou Mac:

        ```source <NOME_DO_VENV>/bin/activate```

        Ex.:
        ```source venv/bin/activate```

    - No Windows:

        ```<NOME_DO_VENV>\Scripts\Activate```

5. Instalação do ambiente Flappy Bird (gymnasium)

    ```pip install -e flappy_bird_gymnasium```

6. Instalação dos demais pacotes usados

    ```pip install -r requirements.txt```

    Obs.: esse comando instala os pacotes listados no arquivo `requirements.txt`, que está presente dentro desse repositório.

7. Treinar o algoritmo:

    ```python main_flappy_bird.py --train```

    Esse passo vai salvar uma pasta `logs/` contendo:
    - Um checkpoint (.zip), com os pesos salvos do modelo treinado a cada 100K de iterações (`logs/rl_model_100000_steps.zip`, `logs/rl_model_200000_steps.zip`, ...)
    - O melhor modelo avaliado durante as etapas de avaliação (`logs/best_model.zip`)
    - O resultado de todas as avaliações executadas (`logs/evaluations.npz`)

8. Testar o modelo treinado:

    ```python main_flappy_bird.py --test <PATH_PARA_ZIP_MODELO_TREINADO>```

    Ex.: 

    ```python main_flappy_bird.py --test logs/best_model.zip```

9. Plotar o gráfico do reward total obtido durante as etapas de avaliação:

    ```python main_flappy_bird.py --plot <PATH_PARA_ARQUIVO_DE_EVALUATIONS> -n <NUMERO_TIMESTEPS_TREINO>```

    Ex.: 

    ```python main_flappy_bird.py --plot logs/evaluations.npz -n 10000000```

### Help CLI

```shell
$ python main_flappy_bird.py --help

usage: main_flappy_bird.py [-h] [--train] [--test TEST_FILEPATH] [--plot PLOT_FILEPATHS [PLOT_FILEPATHS ...]] [-n N_STEPS]

RL Agent for Flappy Bird.

options:
  -h, --help            show this help message and exit
  --train               Train algorithm (default: False)
  --test TEST_FILEPATH  Test algorithm on the trained model saved in `TEST_FILEPATH` (default: None)
  --plot PLOT_FILEPATHS [PLOT_FILEPATHS ...]
                        Evaluations files. (default: None)
  -n N_STEPS            Nb. timesteps used in training (only used with the --plot option) (default: 100000000)
```