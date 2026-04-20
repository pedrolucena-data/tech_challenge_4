# Tech Challenge 4 

Pós graduação em Data Analytics: Projeto da fase 2.
Tema: Data viz and production models.

Autor: **Pedro Lopes de Lucena** - RM 367572

---

## Índice

- [Contexto do Projeto](#contexto-do-projeto)
- [Objetivos](#objetivos)
- [Dicionário de Dados](#dicionário-de-dados)
- [Classes de Obesidade Preditas](#classes-de-obesidade-preditas)
- [Arquivos no Repositório](#arquivos-no-repositório)
- [Ferramentas e Tecnologias Utilizadas](#ferramentas-e-tecnologias-utilizadas)

## Contexto do Projeto

Você foi contratado como cientista de dados de um hospital
e tem o desafio de desenvolver um modelo de Machine Learning 
para auxiliar os médicos e médicas a prever se uma pessoa pode ter obesidade.
Utilizando a base de dados disponibilizada neste desafio em **obesity.csv**,
desenvolva um modelo preditivo e crie um sistema preditivo para auxiliar a 
tomada de decisão da equipe médica a diagnosticar a obesidade.

## Objetivos
- **Classificação Precisa:** Utilizar algoritmos de ponta para classificar o nível de obesidade com alta acurácia.
- **Análise de Fatores de Risco:** Identificar quais hábitos e características demográficas mais influenciam no ganho de peso.
- **Ferramenta de Apoio à Decisão:** Fornecer uma interface intuitiva para que médicos e nutricionistas possam realizar predições rápidas.

## Dicionário de Dados
| Variável | Descrição | Tipo |
| :--- | :--- | :--- |
| Gender | Gênero (0: Female, 1: Male) | Binário |
| Age | Idade em anos | Numérico |
| Height | Altura em metros | Numérico |
| Weight | Peso em quilogramas | Numérico |
| family_history | Histórico familiar de excesso de peso | Binário |
| FAVC | Consumo frequente de alimentos calóricos | Binário |
| FCVC | Frequência de consumo de vegetais | Numérico |
| NCP | Refeições principais por dia | Numérico |
| CAEC | Alimentação entre refeições | Categórico |
| SMOKE | Tabagismo | Binário |
| CH2O | Consumo de água (L/dia) | Numérico |
| SCC | Monitoramento de calorias | Binário |
| FAF | Frequência de atividade física | Numérico |
| TUE | Tempo de uso de dispositivos (h/dia) | Numérico |
| CALC | Frequência de consumo de álcool | Categórico |
| MTRANS | Meio de transporte | Categórico |

## Classes de Obesidade Preditas
| Classe | Descrição |
| :--- | :--- |
| **Normal_Weight** | Peso dentro da faixa saudável |
| **Insufficient_Weight** | Peso abaixo do ideal |
| **Overweight_Level_I** | Sobrepeso grau I |
| **Overweight_Level_II** | Sobrepeso grau II |
| **Obesity_Type_I** | Obesidade grau I |
| **Obesity_Type_II** | Obesidade grau II |
| **Obesity_Type_III** | Obesidade grau III (mórbida) |

## Arquivos no Repositório

- `train_model.py`: Motor de modelo oficial e script de treinamento.
- `app.py`: Aplicação principal Streamlit.
- `data/obesity.csv`: Dataset de treinamento.
- `requirements.txt`: Arquivo de texto com as bibliotecas necessárias para executar o código.

## Ferramentas e Tecnologias Utilizadas

- **Python**: Linguagem de programação principal utilizada no projeto.
- **Streamlit**: Interface Web.
- **Scikit-Learn**: Utilizado para a construção e avaliação dos modelos de Machine Learning.
- **XGBoost**: O Modelo Preditivo utilizado no projeto.
- **Plotly**: Para gerar Visualizações Interativas.
