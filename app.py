import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from train_model import MinMax, OneHotEncodingNames, OrdinalFeatures

# ==========================================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================================
st.set_page_config(
    page_title="Análise e Predição de Obesidade",
    page_icon="🔬",
    layout="wide"
)

# Estilo CSS customizado
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# CARREGAMENTO DO MODELO E DADOS
# ==========================================================
@st.cache_resource
def load_model():
    with open("model/obesity_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("data/Obesity.csv")

try:
    model_data = load_model()
    df = load_data()
    
    # Componentes do pickle
    model = model_data["model"]
    transform_pipeline = model_data["transform_pipeline"]
    target_names = model_data["target_names"]
except Exception as e:
    st.error(f"Erro ao carregar modelo ou dados: {e}")
    st.stop()

# ==========================================================
# BARRA LATERAL - NAVEGAÇÃO
# ==========================================================
st.sidebar.title("Navegação")
page = st.sidebar.radio("Ir para:", ["Avaliação de Risco de Obesidade", "DashBoard Visão Analitica", "Métricas do Modelo"])

# ==========================================================
# PÁGINA: PREDIÇÃO
# ==========================================================
if page == "Avaliação de Risco de Obesidade":
    st.title("🔬 Predição de Obesidade")
    
    st.subheader("Preencha os dados do paciente")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            Gender = st.selectbox("Gênero", ["Female", "Male"])
            Age = st.number_input("Idade", 1, 100, 25)
            Height = st.number_input("Altura (m)", 1.0, 2.5, 1.70)
            Weight = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)
        
        with col2:
            family_history = st.selectbox("Histórico familiar de sobrepeso", ["yes", "no"])
            FAVC = st.selectbox("Consumo frequente de alimentos calóricos", ["yes", "no"])
            FCVC = st.slider("Consumo de vegetais (1-3)", 1.0, 3.0, 2.0)
            NCP = st.slider("Número de refeições por dia", 1.0, 4.0, 3.0)
        
        with col3:
            CAEC = st.selectbox("Alimentos entre refeições", ["no", "Sometimes", "Frequently", "Always"])
            SMOKE = st.selectbox("Fumante", ["yes", "no"])
            CH2O = st.slider("Consumo de água (L/dia)", 1.0, 3.0, 2.0)
            SCC = st.selectbox("Monitoramento de calorias", ["yes", "no"])
            
        col4, col5 = st.columns(2)
        with col4:
            FAF = st.slider("Atividade física (dias/semana)", 0.0, 3.0, 1.0)
            TUE = st.slider("Uso de tecnologia (horas/dia)", 0.0, 2.0, 1.0)
        with col5:
            CALC = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
            MTRANS = st.selectbox("Meio de transporte", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

        submit = st.form_submit_button("Realizar Predição")

    if submit:
        try:
            input_df = pd.DataFrame([{
                "Gender": 1 if Gender == "Male" else 0,
                "Age": Age,
                "Height": Height,
                "Weight": Weight,
                "family_history": family_history,
                "FAVC": FAVC,
                "FCVC": round(FCVC),
                "NCP": round(NCP),
                "CAEC": CAEC,
                "SMOKE": SMOKE,
                "CH2O": round(CH2O),
                "SCC": SCC,
                "FAF": round(FAF),
                "TUE": round(TUE),
                "CALC": CALC,
                "MTRANS": MTRANS
            }])

            # Aplica transformação
            input_transformed = transform_pipeline.transform(input_df)
            
            # Predição
            pred_numeric = model.predict(input_transformed)[0]
            pred_class = target_names[int(pred_numeric)]
            probabilities = model.predict_proba(input_transformed)[0]

            st.success(f"### Classificação prevista: **{pred_class}**")

            prob_df = pd.DataFrame({
                "Classe": target_names,
                "Probabilidade": probabilities
            }).sort_values("Probabilidade", ascending=False)

            st.subheader("Distribuição de Probabilidades")
            fig = px.bar(prob_df, x='Probabilidade', y='Classe', orientation='h', 
                         color='Probabilidade', color_continuous_scale='Viridis',
                         text_auto='.2%')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao realizar predição: {e}")

# ==========================================================
# PÁGINA: VALIDAÇÃO DE DADOS
# ==========================================================
elif page == "DashBoard Visão Analitica":
    st.title("📊 Validação e Análise do Dataset")
    st.write("""
    Esta página apresenta um dashboard técnico com as informações obtidas do arquivo `Obesity.csv`. 
    """)
    
    # Métricas Rápidas
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total de Registros", len(df))
    m2.metric("Total de Colunas", len(df.columns))
    m3.metric("Valores Ausentes", df.isnull().sum().sum())
    m4.metric("Duplicatas", df.duplicated().sum())

    st.divider()

    # Seção 1: Distribuição da Variável Alvo
    st.subheader("1. Equilíbrio das Classes (Variável Alvo)")
    st.write("É fundamental verificar se as classes de obesidade estão equilibradas para evitar viés no modelo.")
    
    fig_target = px.pie(df, names='Obesity', title='Distribuição dos Níveis de Obesidade',
                        hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_target, use_container_width=True)

    # Seção 2: Análise de Correlação e Dispersão
    st.subheader("2. Relação entre Peso, Altura e Idade")
    st.write("O IMC é derivado de Peso e Altura. Este gráfico mostra como essas variáveis se agrupam por nível de obesidade.")
    
    fig_scatter = px.scatter(df, x="Height", y="Weight", color="Obesity", size="Age",
                             hover_data=['Gender'], title="Dispersão: Peso vs Altura (Tamanho = Idade)",
                             color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Seção 3: Comportamento dos Hábitos
    st.subheader("3. Análise de Hábitos por Nível de Obesidade")
    col_hab1, col_hab2 = st.columns(2)
    
    with col_hab1:
        st.write("**Consumo de Alimentos Calóricos (FAVC) vs Obesidade**")
        fig_favc = px.histogram(df, x="Obesity", color="FAVC", barmode="group",
                                title="Impacto do Consumo de Alimentos Calóricos")
        st.plotly_chart(fig_favc, use_container_width=True)
        
    with col_hab2:
        st.write("**Atividade Física (FAF) por Nível de Obesidade**")
        fig_faf = px.box(df, x="Obesity", y="FAF", color="Obesity",
                         title="Distribuição de Atividade Física")
        st.plotly_chart(fig_faf, use_container_width=True)

    # Seção 4: Estatísticas Descritivas
    st.subheader("4. Estatísticas Descritivas")
    st.write("Resumo estatístico das variáveis numéricas do dataset.")
    st.dataframe(df.describe().T, use_container_width=True)

# ==========================================================
# PÁGINA: MÉTRICAS DO MODELO
# ==========================================================
elif page == "Métricas do Modelo":
    st.title("📈 Performance e Validação dos Modelos")
    st.write("Resultados das métricas de validação cruzada e importância das variáveis para as decisões do modelo.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("XGBoost (Modelo Principal)")
        st.metric("Acurácia de Teste", f"{model_data.get('xgb_metrics', {}).get('accuracy', 0):.2%}")
        st.metric("Média Validação Cruzada (K=5)", f"{model_data.get('xgb_metrics', {}).get('cv_mean', 0):.2%}")
        
    with col2:
        st.subheader("Random Forest (Comparativo)")
        st.metric("Acurácia de Teste", f"{model_data.get('rf_metrics', {}).get('accuracy', 0):.2%}")
        st.metric("Média Validação Cruzada (K=5)", f"{model_data.get('rf_metrics', {}).get('cv_mean', 0):.2%}")
    
    st.divider()
    
    # Feature Importance
    st.subheader("🔍 O que o modelo mais considera?")
    st.write("Abaixo estão as 10 variáveis que mais impactaram as predições do XGBoost.")
    
    importances = model_data.get('xgb_importances', {})
    if importances:
        imp_df = pd.DataFrame({
            'Variável': list(importances.keys()),
            'Importância': list(importances.values())
        }).sort_values('Importância', ascending=True)
        
        fig_imp = px.bar(imp_df, x='Importância', y='Variável', orientation='h',
                         color='Importância', color_continuous_scale='Blues',
                         text_auto='.4f')
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("Dados de importância das variáveis não encontrados.")
    
    # Tabela de Comparação Final
    st.subheader("Comparação Detalhada")
    comparison_df = pd.DataFrame({
        'Métrica': ['Acurácia (Hold-out)', 'Média CV (5-folds)'],
        'XGBoost': [f"{model_data.get('xgb_metrics', {}).get('accuracy', 0):.4f}", f"{model_data.get('xgb_metrics', {}).get('cv_mean', 0):.4f}"],
        'Random Forest': [f"{model_data.get('rf_metrics', {}).get('accuracy', 0):.4f}", f"{model_data.get('rf_metrics', {}).get('cv_mean', 0):.4f}"]
    })
    st.table(comparison_df)