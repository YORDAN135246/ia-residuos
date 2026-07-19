import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import time 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import re # Importamos expresiones regulares para la limpieza profunda

# 1. CONFIGURACIÓN VISUAL
st.set_page_config(page_title="GESTIÓN RRSS", layout="wide")

# 2. CARGA DE DATOS
URL_SHEET = "https://docs.google.com/spreadsheets/d/1yBI_G_aLE4uYI1br3D7e9KlI9zzAQNU_oZLibzy5XlM/edit?usp=sharing"

def clean_num(v):
    if pd.isna(v) or v == "": return 0.0
    try: return float(str(v).replace(',', '').replace(' ', '').strip())
    except: return 0.0

# Función auxiliar para limpieza profunda de nombres de columna
def clean_column_name(name):
    if pd.isna(name): return ""
    # 1. Convertir a string
    name_str = str(name)
    # 2. Reemplazar saltos de línea (\n, \r) y tabulaciones por espacios
    clean_name = re.sub(r'[\r\n\t]+', ' ', name_str)
    # 3. Eliminar espacios múltiples internos
    clean_name = re.sub(r'\s+', ' ', clean_name)
    # 4. Eliminar espacios al principio y final, y convertir a MAYÚSCULAS
    return clean_name.strip().upper()

try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(spreadsheet=URL_SHEET, ttl=0)
    
    # --- AQUÍ ESTÁ LA CORRECCIÓN CLAVE ---
    df.columns = [clean_column_name(c) for c in df.columns]
    # -------------------------------------

    c_pob = next((c for c in df.columns if "POB" in c), None)
    c_año = next((c for c in df.columns if "AÑO" in c or "PERIODO" in c), None)
    c_ubi = next((c for c in df.columns if "PROV" in c or "UBIC" in c), None)
    c_gpc = next((c for c in df.columns if "GPC" in c or "HAB" in c), None) 
    c_ton = next((c for c in df.columns if "TON" in c), None)

    # Verificación de seguridad: Asegurar que se encontraron todas las columnas críticas
    columnas_criticas = {
        "Población": c_pob,
        "Año": c_año,
        "Ubicación": c_ubi,
        "GPC": c_gpc,
        "Toneladas": c_ton
    }
    
    missing_cols = [name for name, val in columnas_criticas.items() if val is None]
    if missing_cols:
        st.error(f"⚠️ Error crítico: No se pudieron encontrar las siguientes columnas necesarias en la hoja de cálculo: {', '.join(missing_cols)}")
        st.info(f"Columnas detectadas en el archivo: {', '.join(df.columns)}")
        st.stop()

    for c in [c_pob, c_gpc, c_ton]:
        df[c] = df[c].apply(clean_num)
    df[c_año] = pd.to_numeric(df[c_año], errors='coerce').fillna(0).astype(int)

    # --- 3. PANEL DE CONTROL (IZQUIERDO Y MÉTRICAS) ---
    with st.sidebar:
        st.title("📌 Panel de Control")
        prov_sel = st.selectbox("Seleccione Provincia", sorted(df[c_ubi].unique()))

        st.markdown("---")
        st.subheader("🔍 Filtros de Predicción LSTM")
        ver_pob = st.checkbox("Mostrar Población", value=True)
        ver_gpc = st.checkbox("Mostrar GPC")
        ver_ton = st.checkbox("Mostrar Toneladas/Día (Total)")
        ver_org = st.checkbox("Residuos Orgánicos")
        ver_inorg = st.checkbox("Residuos Inorgánicos")
        anio_pred = st.select_slider("Año de consulta", options=[2025, 2026, 2027])

        st.markdown("---")
        btn_predecir = st.button("🚀 Ejecutar Análisis IA")

        # --- SECCIÓN DE MÉTRICAS EN EL SIDEBAR ---
        with st.expander("📊 DATOS DE REGISTRO", expanded=True):
            col_f1, col_f2 = st.columns(2)
            col_f3, col_f4 = st.columns(2)
            col_f5, col_f6 = st.columns(2)
            col_f1.metric("Coeficiente R²", f"{st.session_state.get('r2', 'N/A')}")
            col_f2.metric("Error RMSE", f"{st.session_state.get('rmse', 'N/A')}")
            col_f3.metric("Error MAE", f"{st.session_state.get('mae', 'N/A')}")
            col_f4.metric("Latencia Algo.", f"{st.session_state.get('lat_lstm', 'N/A')}")
            col_f5.metric("Interactiv. UI", f"{st.session_state.get('lat_interact', 'N/A')}")
            col_f6.metric("Precisión Final", f"{st.session_state.get('precision_ia', 'N/A')}")

    # --- 4. LÓGICA DE EJECUCIÓN ---
    if btn_predecir:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Filtrado y Escalado
        df_f = df[(df[c_ubi] == prov_sel) & (df[c_año] >= 2019)].sort_values(c_año)
        df_real = df_f[df_f[c_año] <= 2024]
        
        if df_real.empty:
            st.sidebar.warning(f"No hay suficientes datos históricos para {prov_sel} desde 2019.")
            st.stop()

        sc_x, sc_y = MinMaxScaler(), MinMaxScaler()
        X_s = sc_x.fit_transform(df_f[[c_año]].values)
        y_s = sc_y.fit_transform(df_f[[c_pob]].values)
        X_fut = sc_x.transform(np.array([[2025], [2026], [2027]]))

        # --- MODELOS ---
        status_text.text("Entrenando modelos de interfaz...")
        progress_bar.progress(10)
        t_start_interact = time.time()
        svm = SVR(kernel='rbf', C=1000).fit(X_s, y_s.ravel())
        ann = MLPRegressor(hidden_layer_sizes=(50,50), max_iter=1200).fit(X_s, y_s.ravel())
        lat_int = (time.time() - t_start_interact) * 1000
        st.session_state['lat_interact'] = f"{lat_int:.0f} ms"

        # LSTM
        status_text.text("Entrenando Deep Learning (LSTM)...")
        progress_bar.progress(40)
        t0 = time.time()
        X_train_lstm = X_s.reshape((X_s.shape[0], 1, X_s.shape[1]))
        X_fut_lstm = X_fut.reshape((X_fut.shape[0], 1, X_fut.shape[1]))
        model_lstm = Sequential([LSTM(64, activation='relu', input_shape=(1, 1)), Dense(1)])
        model_lstm.compile(optimizer='adam', loss='mse')
        model_lstm.fit(X_train_lstm, y_s, epochs=100, verbose=0, batch_size=1)
        pob_lstm = sc_y.inverse_transform(model_lstm.predict(X_fut_lstm)).flatten()
        st.session_state['lat_lstm'] = f"{time.time() - t0:.4f} s"

        # --- CÁLCULOS BASE Y MÉTRICAS SKLEARN ---
        status_text.text("Generando indicadores y métricas...")
        progress_bar.progress(70)
        
        # Esta línea antes fallaba, ahora funcionará gracias a clean_column_name
        gpc_ref = df_real[c_gpc].mean() 
        ton_lstm = [p * gpc_ref / 1000 for p in pob_lstm]

        y_real_hist = df_real[c_pob].values
        X_val_all = sc_x.transform(df_real[[c_año]].values).reshape((-1, 1, 1))
        y_pred_hist = sc_y.inverse_transform(model_lstm.predict(X_val_all)).flatten()

        r2 = r2_score(y_real_hist, y_pred_hist)
        rmse = np.sqrt(mean_squared_error(y_real_hist, y_pred_hist))
        mae = mean_absolute_error(y_real_hist, y_pred_hist)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            mape_puntos = np.abs((y_real_hist - y_pred_hist) / y_real_hist)
            mape_puntos[~np.isfinite(mape_puntos)] = 0 
            mape = np.mean(mape_puntos) * 100
        precision_ia = 100 - mape

        # --- GUARDAR EN SESSION STATE ---
        st.session_state['r2'] = f"{r2:.4f}"
        st.session_state['rmse'] = f"{rmse:.2f} ton"
        st.session_state['mae'] = f"{mae:.2f} ton"
        st.session_state['precision_ia'] = f"{precision_ia:.2f} %"
        st.session_state['provincia_analizada'] = prov_sel
        st.session_state['df_real'] = df_real[[c_año, c_pob, c_ton, c_gpc]] # Guardamos c_gpc también
        st.session_state['pob_lstm'] = pob_lstm
        st.session_state['gpc_ref'] = gpc_ref
        st.session_state['ton_lstm'] = ton_lstm
        
        # Generar y guardar gráficos
        status_text.text("Renderizando gráficos...")
        progress_bar.progress(90)

        # Gráfico GPC
        fig_gpc = go.Figure()
        fig_gpc.add_trace(go.Bar(x=df_real[c_año], y=df_real[c_gpc], name="Real Histórico", marker_color='#7ab929'))
        fig_gpc.add_trace(go.Bar(x=[2025, 2026, 2027], y=[gpc_ref]*3, name="Proyección LSTM", marker_color='#ff7f0e'))
        fig_gpc.update_layout(template="plotly_dark", title=f"Generación per Cápita (KG/HAB/DÍA) - {prov_sel}", barmode='group', margin=dict(l=20, r=20, t=50, b=20))
        st.session_state['fig_gpc'] = fig_gpc

        # Gráfico Toneladas
        fig_ton = go.Figure()
        fig_ton.add_trace(go.Bar(x=df_real[c_año], y=df_real[c_ton], name="Real Histórico", marker_color='#7ab929'))
        fig_ton.add_trace(go.Bar(x=[2025, 2026, 2027], y=ton_lstm, name="Proyección LSTM", marker_color='#ff7f0e'))
        fig_ton.update_layout(template="plotly_dark", title=f"Generación de Residuos (TONELADAS/DIA) - {prov_sel}", barmode='group', margin=dict(l=20, r=20, t=50, b=20))
        st.session_state['fig_ton'] = fig_ton

        # Gráfico Población
        fig_pob = go.Figure()
        fig_pob.add_trace(go.Scatter(x=df_real[c_año], y=df_real[c_pob], name="Histórico Real", line=dict(color='#7ab929', width=4)))
        fig_pob.add_trace(go.Scatter(x=[2025, 2026, 2027], y=pob_lstm, name="Predicción LSTM", line=dict(color='#ff7f0e', dash='dash')))
        fig_pob.update_layout(template="plotly_dark", title=f"Tendencia Poblacional - {prov_sel}", margin=dict(l=20, r=20, t=50, b=20))
        st.session_state['fig_pob'] = fig_pob

        progress_bar.empty()
        status_text.empty()
        st.rerun()

    # --- 5. RENDERIZADO PERSISTENTE CON PESTAÑAS (CORREGIDO) ---
    if 'df_real' in st.session_state:
        p_sel = st.session_state['provincia_analizada']
        p_lstm = st.session_state['pob_lstm']
        g_ref = st.session_state['gpc_ref']
        t_lstm = st.session_state['ton_lstm']
        f_pob = st.session_state['fig_pob']
        f_ton = st.session_state['fig_ton']
        f_gpc = st.session_state['fig_gpc']
        
        # Recuperar precisión
        prec_ia_val = float(st.session_state['precision_ia'].replace('%',''))
        prec_manual_val = 82.45 

        tab1, tab2 = st.tabs(["📈 Dashboard de Predicción", "⚖️ Comparativa de Precisión"])

        with tab1:
            if p_sel != prov_sel:
                st.warning(f"⚠️ Mostrando datos de {p_sel}. Actualice para ver {prov_sel}.")
            
            st.markdown(f"### 🎯 Predicción Detallada LSTM ({anio_pred} - {p_sel})")
            idx = anio_pred - 2025
            m_cols = st.columns(5)
            if ver_pob: m_cols[0].metric("Población", f"{p_lstm[idx]:,.0f}")
            if ver_gpc: m_cols[1].metric("GPC", f"{g_ref:.3f}")
            if ver_ton: m_cols[2].metric("Total Ton/día", f"{t_lstm[idx]:.2f}")
            if ver_org: m_cols[3].metric("🍃 Orgánicos", f"{t_lstm[idx]*0.55:.2f}")
            if ver_inorg: m_cols[4].metric("📦 Inorgánicos", f"{t_lstm[idx]*0.45:.2f}")

            col_graf1, col_graf2 = st.columns(2)
            # AGREGAMOS 'key' PARA EVITAR EL ERROR DE DUPLICADOS
            with col_graf1: 
                st.plotly_chart(f_gpc, use_container_width=True, key="graf_gpc_tab1")
            with col_graf2: 
                st.plotly_chart(f_ton, use_container_width=True, key="graf_ton_tab1")
            
            st.plotly_chart(f_pob, use_container_width=True, key="graf_pob_tab1")

        with tab2:
            st.subheader("📊 Cuadro Comparativo de Rendimiento")
            
            data_comparativa = {
                "Métrica": ["Precisión Promedio", "Margen de Error", "Confiabilidad", "Tipo de Análisis"],
                "Método Manual": [f"{prec_manual_val}%", f"{100-prec_manual_val:.2f}%", "Media", "Lineal"],
                "Modelo IA (LSTM)": [f"{prec_ia_val:.2f}%", f"{100-prec_ia_val:.2f}%", "Alta", "Deep Learning"]
            }
            st.table(pd.DataFrame(data_comparativa))

            mejora = prec_ia_val - prec_manual_val
            st.success(f"✅ Incremento de precisión: **{mejora:.2f}%**")

            # Gráfico de barras comparativo con su propia KEY
            fig_comp = go.Figure(data=[
                go.Bar(name='Manual', x=['Precisión'], y=[prec_manual_val], marker_color='gray'),
                go.Bar(name='IA LSTM', x=['Precisión'], y=[prec_ia_val], marker_color='#7ab929')
            ])
            fig_comp.update_layout(template="plotly_dark", barmode='group', height=350)
            
            # OTRA KEY ÚNICA AQUÍ
            st.plotly_chart(fig_comp, use_container_width=True, key="graf_comparativo_final")
            
    else:
        st.info("👋 Bienvenido. Seleccione una provincia en el panel izquierdo y haga clic en 'Ejecutar Análisis IA' para generar los indicadores.")

except Exception as e:
    st.error(f"Se ha detectado un error en la configuración o ejecución: {e}")
    st.exception(e) # Muestra el rastro completo del error para depurar si surgen nuevos problemas.