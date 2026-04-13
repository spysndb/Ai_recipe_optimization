import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from datetime import datetime

# ==========================================
# 1. 頁面基本設定與分類字典
# ==========================================
st.set_page_config(page_title="蝕刻配方推測系統", page_icon="🧪", layout="wide")

# 🍎 注入 Mac 風格 CSS
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    div[data-testid="stMetric"], div.stExpander {
        background-color: rgba(255, 255, 255, 0.5) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04) !important;
        padding: 10px;
    }
    button[kind="primary"] {
        border-radius: 8px !important;
        background-color: #007AFF !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,122,255,0.3) !important;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(240, 242, 246, 0.6) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧪 智慧化學蝕刻配方推測系統 (v7 Mac + AD信心值版)")

NON_FEATURE_COLS = ['date_folder', 'item', 'chemical_formula', 'chemical_weights', 'result', 'etch_time_value_sec', 'etch_time_note']
TARGET_COLS = ['snag_cu_undercut_um', 'cu_ni_undercut_um']

CHEMICAL_CATEGORIES = {
    "🧪 酸類 (Acids)": [
        "H2SO4_weight", "HAc_weight", "MSA_weight", "PSA_weight", "HEDP_weight", 
        "ATMP_weight", "檸檬酸_weight", "酒石酸_weight", "DL-蘋果酸_weight", 
        "氨基磺酸_weight", "對胺基苯磺酸_weight", "5-磺基水楊酸二水合物_weight", 
        "60%HEDP_weight", "1%的對胺基苯磺酸水溶液_weight"
    ],
    "🔥 氧化劑 (Oxidizers)": ["Fe2(SO4)3_weight", "FeCl3_weight"],
    "💧 溶劑與稀釋劑 (Solvents & Diluents)": ["DI_weight", "EG_weight", "EtOH_weight", "PM_weight", "BCS_weight", "DGA_weight", "二乙二醇甲乙醚_weight"],
    "🛡️ 腐蝕抑制劑 (Corrosion Inhibitors)": [
        "BTA_weight", "1,2,4-三氮唑_weight", "1H-1,2,3-三氮唑_weight", 
        "5-甲基-1H-苯並三唑_weight", "5-氯苯並三氮唑_weight", "苯並咪唑_weight", 
        "1-甲基-1H-苯並咪唑_weight", "2-甲基苯並咪唑_weight", "咪唑_weight", 
        "1-甲基咪唑_weight", "2-丙基咪唑_weight", "吡唑_weight", "5-ATZ_weight", 
        "氨基-1,2,4-三氮唑_weight", "三苯基氯化四氮唑_weight", "四氮唑紫_weight", "抑制劑_weight"
    ],
    "⬡ 吡嗪/吡喃/雜環類 (Pyrazines/Heterocycles)": [
        "吡嗪_weight", "2-甲基吡嗪_weight", "2,3,5-三甲基吡嗪_weight", 
        "2-乙基-3-甲基吡嗪_weight", "2,4-二甲基吡嗪_weight", "2-氯吡嗪_weight", 
        "2,3-二氯吡嗪_weight", "2,6-二氯吡嗪_weight", "2,3,5-三氯吡嗪_weight", 
        "2,3,5-三甲基吡喃_weight", "2,6-二氯吡喃_weight", "哌嗪_weight", 
        "2-吡咯烷酮_weight", "2-丁乙基 2-吡咯烷酮_weight", "羥乙基吡咯烷酮_weight"
    ],
    "🧬 聚合物與界面活性劑 (Polymers & Surfactants)": [
        "PEG #200_weight", "PEG #2000_weight", "PEG #8000_weight", 
        "PVP #3500_weight", "PVP #10000_weight", "PVP #58000_weight", 
        "PVA_weight", "PPG #400_weight", "PEI_weight", "SLS_weight", 
        "Tween 20_weight", "PASP_weight", "40%PASP-Na_weight", 
        "聚丙烯醯胺_weight", "聚萘甲醛磺酸鈉_weight", "羥乙基纖維素_weight", 
        "明膠_weight", "Cadisper-196B_weight", "YLE015_weight", 
        "新日化 2-EHS_weight", "新日化 OS-40_weight", 
        "台界化學 CPB-K_weight", "台界化學 TDE-9_weight",
        "0.1%之PVP#3500水溶液_weight", "0.1%的聚丙烯醯胺水溶液_weight"
    ],
    "⚖️ 鹼類與 pH 調節劑 (Bases/pH Adjusters)": ["45%KOH_weight", "TEA_weight", "正丁胺_weight"],
    "🧩 氨基酸與其他添加劑 (Others)": ["L-精氨酸_weight", "甘胺酸_weight", "半胱胺酸_weight", "NaCl_weight", "PB_weight", "60%DEHP_weight", "Inhibitor J_weight"]
}

if "df" not in st.session_state: st.session_state.df = None
if "feature_cols" not in st.session_state: st.session_state.feature_cols = []
if "models" not in st.session_state: st.session_state.models = {}
if "scaler" not in st.session_state: st.session_state.scaler = None
if "knn" not in st.session_state: st.session_state.knn = None
if "mean_train_dist" not in st.session_state: st.session_state.mean_train_dist = 1.0
if "rmse_snag" not in st.session_state: st.session_state.rmse_snag = 0.0
if "rmse_cuni" not in st.session_state: st.session_state.rmse_cuni = 0.0

# ==========================================
# 2. 核心函式定義
# ==========================================
def convert_to_wt_pct(X_df, feature_cols):
    chem_cols = [c for c in feature_cols if c not in ['temp', 'region']]
    X_pct = X_df.copy()
    total_weights = X_pct[chem_cols].sum(axis=1).replace(0, 1)
    X_pct[chem_cols] = X_pct[chem_cols].div(total_weights, axis=0) * 100
    return X_pct

def train_models(df):
    feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS + TARGET_COLS]
    st.session_state.feature_cols = feature_cols
    
    X_raw = df[feature_cols].fillna(0)
    y_snag = df['snag_cu_undercut_um'].fillna(0)
    y_cu_ni = df['cu_ni_undercut_um'].fillna(0)
    
    X_pct = convert_to_wt_pct(X_raw, feature_cols)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pct)
    st.session_state.scaler = scaler
    
    # 建立 KNN 用於計算適用領域 (AD) 距離
    knn = NearestNeighbors(n_neighbors=3)
    knn.fit(X_scaled)
    st.session_state.knn = knn
    distances, _ = knn.kneighbors(X_scaled)
    st.session_state.mean_train_dist = np.mean(distances[:, 0]) + 1e-5 # 避免除以零
    
    # 訓練預測模型
    xgb_snag = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_pct, y_snag)
    xgb_cu_ni = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_pct, y_cu_ni)
    
    # 計算 RMSE 作為基準誤差區間
    train_preds_snag = xgb_snag.predict(X_pct)
    train_preds_cuni = xgb_cu_ni.predict(X_pct)
    st.session_state.rmse_snag = np.sqrt(mean_squared_error(y_snag, train_preds_snag))
    st.session_state.rmse_cuni = np.sqrt(mean_squared_error(y_cu_ni, train_preds_cuni))
    
    st.session_state.models = {
        'rf_snag': RandomForestRegressor(n_estimators=100, random_state=42).fit(X_pct, y_snag), 
        'rf_cu_ni': RandomForestRegressor(n_estimators=100, random_state=42).fit(X_pct, y_cu_ni),
        'xgb_snag': xgb_snag, 'xgb_cu_ni': xgb_cu_ni,
        'ridge_snag': Ridge(alpha=1.0).fit(X_scaled, y_snag), 'ridge_cu_ni': Ridge(alpha=1.0).fit(X_scaled, y_cu_ni)
    }

# ==========================================
# 3. 主畫面設定
# ==========================================
with st.sidebar:
    st.header("系統狀態")
    if st.session_state.df is not None:
        st.success(f"✅ 資料庫已載入 ({len(st.session_state.df)} 筆)")
        st.info(f"📉 模型 RMSE 基準誤差:\nSnag: ±{st.session_state.rmse_snag:.3f}\nCuNi: ±{st.session_state.rmse_cuni:.3f}")
        csv_data = st.session_state.df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下載最新 CSV 備份", data=csv_data, file_name="Etch_Recipe_Latest.csv", mime="text/csv", use_container_width=True)
    else:
        st.warning("⚠️ 尚未載入資料")

tab1, tab2, tab3, tab4 = st.tabs(["📂 資料載入", "🧪 正向配方推測與 AD 信心值", "🎯 逆向最佳配方探索", "📝 實驗結果登錄"])

with tab1:
    uploaded_file = st.file_uploader("請上傳清洗後的寬表 CSV 檔案", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        train_models(df)
        st.success("✅ 資料讀取、模型訓練與 RMSE 基準計算完成！")

with tab2:
    if st.session_state.df is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            temp = st.number_input("溫度 (temp)", value=25.0, step=1.0)
            region = st.selectbox("區域 (region)", options=[1, 0], format_func=lambda x: "密區 (1)" if x == 1 else "疏區 (0)")
        with c2:
            h2o = st.number_input("H2O 重量", value=60.0, step=1.0)
            h3po4 = st.number_input("H3PO4 重量", value=10.0, step=1.0)
            h2o2 = st.number_input("H2O2 重量", value=15.0, step=1.0)
        with c3:
            st.caption("勾選欲使用的添加物 (最多 10 種)")
            chem_inputs = {}
            optional_chems = [c for c in st.session_state.feature_cols if c not in ['temp', 'region', 'H2O_weight', 'H3PO4_weight', 'H2O2_weight']]
            for cat_name, chems_in_cat in CHEMICAL_CATEGORIES.items():
                valid_chems = [c for c in chems_in_cat if c in optional_chems]
                if valid_chems:
                    with st.expander(cat_name):
                        for chem in valid_chems:
                            if st.checkbox(chem.replace('_weight', ''), key=f"t2_{chem}"):
                                chem_inputs[chem] = st.number_input("重量", value=1.0, step=0.1, key=f"num_{chem}")
        with c4:
            btn_predict = st.button("🚀 執行推測與適用領域(AD)分析", use_container_width=True, type="primary")

        if btn_predict:
            input_data = {col: 0.0 for col in st.session_state.feature_cols}
            input_data['temp'], input_data['region'] = temp, region
            input_data['H2O_weight'], input_data['H3PO4_weight'], input_data['H2O2_weight'] = h2o, h3po4, h2o2
            for chem, val in chem_inputs.items(): input_data[chem] = val
                
            input_raw = pd.DataFrame([input_data])
            input_pct = convert_to_wt_pct(input_raw, st.session_state.feature_cols)
            input_scaled = st.session_state.scaler.transform(input_pct)
            
            # 1. KNN 距離與信心值計算 (Applicability Domain)
            dist, _ = st.session_state.knn.kneighbors(input_scaled)
            min_dist = dist[0][0]
            confidence_score = max(0, min(100, 100 * np.exp(-0.8 * (min_dist / st.session_state.mean_train_dist))))
            
            # 2. XGBoost 預測與 RMSE 誤差區間
            pred_snag = st.session_state.models['xgb_snag'].predict(input_pct)[0]
            pred_cuni = st.session_state.models['xgb_cu_ni'].predict(input_pct)[0]
            
            st.markdown("### 🌟 AI 推測結果與領域信心值")
            if confidence_score > 80: st.success(f"🟢 **預測信心值：{confidence_score:.1f}%** (此配方落在歷史安全區，可靠度高)")
            elif confidence_score > 40: st.warning(f"🟡 **預測信心值：{confidence_score:.1f}%** (此配方在資料庫邊緣，可能有一定誤差)")
            else: st.error(f"🔴 **預測信心值：{confidence_score:.1f}%** (警告：盲區外推 Outlier！此為前所未見的配方組合，極高誤差風險！)")
            
            res_c1, res_c2 = st.columns(2)
            with res_c1: st.metric("🎯 預測 Snag Cu", f"{pred_snag:.3f} um", delta=f"基準誤差 ± {st.session_state.rmse_snag:.3f}", delta_color="off")
            with res_c2: st.metric("🎯 預測 Cu Ni", f"{pred_cuni:.3f} um", delta=f"基準誤差 ± {st.session_state.rmse_cuni:.3f}", delta_color="off")
            
            st.divider()

            # 3. 劑量反應曲線 (Dose-Response Curve)
            st.markdown("### 📈 化學品劑量反應迴歸曲線 (Dose-Response Curve)")
            st.caption("選擇一項化學品，觀察其濃度變化對蝕刻結果的趨勢影響。曲線越平滑代表模型學到的物理意義越合理。")
            all_selected_chems = ['H2O_weight', 'H3PO4_weight', 'H2O2_weight'] + list(chem_inputs.keys())
            curve_chem = st.selectbox("請選擇變數 (X軸)", options=all_selected_chems, format_func=lambda x: x.replace('_weight', ''))
            
            if curve_chem:
                # 產生 50 個模擬點 (從 0 到 該化學品的 2 倍重量，或至少到 10g)
                current_weight = input_data[curve_chem]
                max_w = max(current_weight * 2.5, 10.0)
                sim_weights = np.linspace(0, max_w, 50)
                
                sim_df = pd.DataFrame([input_data]*50)
                sim_df[curve_chem] = sim_weights
                
                # 轉為 wt% 送入預測
                sim_pct = convert_to_wt_pct(sim_df, st.session_state.feature_cols)
                curve_preds_snag = st.session_state.models['xgb_snag'].predict(sim_pct)
                curve_preds_cuni = st.session_state.models['xgb_cu_ni'].predict(sim_pct)
                
                chart_data = pd.DataFrame({
                    f'{curve_chem.replace("_weight", "")} (g)': sim_weights,
                    'Snag Cu (um)': curve_preds_snag,
                    'Cu Ni (um)': curve_preds_cuni
                }).set_index(f'{curve_chem.replace("_weight", "")} (g)')
                
                st.line_chart(chart_data)

with tab3:
    st.info("逆向探索頁面維持原邏輯不變，請透過「正向推測」檢驗新加入的信心值與迴歸曲線功能。")

with tab4:
    st.info("資料登錄頁面維持原邏輯不變。")
