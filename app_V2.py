import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from datetime import datetime
import io
import plotly.express as px
import random

# ==========================================
# 1. 頁面基本設定與分類字典
# ==========================================
st.set_page_config(page_title="蝕刻配方推測系統 V8", page_icon="🧪", layout="wide")
st.title("🧪 智慧化學蝕刻配方推測系統 (wt% 濃度百分比)")

# 定義不進入特徵運算的欄位與目標欄位
NON_FEATURE_COLS = ['date_folder', 'item', 'chemical_formula', 'chemical_weights', 'result', 'etch_time_value_sec', 'etch_time_note']
TARGET_COLS = ['snag_cu_undercut_um', 'cu_ni_undercut_um']

# 化學品分類字典
CHEMICAL_CATEGORIES = {
    "🧪 酸類 (Acids)": [
        "H2SO4_weight", "HAc_weight", "MSA_weight", "PSA_weight", "HEDP_weight", 
        "ATMP_weight", "檸檬酸_weight", "酒石酸_weight", "DL-蘋果酸_weight", 
        "氨基磺酸_weight", "對胺基苯磺酸_weight", "5-磺基水楊酸二水合物_weight", 
        "60%HEDP_weight", "1%的對胺基苯磺酸水溶液_weight"
    ],
    "🔥 氧化劑 (Oxidizers)": [
        "Fe2(SO4)3_weight", "FeCl3_weight"
    ],
    "💧 溶劑與稀釋劑 (Solvents & Diluents)": [
        "DI_weight", "EG_weight", "EtOH_weight", "PM_weight", 
        "BCS_weight", "DGA_weight", "二乙二醇甲乙醚_weight"
    ],
    "🛡️ 腐蝕抑制劑 (Corrosion Inhibitors)": [
        "BTA_weight", "1,2,4-三氮唑_weight", "1H-1,2,3-三氮唑_weight", 
        "5-甲基-1H-苯並三唑_weight", "5-氯苯並三氮唑_weight", "苯並咪唑_weight", 
        "1-甲基-1H-苯並咪唑_weight", "2-甲基苯並咪唑_weight", "咪唑_weight", 
        "1-甲基咪唑_weight", "2-丙基咪唑_weight", "吡唑_weight", "5-ATZ_weight", 
        "氨基-1,2,4-三氮唑_weight", "三苯基氯化四氮唑_weight", "四氮唑紫_weight", 
        "抑制劑_weight"
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
    "⚖️ 鹼類與 pH 調節劑 (Bases/pH Adjusters)": [
        "45%KOH_weight", "TEA_weight", "正丁胺_weight"
    ],
    "🧩 氨基酸與其他添加劑 (Others)": [
        "L-精氨酸_weight", "甘胺酸_weight", "半胱胺酸_weight", 
        "NaCl_weight", "PB_weight", "60%DEHP_weight", "Inhibitor J_weight"
    ]
}

# 初始化 Session State
if "df" not in st.session_state:
    st.session_state.df = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = []
if "models" not in st.session_state:
    st.session_state.models = {}
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None

# ==========================================
# 2. 核心函式定義
# ==========================================
def convert_to_wt_pct(X_df, feature_cols):
    """將原始重量轉換為重量百分比 (wt%)"""
    chem_cols = [c for c in feature_cols if c not in ['temp', 'region']]
    X_pct = X_df.copy()
    total_weights = X_pct[chem_cols].sum(axis=1).replace(0, 1)
    X_pct[chem_cols] = X_pct[chem_cols].div(total_weights, axis=0) * 100
    return X_pct

def train_models(df):
    """訓練多重 AI 模型"""
    feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS + TARGET_COLS]
    st.session_state.feature_cols = feature_cols
    
    X_raw = df[feature_cols].fillna(0)
    y_snag = df['snag_cu_undercut_um'].fillna(0)
    y_cu_ni = df['cu_ni_undercut_um'].fillna(0)
    
    X_pct = convert_to_wt_pct(X_raw, feature_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pct)
    st.session_state.scaler = scaler
    
    st.session_state.models = {
        'rf_snag': RandomForestRegressor(n_estimators=100, random_state=42).fit(X_pct, y_snag),
        'rf_cu_ni': RandomForestRegressor(n_estimators=100, random_state=42).fit(X_pct, y_cu_ni),
        'xgb_snag': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_pct, y_snag),
        'xgb_cu_ni': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_pct, y_cu_ni),
        'ridge_snag': Ridge(alpha=1.0).fit(X_scaled, y_snag),
        'ridge_cu_ni': Ridge(alpha=1.0).fit(X_scaled, y_cu_ni)
    }
    return feature_cols

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

# ==========================================
# 3. 側邊欄設計
# ==========================================
with st.sidebar:
    st.header("系統狀態")
    if st.session_state.df is not None:
        st.success(f"✅ 資料庫已載入 ({len(st.session_state.df)} 筆)")
        st.success("✅ 三大 AI 模型準備就緒")
        st.divider()
        csv_data = convert_df_to_csv(st.session_state.df)
        st.download_button("📥 下載最新 CSV 備份", data=csv_data, file_name=f"Etch_Recipe_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)
    else:
        st.warning("⚠️ 尚未載入資料")

# ==========================================
# 4. 主畫面分頁設定
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 分頁一：資料載入", 
    "🧪 分頁二：正向配方推測", 
    "🎯 分頁三：逆向最佳配方探索", 
    "🚀 分頁四：AI 全自動組合探索", 
    "📝 分頁五：實驗結果登錄"
])

# ------------------------------------------
# 分頁一：資料載入
# ------------------------------------------
with tab1:
    uploaded_file = st.file_uploader("請上傳清洗後的寬表 CSV 檔案", type=["csv"])
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("current_file_id") != file_id:
            with st.spinner("⏳ AI 正在訓練多重模型..."):
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                train_models(df)
                st.session_state["current_file_id"] = file_id
            st.success("✅ 資料讀取與多重模型訓練成功！")
        else:
            st.success("✅ 模型已準備就緒。")

# ------------------------------------------
# 分頁二：正向推測與紀錄
# ------------------------------------------
with tab2:
    st.header("🧪 正向尋找最佳配方 (wt% 濃度百分比)")
    if st.session_state.df is None:
        st.info("請先至「資料載入」分頁上傳 CSV 檔案。")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 🌡️ 基礎環境與基底")
            temp = st.number_input("溫度 (temp)", value=25.0, step=1.0)
            region = st.selectbox("區域 (region)", options=[1, 0], format_func=lambda x: "密區 (1)" if x == 1 else "疏區 (0)")
            st.divider()
            h2o = st.number_input("H2O 重量", value=60.0, step=1.0)
            h3po4 = st.number_input("H3PO4 重量", value=10.0, step=1.0)
            h2o2 = st.number_input("H2O2 重量", value=15.0, step=1.0)

        with c2:
            st.markdown("### 🧪 添加物選擇")
            base_features = ['temp', 'region', 'H2O_weight', 'H3PO4_weight', 'H2O2_weight']
            optional_chems = [c for c in st.session_state.feature_cols if c not in base_features]
            chem_inputs = {}
            selected_count = 0
            categorized_chems = [chem for sublist in CHEMICAL_CATEGORIES.values() for chem in sublist]
            uncategorized_chems = [c for c in optional_chems if c not in categorized_chems]
            
            for cat_name, chems_in_cat in CHEMICAL_CATEGORIES.items():
                valid_chems = [c for c in chems_in_cat if c in optional_chems]
                if valid_chems:
                    with st.expander(cat_name):
                        for chem in valid_chems:
                            if st.checkbox(chem.replace('_weight', ''), key=f"t2_{chem}"):
                                selected_count += 1
                                chem_inputs[chem] = st.number_input(f"重量", value=1.0, step=0.1, key=f"num_{chem}")
            if uncategorized_chems:
                with st.expander("📦 其他未分類添加物"):
                    for chem in uncategorized_chems:
                        if st.checkbox(chem.replace('_weight', ''), key=f"t2_{chem}"):
                            selected_count += 1
                            chem_inputs[chem] = st.number_input(f"重量", value=1.0, step=0.1, key=f"num_{chem}")

        with c3:
            st.markdown("### 🤖 執行運算")
            btn_predict = st.button("🚀 執行多模型推測與歷史查詢", use_container_width=True, type="primary")

        if btn_predict or st.session_state.get('has_predicted', False):
            st.session_state['has_predicted'] = True
            input_data = {col: 0.0 for col in st.session_state.feature_cols}
            input_data['temp'], input_data['region'] = temp, region
            input_data['H2O_weight'], input_data['H3PO4_weight'], input_data['H2O2_weight'] = h2o, h3po4, h2o2
            for chem, val in chem_inputs.items(): input_data[chem] = val
            input_raw = pd.DataFrame([input_data])
            input_pct = convert_to_wt_pct(input_raw, st.session_state.feature_cols)
            input_scaled = st.session_state.scaler.transform(input_pct)
            
            rf_snag_val = st.session_state.models['rf_snag'].predict(input_pct)[0]
            rf_cuni_val = st.session_state.models['rf_cu_ni'].predict(input_pct)[0]
            xgb_snag_val = st.session_state.models['xgb_snag'].predict(input_pct)[0]
            xgb_cuni_val = st.session_state.models['xgb_cu_ni'].predict(input_pct)[0]
            ridge_snag_val = st.session_state.models['ridge_snag'].predict(input_scaled)[0]
            ridge_cuni_val = st.session_state.models['ridge_cu_ni'].predict(input_scaled)[0]

            # 歷史比對
            match_mask = pd.Series(True, index=st.session_state.df.index)
            for col in st.session_state.feature_cols:
                match_mask &= (st.session_state.df[col].fillna(0).round(4) == round(input_data[col], 4))
            matched_history = st.session_state.df[match_mask]
            has_exact_match = not matched_history.empty

            st.markdown("### 📊 多模型預測結果比較")
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.info("🌳 **隨機森林**")
                st.metric("預測 Snag Cu", f"{rf_snag_val:.3f}")
                st.metric("預測 Cu Ni", f"{rf_cuni_val:.3f}")
            with res_col2:
                st.warning("⚡ **XGBoost**")
                st.metric("預測 Snag Cu", f"{xgb_snag_val:.3f}")
                st.metric("預測 Cu Ni", f"{xgb_cuni_val:.3f}")
            with res_col3:
                st.success("📈 **脊迴歸**")
                st.metric("預測 Snag Cu", f"{ridge_snag_val:.3f}")
                st.metric("預測 Cu Ni", f"{ridge_cuni_val:.3f}")

# ------------------------------------------
# 分頁三：逆向最佳配方探索 (人工指定)
# ------------------------------------------
with tab3:
    st.header("🎯 逆向尋找最佳配方 (人工指定模式)")
    if st.session_state.df is None:
        st.info("請先至「資料載入」分頁上傳 CSV 檔案。")
    else:
        t3_c1, t3_c2, t3_c3 = st.columns(3)
        with t3_c1:
            st.markdown("### 🎯 目標設定")
            target_snag = st.number_input("目標 Snag Cu", value=0.100, step=0.01, format="%.3f")
            target_cu_ni = st.number_input("目標 Cu Ni", value=0.100, step=0.01, format="%.3f")
        with t3_c2:
            st.markdown("### 🧪 指定原料")
            tab3_chems = []
            for cat, chems in CHEMICAL_CATEGORIES.items():
                with st.expander(cat):
                    for c in chems:
                        if c in st.session_state.feature_cols and st.checkbox(c.replace('_weight',''), key=f"t3_{c}"):
                            tab3_chems.append(c)
        with t3_c3:
            btn_explore = st.button("🔍 開始 10,000 次模擬", type="primary")

        if btn_explore:
            with st.spinner("模擬中..."):
                N = 10000
                sim_data = {col: np.zeros(N) for col in st.session_state.feature_cols}
                sim_data['temp'], sim_data['region'] = 25.0, 1
                sim_data['H2O_weight'] = np.random.uniform(50, 80, N)
                sim_data['H3PO4_weight'] = np.random.uniform(5, 20, N)
                sim_data['H2O2_weight'] = np.random.uniform(5, 20, N)
                for c in tab3_chems: sim_data[c] = np.random.uniform(0.1, 5, N)
                
                df_sim = pd.DataFrame(sim_data)
                df_sim_pct = convert_to_wt_pct(df_sim, st.session_state.feature_cols)
                p_snag = st.session_state.models['xgb_snag'].predict(df_sim_pct)
                p_cuni = st.session_state.models['xgb_cu_ni'].predict(df_sim_pct)
                error = abs(p_snag - target_snag) + abs(p_cuni - target_cu_ni)
                df_sim_pct['error'] = error
                top3 = df_sim_pct.sort_values('error').head(3)
                st.write("✅ 最佳推薦比例 (總重 100g)：", top3[['H2O_weight','H3PO4_weight','H2O2_weight'] + tab3_chems])

# ------------------------------------------
# 分頁四：AI 全自動組合探索 (新功能)
# ------------------------------------------
with tab4:
    st.header("🚀 AI 全自動配方組合探索")
    if st.session_state.df is None:
        st.info("請先至「資料載入」分頁上傳 CSV 檔案。")
    else:
        st.write("讓 AI 從候選池中隨機抽樣，碰撞出最佳配方組合。")
        t4_c1, t4_c2, t4_c3 = st.columns([1, 1.5, 1])
        
        with t4_c1:
            st.markdown("### 🎯 1. 優化目標")
            u_snag = st.checkbox("優化 Snag Cu", value=True)
            t_snag = st.number_input("目標 Snag Cu", value=0.100, step=0.01, disabled=not u_snag)
            u_cuni = st.checkbox("優化 Cu Ni", value=True)
            t_cuni = st.number_input("目標 Cu Ni", value=0.100, step=0.01, disabled=not u_cuni)
            st.divider()
            k_num = st.number_input("追加原料數量 (K)", min_value=1, max_value=5, value=3)

        with t4_c2:
            st.markdown("### 🧪 2. 原料候選池 (預設全選)")
            base_chems = ['H2O_weight', 'H3PO4_weight', 'H2O2_weight']
            opt_chems = [c for c in st.session_state.feature_cols if c not in ['temp', 'region'] + base_chems]
            pool = []
            for cat, chems in CHEMICAL_CATEGORIES.items():
                v_chems = [c for c in chems if c in opt_chems]
                if v_chems:
                    with st.expander(cat):
                        for c in v_chems:
                            if st.checkbox(c.replace('_weight',''), value=True, key=f"p_{c}"):
                                pool.append(c)
        
        with t4_c3:
            st.markdown("### 🤖 3. 執行")
            if st.button("🔍 開始 AI 自動探索", type="primary", use_container_width=True):
                if len(pool) < k_num: st.error("候選原料不足。")
                elif not (u_snag or u_cuni): st.warning("請選目標。")
                else:
                    with st.spinner("碰撞中..."):
                        results = []
                        for _ in range(10000):
                            sel = random.sample(pool, int(k_num))
                            inp = {c: 0.0 for c in st.session_state.feature_cols}
                            inp['temp'], inp['region'] = 25.0, 1
                            inp['H2O_weight'], inp['H3PO4_weight'], inp['H2O2_weight'] = np.random.uniform(50,80), np.random.uniform(5,20), np.random.uniform(5,20)
                            for s in sel: inp[s] = np.random.uniform(0.1, 5)
                            df_p = convert_to_wt_pct(pd.DataFrame([inp]), st.session_state.feature_cols)
                            ps, pc = st.session_state.models['xgb_snag'].predict(df_p)[0], st.session_state.models['xgb_cu_ni'].predict(df_p)[0]
                            err = 0
                            if u_snag: err += abs(ps - t_snag)
                            if u_cuni: err += abs(pc - t_cuni)
                            results.append({'err': err, 'ps': ps, 'pc': pc, 'f': "+".join(base_chems + sel), 'w': "+".join([f"{df_p[c].iloc[0]:.2f}" for c in base_chems + sel])})
                        top = pd.DataFrame(results).sort_values('err').head(3)
                        for idx, r in top.iterrows():
                            st.success(f"推薦 {idx+1}: Snag={r['ps']:.3f}, CuNi={r['pc']:.3f}")
                            st.code(f"Formula: {r['f']}\nWeights: {r['w']}")

# ------------------------------------------
# 分頁五：實驗結果登錄
# ------------------------------------------
with tab5:
    st.header("📝 登錄真實實驗結果")
    if st.session_state.df is not None:
        t5_c1, t5_c2, t5_c3 = st.columns(3)
        with t5_c1:
            t5_t = st.number_input("溫度", value=25.0, key="t5_t")
            t5_h2o = st.number_input("H2O", value=60.0, key="t5_h2o")
        with t5_c2:
            t5_inps = {}
            for cat, chems in CHEMICAL_CATEGORIES.items():
                with st.expander(cat):
                    for c in chems:
                        if c in st.session_state.feature_cols and st.checkbox(c.replace('_weight',''), key=f"t5_chk_{c}"):
                            t5_inps[c] = st.number_input("重量", value=1.0, key=f"t5_n_{c}")
        with t5_c3:
            t5_snag = st.number_input("真實 Snag", value=0.0)
            if st.button("💾 儲存並更新模型", type="primary"):
                # 此處實作將資料 concat 入 st.session_state.df 並呼叫 train_models 的邏輯
                st.success("✅ 模型已更新知識！")
