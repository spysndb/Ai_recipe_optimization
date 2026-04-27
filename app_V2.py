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
        "L-精丸酸_weight", "甘胺酸_weight", "半胱胺酸_weight", 
        "NaCl_weight", "PB_weight", "60%DEHP_weight", "Inhibitor J_weight"
    ]
}

# 初始化 Session State
if "df" not in st.session_state: st.session_state.df = None
if "feature_cols" not in st.session_state: st.session_state.feature_cols = []
if "models" not in st.session_state: st.session_state.models = {}
if "scaler" not in st.session_state: st.session_state.scaler = None
if "current_file_id" not in st.session_state: st.session_state.current_file_id = None

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
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_pct)
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

def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8-sig')

# ==========================================
# 3. 側邊欄設計
# ==========================================
with st.sidebar:
    st.header("系統狀態")
    if st.session_state.df is not None:
        st.success(f"✅ 資料庫已載入 ({len(st.session_state.df)} 筆)")
        csv_data = convert_df_to_csv(st.session_state.df)
        st.download_button("📥 下載最新 CSV 備份", data=csv_data, file_name=f"Etch_Recipe_{datetime.now().strftime('%Y%m%d')}.csv", use_container_width=True)
    else: st.warning("⚠️ 尚未載入資料")

# ==========================================
# 4. 主畫面分頁設定
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📂 分頁一：資料載入", "🧪 分頁二：正向配方推測", "🎯 分頁三：逆向最佳配方探索", "🚀 分頁四：AI 全自動組合探索", "📝 分頁五：實驗結果登錄"])

# ------------------------------------------
# 分頁一：資料載入
# ------------------------------------------
with tab1:
    uploaded_file = st.file_uploader("請上傳清洗後的寬表 CSV 檔案", type=["csv"])
    if uploaded_file:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("current_file_id") != file_id:
            with st.spinner("⏳ AI 正在建立模型..."):
                df = pd.read_csv(uploaded_file); st.session_state.df = df
                train_models(df); st.session_state["current_file_id"] = file_id
            st.success("✅ 資料讀取與模型訓練成功！")

# ------------------------------------------
# 分頁二：正向推測與紀錄
# ------------------------------------------
with tab2:
    st.header("🧪 正向尋找最佳配方 (wt% 濃度百分比)")
    if st.session_state.df is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 🌡️ 基礎環境與基底")
            temp = st.number_input("溫度 (temp)", value=25.0, step=1.0)
            region = st.selectbox("區域 (region)", options=[1, 0], format_func=lambda x: "密區 (1)" if x == 1 else "疏區 (0)")
            h2o = st.number_input("H2O 重量", value=60.0, step=1.0)
            h3po4 = st.number_input("H3PO4 重量", value=10.0, step=1.0)
            h2o2 = st.number_input("H2O2 重量", value=15.0, step=1.0)
        with c2:
            st.markdown("### 🧪 添加物選擇")
            base_features = ['temp', 'region', 'H2O_weight', 'H3PO4_weight', 'H2O2_weight']
            optional_chems = [c for c in st.session_state.feature_cols if c not in base_features]
            chem_inputs = {}; selected_count = 0
            for cat_name, chems_in_cat in CHEMICAL_CATEGORIES.items():
                valid_chems = [c for c in chems_in_cat if c in optional_chems]
                if valid_chems:
                    with st.expander(cat_name):
                        for chem in valid_chems:
                            if st.checkbox(chem.replace('_weight', ''), key=f"t2_{chem}"):
                                selected_count += 1
                                chem_inputs[chem] = st.number_input(f"重量", value=1.0, step=0.1, key=f"num_{chem}")
        with c3:
            st.markdown("### 🤖 執行運算")
            btn_predict = st.button("🚀 執行多模型推測與歷史查詢", use_container_width=True, type="primary")

        if btn_predict: st.session_state['has_predicted'] = True
        if st.session_state.get('has_predicted', False) and selected_count <= 10:
            input_data = {col: 0.0 for col in st.session_state.feature_cols}
            input_data.update({'temp':temp, 'region':region, 'H2O_weight':h2o, 'H3PO4_weight':h3po4, 'H2O2_weight':h2o2})
            for chem, val in chem_inputs.items(): input_data[chem] = val
            input_raw = pd.DataFrame([input_data])
            input_pct = convert_to_wt_pct(input_raw, st.session_state.feature_cols)
            input_scaled = st.session_state.scaler.transform(input_pct)
            
            # 預測
            rf_snag_val = st.session_state.models['rf_snag'].predict(input_pct)[0]
            rf_cuni_val = st.session_state.models['rf_cu_ni'].predict(input_pct)[0]
            xgb_snag_val = st.session_state.models['xgb_snag'].predict(input_pct)[0]
            xgb_cuni_val = st.session_state.models['xgb_cu_ni'].predict(input_pct)[0]
            ridge_snag_val = st.session_state.models['ridge_snag'].predict(input_scaled)[0]
            ridge_cuni_val = st.session_state.models['ridge_cu_ni'].predict(input_scaled)[0]

            # 歷史比對邏輯
            match_mask = pd.Series(True, index=st.session_state.df.index)
            for col in st.session_state.feature_cols:
                match_mask &= (st.session_state.df[col].fillna(0) == input_data[col])
            matched_history = st.session_state.df[match_mask]
            has_exact_match = not matched_history.empty

            st.markdown("### 📊 多模型預測結果比較")
            if has_exact_match:
                real_snag = matched_history['snag_cu_undercut_um'].iloc[0]
                real_cuni = matched_history['cu_ni_undercut_um'].iloc[0]
                def get_gap(p, r): return ((p-r)/r)*100 if r!=0 else 0
                st.write(f"💡 **偵測到相同歷史配方**：真實 Snag Cu: `{real_snag:.3f}`, 真實 Cu Ni: `{real_cuni:.3f}`")
            
            r1, r2, r3 = st.columns(3)
            with r1:
                st.info("🌳 **隨機森林**")
                st.metric("預測 Snag Cu", f"{rf_snag_val:.3f}", delta=f"{get_gap(rf_snag_val, real_snag):+.1f}%" if has_exact_match else None, delta_color="inverse")
                st.metric("預測 Cu Ni", f"{rf_cuni_val:.3f}", delta=f"{get_gap(rf_cuni_val, real_cuni):+.1f}%" if has_exact_match else None, delta_color="inverse")
            with r2:
                st.warning("⚡ **XGBoost**")
                st.metric("預測 Snag Cu", f"{xgb_snag_val:.3f}", delta=f"{get_gap(xgb_snag_val, real_snag):+.1f}%" if has_exact_match else None, delta_color="inverse")
                st.metric("預測 Cu Ni", f"{xgb_cuni_val:.3f}", delta=f"{get_gap(xgb_cuni_val, real_cuni):+.1f}%" if has_exact_match else None, delta_color="inverse")
            with r3:
                st.success("📈 **脊迴歸**")
                st.metric("預測 Snag Cu", f"{ridge_snag_val:.3f}", delta=f"{get_gap(ridge_snag_val, real_snag):+.1f}%" if has_exact_match else None, delta_color="inverse")
                st.metric("預測 Cu Ni", f"{ridge_cuni_val:.3f}", delta=f"{get_gap(ridge_cuni_val, real_cuni):+.1f}%" if has_exact_match else None, delta_color="inverse")

            # 影響力 Plotly 圖表
            st.divider(); st.subheader("🎯 當前配方成分影響力分析")
            active_ingredients = ['H2O_weight', 'H3PO4_weight', 'H2O2_weight'] + list(chem_inputs.keys())
            model_xgb = st.session_state.models['xgb_snag']
            dmatrix = xgb.DMatrix(input_pct)
            local_contribs = model_xgb.get_booster().predict(dmatrix, pred_contribs=True)[0][:-1]
            all_feats = st.session_state.feature_cols
            importance_data = [{'成分': f.replace('_weight',''), '原始重要程度': abs(imp)} for f, imp in zip(all_feats, local_contribs) if f in active_ingredients]
            df_imp = pd.DataFrame(importance_data)
            
            selected_display = []
            cols = st.columns(6)
            for i, chem in enumerate(df_imp['成分'].tolist()):
                with cols[i%6]:
                    if st.checkbox(chem, value=True, key=f"imp_{chem}"): selected_display.append(chem)
            if selected_display:
                plot_df = df_imp[df_imp['成分'].isin(selected_display)].copy()
                total_imp = plot_df['原始重要程度'].sum()
                plot_df['相對百分比'] = (plot_df['原始重要程度'] / total_imp * 100) if total_imp > 0 else 0
                fig_imp = px.bar(plot_df.sort_values('相對百分比'), x='相對百分比', y='成分', orientation='h', text=plot_df['相對百分比'].apply(lambda x: f'{x:.1f}%'))
                st.plotly_chart(fig_imp, use_container_width=True)

            # 信心評估面板
            st.divider(); st.subheader("🛡️ AI 推測信心評估報告")
            X_all_pct = convert_to_wt_pct(st.session_state.df[st.session_state.feature_cols].fillna(0), st.session_state.feature_cols)
            min_dist = np.sqrt(((X_all_pct - input_pct.iloc[0])**2).sum(axis=1)).min()
            proximity = max(0.0, min(100.0, 100.0 - (min_dist * 15)))
            cv = np.std([rf_snag_val, xgb_snag_val, ridge_snag_val]) / np.mean([rf_snag_val, xgb_snag_val, ridge_snag_val]) if np.mean([rf_snag_val, xgb_snag_val, ridge_snag_val]) != 0 else 0
            consensus = max(0.0, min(100.0, 100.0 - (cv * 200)))
            total_conf = (proximity * 0.6) + (consensus * 0.4)
            conf_c1, conf_c2, conf_c3 = st.columns(3)
            with conf_c1: st.write("**數據接近度**"); st.write(f"{proximity:.1f}%")
            with conf_c2: st.write("**模型共識度**"); st.write(f"{consensus:.1f}%")
            with conf_c3: st.write("**綜合信賴指數**"); st.progress(total_conf/100); st.write(f"{total_conf:.1f}% 可信" if total_conf > 50 else f"{total_conf:.1f}% 風險高")

# ------------------------------------------
# 分頁三：逆向最佳配方探索 (人工指定)
# ------------------------------------------
with tab3:
    st.header("🎯 逆向尋找最佳配方 (人工模式)")
    if st.session_state.df is not None:
        st.write("設定目標，AI 會自動換算為總重 100g 比例。")
        t3_c1, t3_c2, t3_c3 = st.columns(3)
        with t3_c1:
            st.markdown("### 🎯 目標")
            target_snag = st.number_input("目標 Snag Cu", value=0.100, step=0.01, format="%.3f", key="t3_tsnag")
            target_cu_ni = st.number_input("目標 Cu Ni", value=0.100, step=0.01, format="%.3f", key="t3_tcuni")
            tab3_bases = []
            if st.checkbox("H2O", value=True, key="t3_h2o"): tab3_bases.append('H2O_weight')
            if st.checkbox("H3PO4", value=True, key="t3_h3p"): tab3_bases.append('H3PO4_weight')
            if st.checkbox("H2O2", value=True, key="t3_h2o2"): tab3_bases.append('H2O2_weight')
        with t3_c2:
            st.markdown("### 🧪 指定原料")
            tab3_chems = []
            for cat, chems in CHEMICAL_CATEGORIES.items():
                with st.expander(cat):
                    for c in chems:
                        if c in st.session_state.feature_cols and st.checkbox(c.replace('_weight',''), key=f"t3_{c}"):
                            tab3_chems.append(c)
        with t3_c3:
            btn_explore = st.button("🔍 開始 10,000 次模擬", type="primary", use_container_width=True, key="t3_btn")
            if btn_explore:
                with st.spinner("模擬中..."):
                    N = 10000; sim = {col: np.zeros(N) for col in st.session_state.feature_cols}
                    for k in ['temp', 'region']: sim[k] = 25.0 if k=='temp' else 1
                    for b in tab3_bases: sim[b] = np.random.uniform(40, 80, N)
                    for c in tab3_chems: sim[c] = np.random.uniform(0.1, 8, N)
                    df_sim_pct = convert_to_wt_pct(pd.DataFrame(sim), st.session_state.feature_cols)
                    p_s = st.session_state.models['xgb_snag'].predict(df_sim_pct)
                    p_c = st.session_state.models['xgb_cu_ni'].predict(df_sim_pct)
                    df_sim_pct['error'] = abs(p_s-target_snag) + abs(p_c-target_cu_ni)
                    top3 = df_sim_pct.sort_values('error').head(3)
                    for i, r in top3.reset_index(drop=True).iterrows():
                        st.success(f"推薦 {i+1}"); st.write(f"Snag: {p_s[idx]:.3f}, CuNi: {p_c[idx]:.3f}")
                        st.code(f"Weights: {r[tab3_bases+tab3_chems].to_dict()}")

# ------------------------------------------
# 分頁四：AI 全自動組合探索 (全新功能)
# ------------------------------------------
with tab4:
    st.header("🚀 AI 全自動配方組合探索")
    if st.session_state.df is None:
        st.info("請先至「資料載入」分頁上傳 CSV 檔案。")
    else:
        st.write("讓 AI 從候選池中隨機挑選 $K$ 種原料，自動尋找最佳化學配比。")
        t4_c1, t4_c2, t4_c3 = st.columns([1, 1.5, 1])
        
        with t4_c1:
            st.markdown("### 🎯 1. 優化目標設定")
            use_s = st.checkbox("優化 Snag Cu", value=True, key="t4_u_s")
            tar_s = st.number_input("目標 Snag Cu (um)", value=0.100, step=0.01, disabled=not use_s)
            use_c = st.checkbox("優化 Cu Ni", value=True, key="t4_u_c")
            tar_c = st.number_input("目標 Cu Ni (um)", value=0.100, step=0.01, disabled=not use_c)
            st.divider()
            k_num = st.number_input("除了基底，AI 需額外追加幾種原料？", min_value=1, max_value=5, value=3)
            st.caption(f"AI 每回合將隨機抽出正好 {int(k_num)} 種原料。")

        with t4_c2:
            st.markdown("### 🧪 2. 原料候選池 (預設全選)")
            st.caption("取消勾選即代表不讓 AI 使用該原料。")
            base_list = ['H2O_weight', 'H3PO4_weight', 'H2O2_weight']
            pool_list = [c for c in st.session_state.feature_cols if c not in ['temp', 'region'] + base_list]
            final_pool = []
            for cat, chems in CHEMICAL_CATEGORIES.items():
                v_chems = [c for c in chems if c in pool_list]
                if v_chems:
                    with st.expander(cat, expanded=False):
                        for c in v_chems:
                            if st.checkbox(c.replace('_weight',''), value=True, key=f"pool_{c}"):
                                final_pool.append(c)
        
        with t4_c3:
            st.markdown("### 🤖 3. 執行探索")
            if st.button("🚀 開始 10,000 次 AI 組合模擬", type="primary", use_container_width=True):
                if len(final_pool) < k_num: st.error("候選原料不足。")
                elif not (use_s or use_c): st.warning("請至少選一個目標。")
                else:
                    with st.spinner("AI 正在排列組合與演算中..."):
                        results = []
                        for _ in range(10000):
                            sel = random.sample(final_pool, int(k_num))
                            row = {c: 0.0 for c in st.session_state.feature_cols}
                            row.update({'temp':25.0, 'region':1, 'H2O_weight':np.random.uniform(50,80), 'H3PO4_weight':np.random.uniform(5,20), 'H2O2_weight':np.random.uniform(5,20)})
                            for s in sel: row[s] = np.random.uniform(0.1, 5)
                            df_p = convert_to_wt_pct(pd.DataFrame([row]), st.session_state.feature_cols)
                            ps, pc = st.session_state.models['xgb_snag'].predict(df_p)[0], st.session_state.models['xgb_cu_ni'].predict(df_p)[0]
                            err = (abs(ps-tar_s) if use_s else 0) + (abs(pc-tar_c) if use_c else 0)
                            results.append({'error': err, 'snag': ps, 'cuni': pc, 'formula': "+".join(["H2O","H3PO4","H2O2"]+[x.replace('_weight','') for x in sel]), 'weights': "+".join([f"{df_p[x].iloc[0]:.2f}" for x in (['H2O_weight','H3PO4_weight','H2O2_weight']+sel)])})
                        top = pd.DataFrame(results).sort_values('error').head(3)
                        for i, r in top.reset_index(drop=True).iterrows():
                            st.success(f"🏆 推薦組合 {i+1}")
                            st.write(f"預測 Snag: {r['snag']:.3f} / CuNi: {r['cuni']:.3f}")
                            st.code(f"chemical_formula: {r['formula']}\nchemical_weights: {r['weights']}")

# ------------------------------------------
# 分頁五：實驗結果登錄
# ------------------------------------------
with tab5:
    st.header("📝 登錄真實實驗結果")
    if st.session_state.df is not None:
        t5_c1, t5_c2, t5_c3 = st.columns(3)
        with t5_c1:
            st.markdown("### 🌡️ 1. 環境與基底")
            t5_temp = st.number_input("溫度", value=25.0, key="t5_temp")
            t5_reg = st.selectbox("區域", [1,0], format_func=lambda x:"密區(1)" if x==1 else "疏區(0)", key="t5_reg")
            t5_h2o = st.number_input("H2O", value=60.0, key="t5_h2o")
            t5_h3p = st.number_input("H3PO4", value=10.0, key="t5_h3p")
            t5_h2o2 = st.number_input("H2O2", value=15.0, key="t5_h2o2")
        with t5_c2:
            st.markdown("### 🧪 2. 實際添加物")
            t5_inps = {}; t5_cnt = 0
            for cat, chems in CHEMICAL_CATEGORIES.items():
                v_chems = [c for c in chems if c in st.session_state.feature_cols]
                if v_chems:
                    with st.expander(cat):
                        for c in v_chems:
                            if st.checkbox(c.replace('_weight',''), key=f"t5_chk_{c}"):
                                t5_cnt += 1; t5_inps[c] = st.number_input("重量", value=1.0, key=f"t5_val_{c}")
        with t5_c3:
            st.markdown("### 🔬 3. 實驗結果")
            t5_s = st.number_input("真實 Snag Cu", value=0.0, key="t5_s")
            t5_c = st.number_input("真實 Cu Ni", value=0.0, key="t5_c")
            t5_t = st.number_input("蝕刻時間(sec)", value=0, key="t5_t_sec")
            t5_res = st.text_area("備註", height=100, key="t5_res_txt")
            if st.button("💾 寫入並重訓模型", type="primary", use_container_width=True):
                new_row = {c: 0.0 for c in st.session_state.df.columns}
                new_row.update({'temp':t5_temp, 'region':t5_reg, 'H2O_weight':t5_h2o, 'H3PO4_weight':t5_h3p, 'H2O2_weight':t5_h2o2, 'snag_cu_undercut_um':t5_s, 'cu_ni_undercut_um':t5_c, 'etch_time_value_sec':t5_t, 'result':t5_res, 'date_folder':datetime.now().strftime("%Y%m%d"), 'item':'NEW'})
                for c, v in t5_inps.items(): new_row[c] = v
                st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
                train_models(st.session_state.df); st.success("✅ 模型已更新！")
