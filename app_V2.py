import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from datetime import datetime
import io

# 【新增】使用 Plotly 取代 Matplotlib，完美解決中文亂碼與大小問題
import plotly.express as px



# ==========================================
# 【修正點 1】：補上畫圖必須的套件
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns

# 解決圖表中文顯示變亂碼的問題 (涵蓋 Windows 與 Mac)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'PingFang HK', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 頁面基本設定與分類字典
# ==========================================
st.set_page_config(page_title="蝕刻配方推測系統", page_icon="🧪", layout="wide")
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
    """
    【2-b 核心升級】：將原始重量轉換為重量百分比 (wt%)
    排除 temp 與 region，僅將化學品重量相加作為分母。
    """
    chem_cols = [c for c in feature_cols if c not in ['temp', 'region']]
    X_pct = X_df.copy()
    
    # 計算每一列的化學品總重 (取代 0 為 1 避免除以零錯誤)
    total_weights = X_pct[chem_cols].sum(axis=1).replace(0, 1)
    
    # 將每一種化學品重量除以總重，再乘以 100 得到百分比
    X_pct[chem_cols] = X_pct[chem_cols].div(total_weights, axis=0) * 100
    return X_pct

def train_models(df):
    feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS + TARGET_COLS]
    st.session_state.feature_cols = feature_cols
    
    X_raw = df[feature_cols].fillna(0)
    y_snag = df['snag_cu_undercut_um'].fillna(0)
    y_cu_ni = df['cu_ni_undercut_um'].fillna(0)
    
    # 轉換為重量百分比供模型學習
    X_pct = convert_to_wt_pct(X_raw, feature_cols)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pct)
    st.session_state.scaler = scaler
    
    rf_snag = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_pct, y_snag)
    rf_cu_ni = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_pct, y_cu_ni)
    
    xgb_snag = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_pct, y_snag)
    xgb_cu_ni = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_pct, y_cu_ni)
    
    ridge_snag = Ridge(alpha=1.0).fit(X_scaled, y_snag)
    ridge_cu_ni = Ridge(alpha=1.0).fit(X_scaled, y_cu_ni)
    
    st.session_state.models = {
        'rf_snag': rf_snag, 'rf_cu_ni': rf_cu_ni,
        'xgb_snag': xgb_snag, 'xgb_cu_ni': xgb_cu_ni,
        'ridge_snag': ridge_snag, 'ridge_cu_ni': ridge_cu_ni
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
        st.success("✅ 三大 AI 模型 (wt%版) 準備就緒")
        st.divider()
        csv_data = convert_df_to_csv(st.session_state.df)
        st.download_button("📥 下載最新 CSV 備份", data=csv_data, file_name=f"Etch_Recipe_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)
    else:
        st.warning("⚠️ 尚未載入資料")

# ==========================================
# 4. 主畫面分頁設定
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["📂 分頁一：資料載入", "🧪 分頁二：正向配方推測", "🎯 分頁三：逆向最佳配方探索", "📝 分頁四：實驗結果登錄"])

# ------------------------------------------
# 分頁一：資料載入
# ------------------------------------------
with tab1:
    uploaded_file = st.file_uploader("請上傳清洗後的寬表 CSV 檔案", type=["csv"])
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("current_file_id") != file_id:
            with st.spinner("⏳ AI 正在將原始數據轉換為百分比並建立模型，請稍候..."):
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                train_models(df)
                st.session_state["current_file_id"] = file_id
            st.success("✅ 資料讀取與多重模型訓練成功！您可以前往「正向推測」分頁。")
        else:
            st.success("✅ 資料庫與模型已在記憶體中準備就緒！您可以放心操作。")

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
            st.caption("請勾選欲使用的添加物，並填寫重量 (最多 10 種)")
            
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
            
            if selected_count > 10:
                st.error("⚠️ 選擇了超過 10 種添加物，請減少數量。")

        with c3:
            st.markdown("### 🤖 執行運算")
            btn_predict = st.button("🚀 執行多模型推測與歷史查詢", use_container_width=True, type="primary")

        if btn_predict and selected_count <= 10:
            input_data = {col: 0.0 for col in st.session_state.feature_cols}
            input_data['temp'], input_data['region'] = temp, region
            input_data['H2O_weight'], input_data['H3PO4_weight'], input_data['H2O2_weight'] = h2o, h3po4, h2o2
            for chem, val in chem_inputs.items():
                input_data[chem] = val
                
            input_raw = pd.DataFrame([input_data])
            
            # --- 轉成百分比再預測 ---
            input_pct = convert_to_wt_pct(input_raw, st.session_state.feature_cols)
            input_scaled = st.session_state.scaler.transform(input_pct)
            
            rf_snag_val = st.session_state.models['rf_snag'].predict(input_pct)[0]
            rf_cuni_val = st.session_state.models['rf_cu_ni'].predict(input_pct)[0]
            xgb_snag_val = st.session_state.models['xgb_snag'].predict(input_pct)[0]
            xgb_cuni_val = st.session_state.models['xgb_cu_ni'].predict(input_pct)[0]
            ridge_snag_val = st.session_state.models['ridge_snag'].predict(input_scaled)[0]
            ridge_cuni_val = st.session_state.models['ridge_cu_ni'].predict(input_scaled)[0]
            
            st.markdown("### 📊 多模型預測結果比較")
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.info("🌳 **隨機森林 (保守派)**")
                st.metric("預測 Snag Cu (um)", f"{rf_snag_val:.3f}")
                st.metric("預測 Cu Ni (um)", f"{rf_cuni_val:.3f}")
            with res_col2:
                st.warning("⚡ **XGBoost (敏銳派)**")
                st.metric("預測 Snag Cu (um)", f"{xgb_snag_val:.3f}")
                st.metric("預測 Cu Ni (um)", f"{xgb_cuni_val:.3f}")
            with res_col3:
                st.success("📈 **脊迴歸 (趨勢派)**")
                st.metric("預測 Snag Cu (um)", f"{ridge_snag_val:.3f}")
                st.metric("預測 Cu Ni (um)", f"{ridge_cuni_val:.3f}")
            
            st.divider()
            
           # --- 【改用 Plotly：動態成分影響力 (支援勾選與相對百分比)】 ---
            st.divider()
            st.subheader("🎯 當前配方成分影響力分析")
            st.write("您可以取消勾選佔比過大的基底（如 H2O），圖表會自動重新計算剩餘成分的「相對影響百分比」。")

            active_ingredients = ['H2O_weight', 'H3PO4_weight', 'H2O2_weight'] + list(chem_inputs.keys())
            model_xgb = st.session_state.models['xgb_snag']
            all_importances = model_xgb.feature_importances_
            all_feats = st.session_state.feature_cols
            
            importance_data = []
            for feat, imp in zip(all_feats, all_importances):
                if feat in active_ingredients:
                    importance_data.append({'成分': feat.replace('_weight', ''), '原始重要程度': imp})
            
            df_imp = pd.DataFrame(importance_data)

            # 使用 4 個一排的 Checkbox 讓使用者勾選
            st.markdown("##### 🧪 勾選欲觀察的成分：")
            cols = st.columns(4)
            selected_display = []
            
            for i, chem in enumerate(df_imp['成分'].tolist()):
                with cols[i % 4]:
                    # 預設全部勾選
                    if st.checkbox(chem, value=True, key=f"imp_chk_{chem}"):
                        selected_display.append(chem)

            if selected_display:
                plot_df = df_imp[df_imp['成分'].isin(selected_display)].copy()
                
                # 計算相對百分比 (讓被勾選的項目總和強制為 100%)
                total_selected_imp = plot_df['原始重要程度'].sum()
                if total_selected_imp > 0:
                    plot_df['相對百分比'] = (plot_df['原始重要程度'] / total_selected_imp) * 100
                else:
                    plot_df['相對百分比'] = 0.0
                    
                # 排序讓最長的 bar 在最上面
                plot_df = plot_df.sort_values('相對百分比', ascending=True)
                
                # 建立 Plotly 互動圖表 (顯示 % 符號)
                fig_imp = px.bar(
                    plot_df, 
                    x='相對百分比', 
                    y='成分', 
                    orientation='h', 
                    text=plot_df['相對百分比'].apply(lambda x: f'{x:.1f}%'),
                    color_discrete_sequence=['#4A90E2']
                )
                fig_imp.update_traces(textposition='outside')
                # 為了避免文字被切斷，將 x 軸最大值拉寬一點
                fig_imp.update_layout(
                    height=350, 
                    margin=dict(l=0, r=40, t=30, b=0), 
                    xaxis_title="選定成分間的相對影響力 (%)",
                    yaxis_title="",
                    xaxis=dict(range=[0, plot_df['相對百分比'].max() * 1.15]) 
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.warning("⚠️ 請至少勾選一個成分進行分析。")

            # --- 【改用 Plotly：雙目標預測信心區間與全體分佈圖】 ---
            st.divider()
            st.subheader("🛡️ 預測值信心評估 (全體歷史資料庫大盤點)")
            st.write("此圖表顯示的是**「公司過去所有實驗數據」**的總體分佈。目的是讓您了解，本次 AI 推測的數值 (紅色虛線) 落在過往經驗中的哪個位置 (是常態值還是極端值)。")
            
            # 取得歷史預測結果以計算誤差
            X_all_raw = st.session_state.df[st.session_state.feature_cols].fillna(0)
            X_all_pct = convert_to_wt_pct(X_all_raw, st.session_state.feature_cols)
            
            # Snag Cu 誤差計算
            y_all_real_snag = st.session_state.df['snag_cu_undercut_um'].fillna(0)
            y_all_pred_snag = st.session_state.models['xgb_snag'].predict(X_all_pct)
            avg_error_snag = np.mean(np.abs(y_all_real_snag - y_all_pred_snag))
            
            # Cu Ni 誤差計算
            y_all_real_cuni = st.session_state.df['cu_ni_undercut_um'].fillna(0)
            y_all_pred_cuni = st.session_state.models['xgb_cu_ni'].predict(X_all_pct)
            avg_error_cuni = np.mean(np.abs(y_all_real_cuni - y_all_pred_cuni))
            
            # 建立左右兩欄，左邊放 Snag Cu，右邊放 Cu Ni
            conf_c1, conf_c2 = st.columns(2)
            
            with conf_c1:
                st.markdown("#### 🔵 預估 Snag Cu 範圍")
                st.success(f"{xgb_snag_val - avg_error_snag:.3f} ~ {xgb_snag_val + avg_error_snag:.3f} um")
                st.caption(f"(基於全體歷史平均誤差 ±{avg_error_snag:.4f} um)")
                
                # 繪製 Snag Cu 直方分佈圖
                fig_snag = px.histogram(st.session_state.df, x='snag_cu_undercut_um', nbins=30, 
                                        opacity=0.7, color_discrete_sequence=['#8bb1d6'])
                fig_snag.add_vline(x=xgb_snag_val, line_dash="dash", line_color="red", annotation_text="本次推測落點")
                fig_snag.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0), 
                                       xaxis_title="全體歷史 Snag Cu (um)", yaxis_title="發生次數", showlegend=False)
                st.plotly_chart(fig_snag, use_container_width=True)

            with conf_c2:
                st.markdown("#### 🟢 預估 Cu Ni 範圍")
                st.success(f"{xgb_cuni_val - avg_error_cuni:.3f} ~ {xgb_cuni_val + avg_error_cuni:.3f} um")
                st.caption(f"(基於全體歷史平均誤差 ±{avg_error_cuni:.4f} um)")
                
                # 繪製 Cu Ni 直方分佈圖
                fig_cuni = px.histogram(st.session_state.df, x='cu_ni_undercut_um', nbins=30, 
                                        opacity=0.7, color_discrete_sequence=['#95d5b2'])
                fig_cuni.add_vline(x=xgb_cuni_val, line_dash="dash", line_color="red", annotation_text="本次推測落點")
                fig_cuni.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0), 
                                       xaxis_title="全體歷史 Cu Ni (um)", yaxis_title="發生次數", showlegend=False)
                st.plotly_chart(fig_cuni, use_container_width=True)
                
            st.divider()


            # 歷史查詢 (保留原始重量的嚴格比對邏輯)
            st.markdown("#### 📚 歷史相似配方清單 (完全相同 = 🔴紅色字體，多加1種 = 預設顏色)")
            selected_chems = list(chem_inputs.keys())
            unselected_chems = [c for c in optional_chems if c not in selected_chems]
            
            mask_selected = pd.Series(True, index=st.session_state.df.index)
            for c in selected_chems:
                if c in st.session_state.df.columns:
                    mask_selected = mask_selected & (st.session_state.df[c] > 0)
                    
            valid_unselected = [c for c in unselected_chems if c in st.session_state.df.columns]
            extra_additive_count = (st.session_state.df[valid_unselected] > 0).sum(axis=1)
            mask_extra = extra_additive_count <= 1
            
            matched_df = st.session_state.df[mask_selected & mask_extra].copy()
            
            if matched_df.empty:
                st.info("💡 資料庫中目前沒有相近的歷史配方。這是一組全新的嘗試！")
            else:
                st.success(f"🔍 找到 {len(matched_df)} 筆相似的歷史配方！")
                matched_df['extra_count'] = extra_additive_count[mask_selected & mask_extra]
                
                cols_to_show = set(selected_chems)
                for c in valid_unselected:
                    if (matched_df[c] > 0).any():  
                        cols_to_show.add(c)
                        
                display_cols = ['item', 'date_folder', 'chemical_formula', 'region', 'H2O_weight', 'H3PO4_weight', 'H2O2_weight'] + list(cols_to_show) + ['snag_cu_undercut_um', 'cu_ni_undercut_um', 'result']
                display_cols = [c for c in display_cols if c in matched_df.columns]
                df_to_display = matched_df[display_cols]
                
                def highlight_identical(row):
                    is_identical = matched_df.loc[row.name, 'extra_count'] == 0
                    return ['color: #ff4b4b'] * len(row) if is_identical else [''] * len(row)

                styled_df = df_to_display.style.apply(highlight_identical, axis=1)
                st.dataframe(styled_df, use_container_width=True)

        st.divider()

        
# ------------------------------------------
# 分頁三：逆向最佳配方探索
# ------------------------------------------
with tab3:
    st.header("🎯 逆向尋找最佳配方 (自動換算為總重 100g 比例)")
    if st.session_state.df is None:
        st.info("請先至「資料載入」分頁上傳 CSV 檔案。")
    else:
        st.write("設定您的目標與原料，AI 在尋找最佳解答後，將為您直接產出「總重剛好為 100 克」的完美比例配方！")
        
        tab3_selected_bases = []
        tab3_selected_chems = []
        base_chems = ['H2O_weight', 'H3PO4_weight', 'H2O2_weight']
        
        t3_c1, t3_c2, t3_c3 = st.columns(3)
        
        with t3_c1:
            st.markdown("### 🎯 目標")
            target_snag = st.number_input("目標 Snag Cu (um)", value=0.100, step=0.01, format="%.3f")
            target_cu_ni = st.number_input("目標 Cu Ni (um)", value=0.100, step=0.01, format="%.3f")
            st.divider()
            st.markdown("### 💧 基底")
            if st.checkbox("H2O", value=True, key="t3_H2O"): tab3_selected_bases.append('H2O_weight')
            if st.checkbox("H3PO4", value=True, key="t3_H3PO4"): tab3_selected_bases.append('H3PO4_weight')
            if st.checkbox("H2O2", value=True, key="t3_H2O2"): tab3_selected_bases.append('H2O2_weight')

        with t3_c2:
            st.markdown("### 🧪 勾選配方原料")
            optional_chems = [c for c in st.session_state.feature_cols if c not in ['temp', 'region'] + base_chems]
            categorized_chems = [chem for sublist in CHEMICAL_CATEGORIES.values() for chem in sublist]
            uncategorized_chems = [c for c in optional_chems if c not in categorized_chems]
            
            for cat_name, chems_in_cat in CHEMICAL_CATEGORIES.items():
                valid_chems = [c for c in chems_in_cat if c in optional_chems]
                if valid_chems:
                    with st.expander(cat_name):
                        for chem in valid_chems:
                            if st.checkbox(chem.replace('_weight', ''), key=f"t3_{chem}"):
                                tab3_selected_chems.append(chem)
                    
            if uncategorized_chems:
                with st.expander("📦 其他未分類添加物"):
                    for chem in uncategorized_chems:
                        if st.checkbox(chem.replace('_weight', ''), key=f"t3_{chem}"):
                            tab3_selected_chems.append(chem)

        with t3_c3:
            st.markdown("### 🤖 執行運算")
            btn_explore = st.button("🔍 開始 10,000 次平行宇宙模擬探索", type="primary", use_container_width=True)

        if btn_explore:
            if len(tab3_selected_bases) + len(tab3_selected_chems) == 0:
                st.warning("⚠️ 請至少勾選一種原料！")
            else:
                with st.spinner("AI 正在尋找最佳重量，並自動為您換算成 100g 總重比例..."):
                    N_SIMULATIONS = 10000
                    sim_data_1 = {col: np.zeros(N_SIMULATIONS) for col in st.session_state.feature_cols}
                    sim_data_0 = {col: np.zeros(N_SIMULATIONS) for col in st.session_state.feature_cols}
                    
                    for sim_data in [sim_data_1, sim_data_0]: sim_data['temp'] = np.full(N_SIMULATIONS, 25.0)
                    sim_data_1['region'] = np.full(N_SIMULATIONS, 1)
                    sim_data_0['region'] = np.full(N_SIMULATIONS, 0)
                    
                    # 探索亂數範圍 (此時為隨機重量)
                    h2o_w = np.random.uniform(40.0, 90.0, N_SIMULATIONS) if 'H2O_weight' in tab3_selected_bases else np.zeros(N_SIMULATIONS)
                    h3po4_w = np.random.uniform(1.0, 25.0, N_SIMULATIONS) if 'H3PO4_weight' in tab3_selected_bases else np.zeros(N_SIMULATIONS)
                    h2o2_w = np.random.uniform(1.0, 25.0, N_SIMULATIONS) if 'H2O2_weight' in tab3_selected_bases else np.zeros(N_SIMULATIONS)
                    
                    for sim_data in [sim_data_1, sim_data_0]:
                        sim_data['H2O_weight'], sim_data['H3PO4_weight'], sim_data['H2O2_weight'] = h2o_w, h3po4_w, h2o2_w
                        
                    for chem in tab3_selected_chems:
                        rand_w = np.random.uniform(0.1, 8.0, N_SIMULATIONS)
                        sim_data_1[chem], sim_data_0[chem] = rand_w, rand_w
                        
                    df_1 = pd.DataFrame(sim_data_1)
                    df_0 = pd.DataFrame(sim_data_0)
                    
                    # --- 將隨機生成的重量也轉換為百分比 (即總重 100g 比例) ---
                    df_1_pct = convert_to_wt_pct(df_1, st.session_state.feature_cols)
                    df_0_pct = convert_to_wt_pct(df_0, st.session_state.feature_cols)
                    
                    # 使用百分比送進 AI 預測
                    pred_snag_1 = st.session_state.models['xgb_snag'].predict(df_1_pct)
                    pred_cuni_1 = st.session_state.models['xgb_cu_ni'].predict(df_1_pct)
                    pred_snag_0 = st.session_state.models['xgb_snag'].predict(df_0_pct)
                    pred_cuni_0 = st.session_state.models['xgb_cu_ni'].predict(df_0_pct)
                    
                    error_1 = np.abs(pred_snag_1 - target_snag) + np.abs(pred_cuni_1 - target_cu_ni)
                    error_0 = np.abs(pred_snag_0 - target_snag) + np.abs(pred_cuni_0 - target_cu_ni)
                    total_error = error_1 + error_0
                    
                    # 儲存結果時，直接存取 df_1_pct (因為它等於總重 100g 時的精確公克數)
                    results_dict = {
                        'error': total_error,
                        'pred_snag_1': pred_snag_1, 'pred_cuni_1': pred_cuni_1,
                        'pred_snag_0': pred_snag_0, 'pred_cuni_0': pred_cuni_0
                    }
                    if 'H2O_weight' in tab3_selected_bases: results_dict['H2O_weight'] = df_1_pct['H2O_weight']
                    if 'H3PO4_weight' in tab3_selected_bases: results_dict['H3PO4_weight'] = df_1_pct['H3PO4_weight']
                    if 'H2O2_weight' in tab3_selected_bases: results_dict['H2O2_weight'] = df_1_pct['H2O2_weight']
                    
                    for chem in tab3_selected_chems:
                        results_dict[chem] = df_1_pct[chem]
                        
                    results_df = pd.DataFrame(results_dict)
                    top3 = results_df.sort_values('error').head(3)
                    
                st.success("🎉 探索完成！AI 已將推薦配方自動換算為「總計 100g」的秤重比例：")
                
                for idx, row in top3.reset_index(drop=True).iterrows():
                    st.markdown(f"### 🏆 推薦配方 Top {idx + 1} (總重 100g)")
                    
                    res_c1, res_c2 = st.columns(2)
                    with res_c1:
                        st.info("**【密區 (1)】預測結果**")
                        st.metric("Snag Cu", f"{row['pred_snag_1']:.3f} um", delta=f"{row['pred_snag_1'] - target_snag:.3f}", delta_color="inverse")
                        st.metric("Cu Ni", f"{row['pred_cuni_1']:.3f} um", delta=f"{row['pred_cuni_1'] - target_cu_ni:.3f}", delta_color="inverse")
                    with res_c2:
                        st.success("**【疏區 (0)】預測結果**")
                        st.metric("Snag Cu", f"{row['pred_snag_0']:.3f} um", delta=f"{row['pred_snag_0'] - target_snag:.3f}", delta_color="inverse")
                        st.metric("Cu Ni", f"{row['pred_cuni_0']:.3f} um", delta=f"{row['pred_cuni_0'] - target_cu_ni:.3f}", delta_color="inverse")
                    
                    formula_parts, weight_parts = [], []
                    if 'H2O_weight' in tab3_selected_bases:
                        formula_parts.append("H2O")
                        weight_parts.append(f"{row['H2O_weight']:.2f}")
                    if 'H3PO4_weight' in tab3_selected_bases:
                        formula_parts.append("H3PO4")
                        weight_parts.append(f"{row['H3PO4_weight']:.2f}")
                    if 'H2O2_weight' in tab3_selected_bases:
                        formula_parts.append("H2O2")
                        weight_parts.append(f"{row['H2O2_weight']:.2f}")
                        
                    for chem in tab3_selected_chems:
                        formula_parts.append(chem.replace('_weight', ''))
                        weight_parts.append(f"{row[chem]:.2f}")
                    
                    formula_str = "+".join(formula_parts)
                    weight_str = "+".join(weight_parts)
                    
                    st.code(f"chemical_formula: {formula_str}\nchemical_weights: {weight_str}", language="text")
                    st.divider()

# ------------------------------------------
# 分頁四：實驗結果登錄
# ------------------------------------------
with tab4:
    st.header("📝 登錄真實實驗結果")
    if st.session_state.df is None:
        st.info("請先至「資料載入」分頁上傳 CSV 檔案。")
    else:
        st.write("不論您是使用「正向推測」或「逆向探索」，完成實際實驗後，請在此處將最終配方與真實數據寫入資料庫。")
        
        # 使用 3 欄並排，左邊配方，右邊結果
        t4_c1, t4_c2, t4_c3 = st.columns([1, 1, 1])
        
        with t4_c1:
            st.markdown("### 🌡️ 1. 環境與基底")
            t4_temp = st.number_input("溫度 (temp)", value=25.0, step=1.0, key="t4_temp")
            t4_region = st.selectbox("區域 (region)", options=[1, 0], format_func=lambda x: "密區 (1)" if x == 1 else "疏區 (0)", key="t4_region")
            st.divider()
            t4_h2o = st.number_input("H2O 重量", value=60.0, step=1.0, key="t4_h2o")
            t4_h3po4 = st.number_input("H3PO4 重量", value=10.0, step=1.0, key="t4_h3po4")
            t4_h2o2 = st.number_input("H2O2 重量", value=15.0, step=1.0, key="t4_h2o2")

        with t4_c2:
            st.markdown("### 🧪 2. 實際添加物")
            base_features = ['temp', 'region', 'H2O_weight', 'H3PO4_weight', 'H2O2_weight']
            optional_chems = [c for c in st.session_state.feature_cols if c not in base_features]
            categorized_chems = [chem for sublist in CHEMICAL_CATEGORIES.values() for chem in sublist]
            uncategorized_chems = [c for c in optional_chems if c not in categorized_chems]
            
            t4_chem_inputs = {}
            t4_selected_count = 0
            
            # 使用手風琴選單讓操作者勾選實際用到的添加物
            for cat_name, chems_in_cat in CHEMICAL_CATEGORIES.items():
                valid_chems = [c for c in chems_in_cat if c in optional_chems]
                if valid_chems:
                    with st.expander(cat_name):
                        for chem in valid_chems:
                            if st.checkbox(chem.replace('_weight', ''), key=f"t4_chk_{chem}"):
                                t4_selected_count += 1
                                t4_chem_inputs[chem] = st.number_input(f"重量", value=1.0, step=0.1, key=f"t4_num_{chem}")
            
            if uncategorized_chems:
                with st.expander("📦 其他未分類添加物"):
                    for chem in uncategorized_chems:
                        if st.checkbox(chem.replace('_weight', ''), key=f"t4_chk_{chem}"):
                            t4_selected_count += 1
                            t4_chem_inputs[chem] = st.number_input(f"重量", value=1.0, step=0.1, key=f"t4_num_{chem}")
                            
            if t4_selected_count > 10:
                st.error("⚠️ 選擇了超過 10 種添加物，請確認是否輸入錯誤。")

        with t4_c3:
            st.markdown("### 🔬 3. 真實實驗結果")
            t4_real_snag = st.number_input("真實 Snag Cu (um)", value=0.0, step=0.01, key="t4_snag")
            t4_real_cu_ni = st.number_input("真實 Cu Ni (um)", value=0.0, step=0.01, key="t4_cuni")
            t4_real_time = st.number_input("蝕刻時間 (sec)", value=0, step=1, key="t4_time")
            t4_real_result = st.text_area("實驗備註 (Result)", placeholder="請量化數據，吃得很乾淨100％、藥水有點濁...", height=110, key="t4_res")
            
            st.divider()
            if st.button("💾 寫入資料並重訓 AI 模型", use_container_width=True, type="primary"):
                if t4_selected_count <= 10:
                    # 建立新資料列
                    new_row = {col: 0.0 for col in st.session_state.df.columns}
                    new_row['temp'], new_row['region'] = t4_temp, t4_region
                    new_row['H2O_weight'], new_row['H3PO4_weight'], new_row['H2O2_weight'] = t4_h2o, t4_h3po4, t4_h2o2
                    
                    # 準備用來寫入字串的陣列
                    formula_parts = ["H2O", "H3PO4", "H2O2"]
                    weight_parts = [f"{t4_h2o:.2f}", f"{t4_h3po4:.2f}", f"{t4_h2o2:.2f}"]
                    
                    # 填寫添加物重量與字串
                    for chem, val in t4_chem_inputs.items():
                        new_row[chem] = val
                        formula_parts.append(chem.replace('_weight', ''))
                        weight_parts.append(f"{val:.2f}")
                        
                    # 合成人類可讀的配方字串 (解決歷史紀錄空缺的問題)
                    new_row['chemical_formula'] = "+".join(formula_parts)
                    new_row['chemical_weights'] = "+".join(weight_parts)
                        
                    # 填寫實驗結果
                    new_row['snag_cu_undercut_um'], new_row['cu_ni_undercut_um'] = t4_real_snag, t4_real_cu_ni
                    new_row['etch_time_value_sec'], new_row['result'] = t4_real_time, t4_real_result
                    new_row['date_folder'], new_row['item'] = datetime.now().strftime("%Y%m%d"), "NEW"
                    
                    # 將新資料寫入暫存資料庫
                    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
                    
                    # 呼叫重訓函式，讓 AI 吸收新知識
                    train_models(st.session_state.df)
                    
                    st.success("✅ 資料已新增！模型已在背景吸收新經驗，變得更聰明了。")
                    st.info("💡 提示：您可以直接回到前幾個分頁繼續預測。如果準備下班，請點擊左側邊欄的「📥 下載最新 CSV 備份」存檔。")
