import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from datetime import datetime
import io

# ==========================================
# 1. 頁面基本設定與分類字典
# ==========================================
st.set_page_config(page_title="蝕刻配方推測系統", page_icon="🧪", layout="wide")
st.title("🧪 智慧化學蝕刻配方推測系統 (多模型版)")

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
# 存放訓練好的模型與標準化工具
if "models" not in st.session_state:
    st.session_state.models = {}
if "scaler" not in st.session_state:
    st.session_state.scaler = None

# ==========================================
# 2. 核心函式定義：訓練三種模型
# ==========================================
def train_models(df):
    feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS + TARGET_COLS]
    st.session_state.feature_cols = feature_cols
    
    X = df[feature_cols].fillna(0)
    y_snag = df['snag_cu_undercut_um'].fillna(0)
    y_cu_ni = df['cu_ni_undercut_um'].fillna(0)
    
    # 建立標準化工具 (給 Ridge 迴歸使用)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.session_state.scaler = scaler
    
    # --- 1. 隨機森林 (保守派) ---
    rf_snag = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_snag)
    rf_cu_ni = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_cu_ni)
    
    # --- 2. XGBoost (敏銳派) ---
    xgb_snag = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X, y_snag)
    xgb_cu_ni = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X, y_cu_ni)
    
    # --- 3. 脊迴歸 Ridge (趨勢派) ---
    # Ridge 迴歸需要輸入標準化後的資料
    ridge_snag = Ridge(alpha=1.0).fit(X_scaled, y_snag)
    ridge_cu_ni = Ridge(alpha=1.0).fit(X_scaled, y_cu_ni)
    
    # 將所有模型存入 Session State
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
        st.success("✅ 三大 AI 模型 (RF, XGB, Ridge) 訓練完成")
        st.divider()
        csv_data = convert_df_to_csv(st.session_state.df)
        st.download_button("📥 下載最新 CSV 備份", data=csv_data, file_name=f"Etch_Recipe_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)
    else:
        st.warning("⚠️ 尚未載入資料")

# ==========================================
# 4. 主畫面分頁設定
# ==========================================
tab1, tab2, tab3 = st.tabs(["📂 分頁一：資料載入", "🧪 分頁二：正向推測與紀錄", "🎯 分頁三：逆向最佳配方探索"])

# ------------------------------------------
# 分頁一：資料載入
# ------------------------------------------
with tab1:
    uploaded_file = st.file_uploader("請上傳清洗後的寬表 CSV 檔案", type=["csv"])
    
    if uploaded_file is not None:
        # 建立一個檔案專屬標籤 (包含檔名與大小)，用來判斷是不是同一個檔案
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # 檢查：如果系統還沒讀過這個檔案，才執行訓練
        if st.session_state.get("current_file_id") != file_id:
            with st.spinner("⏳ AI 正在大腦中建立三種化學模型 (RF, XGB, Ridge)，請稍候幾秒鐘..."):
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                train_models(df)
                # 紀錄這個檔案已經訓練過了
                st.session_state["current_file_id"] = file_id
                
            st.success("✅ 資料讀取與多重模型訓練成功！您可以前往「正向推測」分頁。")
        else:
            # 如果已經訓練過，直接顯示就緒，不浪費時間重訓
            st.success("✅ 資料庫與模型已在記憶體中準備就緒！您可以放心操作。")

# ------------------------------------------
# 分頁二：正向推測與紀錄
# ------------------------------------------
with tab2:
    if st.session_state.df is None:
        st.info("請先至「資料載入」分頁上傳 CSV 檔案。")
    else:
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown("### 🌡️ 基礎環境")
            temp = st.number_input("溫度 (temp)", value=25.0, step=1.0)
            region = st.selectbox("區域 (region)", options=[1, 0], format_func=lambda x: "密區 (1)" if x == 1 else "疏區 (0)")
        
        with c2:
            st.markdown("### 💧 核心基底 (必填)")
            h2o = st.number_input("H2O 重量", value=60.0, step=1.0)
            h3po4 = st.number_input("H3PO4 重量", value=10.0, step=1.0)
            h2o2 = st.number_input("H2O2 重量", value=15.0, step=1.0)

        with c3:
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
                            use_chem = st.checkbox(chem.replace('_weight', ''), key=f"chk_{chem}")
                            if use_chem:
                                selected_count += 1
                                chem_inputs[chem] = st.number_input(f"重量", value=1.0, step=0.1, key=f"num_{chem}")
            
            if uncategorized_chems:
                with st.expander("📦 其他未分類添加物"):
                    for chem in uncategorized_chems:
                        use_chem = st.checkbox(chem.replace('_weight', ''), key=f"chk_{chem}")
                        if use_chem:
                            selected_count += 1
                            chem_inputs[chem] = st.number_input(f"重量", value=1.0, step=0.1, key=f"num_{chem}")
            
            if selected_count > 10:
                st.error(f"⚠️ 您目前選擇了 {selected_count} 種添加物，請減少至 10 種以內。")

        with c4:
            st.markdown("### 🤖 執行運算")
            btn_predict = st.button("🚀 執行多模型推測與歷史查詢", use_container_width=True, type="primary")

        # ==========================================
        # 執行推測與歷史查詢結果顯示
        # ==========================================
        if btn_predict:
            if selected_count <= 10:
                # 1. 準備輸入特徵
                input_data = {col: 0.0 for col in st.session_state.feature_cols}
                input_data['temp'], input_data['region'] = temp, region
                input_data['H2O_weight'], input_data['H3PO4_weight'], input_data['H2O2_weight'] = h2o, h3po4, h2o2
                for chem, val in chem_inputs.items():
                    input_data[chem] = val
                    
                input_df = pd.DataFrame([input_data])
                
                # Ridge 迴歸需要標準化特徵
                input_scaled = st.session_state.scaler.transform(input_df)
                
                # 2. 取得三種模型的預測結果
                rf_snag_val = st.session_state.models['rf_snag'].predict(input_df)[0]
                rf_cuni_val = st.session_state.models['rf_cu_ni'].predict(input_df)[0]
                
                xgb_snag_val = st.session_state.models['xgb_snag'].predict(input_df)[0]
                xgb_cuni_val = st.session_state.models['xgb_cu_ni'].predict(input_df)[0]
                
                ridge_snag_val = st.session_state.models['ridge_snag'].predict(input_scaled)[0]
                ridge_cuni_val = st.session_state.models['ridge_cu_ni'].predict(input_scaled)[0]
                
                # 3. 畫面顯示推測結果 (三欄並排)
                st.markdown("### 📊 多模型預測結果比較")
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.info("🌳 **隨機森林 (保守派)**\n\n參考相近歷史配方的平均表現，最為穩定保守。")
                    st.metric("預測 Snag Cu (um)", f"{rf_snag_val:.3f}")
                    st.metric("預測 Cu Ni (um)", f"{rf_cuni_val:.3f}")
                    
                with res_col2:
                    st.warning("⚡ **XGBoost (敏銳派)**\n\n對添加物微小差異敏感，能捕捉複雜的化學反應交集。")
                    st.metric("預測 Snag Cu (um)", f"{xgb_snag_val:.3f}")
                    st.metric("預測 Cu Ni (um)", f"{xgb_cuni_val:.3f}")
                    
                with res_col3:
                    st.success("📈 **脊迴歸 (趨勢派)**\n\n純數學趨勢外插，能預測出破歷史紀錄的高低數值。")
                    st.metric("預測 Snag Cu (um)", f"{ridge_snag_val:.3f}")
                    st.metric("預測 Cu Ni (um)", f"{ridge_cuni_val:.3f}")
                
                st.divider()

               # 4. 歷史配方查詢 (擴大範圍：容許最多多加 1 種添加物)
                st.markdown("#### 📚 歷史相似配方清單 (完全相同 = 🔴紅色字體，多加1種 = 預設顏色)")
                selected_chems = list(chem_inputs.keys())
                unselected_chems = [c for c in optional_chems if c not in selected_chems]
                
                # 條件一：使用者勾選的添加物，配方中必須都有 (大於 0)
                mask_selected = pd.Series(True, index=st.session_state.df.index)
                for c in selected_chems:
                    if c in st.session_state.df.columns:
                        mask_selected = mask_selected & (st.session_state.df[c] > 0)
                        
                # 條件二：使用者未勾選的添加物中，最多只能有 1 種大於 0
                valid_unselected = [c for c in unselected_chems if c in st.session_state.df.columns]
                extra_additive_count = (st.session_state.df[valid_unselected] > 0).sum(axis=1)
                mask_extra = extra_additive_count <= 1
                
                # 合併兩個條件過濾資料庫 (使用 copy 避免警告)
                matched_df = st.session_state.df[mask_selected & mask_extra].copy()
                
                if matched_df.empty:
                    st.info("💡 資料庫中目前沒有相近的歷史配方。這是一組全新的嘗試，請參考上方的 AI 預測！")
                else:
                    st.success(f"🔍 找到 {len(matched_df)} 筆相似的歷史配方！")
                    
                    # 紀錄這筆配方到底是「完全相同 (0)」還是「多加一種 (1)」，供後續上色使用
                    matched_df['extra_count'] = extra_additive_count[mask_selected & mask_extra]
                    
                    # 決定要顯示的欄位：包含多出來的那 1 種化學品
                    cols_to_show = set(selected_chems)
                    for c in valid_unselected:
                        if (matched_df[c] > 0).any():  
                            cols_to_show.add(c)
                            
                    # ✅ 需求修改：新增 'chemical_formula' 欄位，供人類閱讀
                    display_cols = ['item', 'date_folder', 'chemical_formula', 'region', 'H2O_weight', 'H3PO4_weight', 'H2O2_weight'] + list(cols_to_show) + ['snag_cu_undercut_um', 'cu_ni_undercut_um', 'result']
                    # 確保要顯示的欄位真的存在於 DataFrame 中
                    display_cols = [c for c in display_cols if c in matched_df.columns]
                    
                    df_to_display = matched_df[display_cols]
                    
                    # ✅ 需求修改：建立上色函式 (Pandas Styler)
                    def highlight_identical(row):
                        # 透過 index 去 matched_df 抓取剛才計算的 extra_count
                        is_identical = matched_df.loc[row.name, 'extra_count'] == 0
                        if is_identical:
                            # 完全相同：使用 Streamlit 官方的高亮紅色
                            return ['color: #ff4b4b'] * len(row)
                        else:
                            # 多加一種：維持預設顏色 (不強制標黑，以防深色模式看不見)
                            return [''] * len(row)

                    # 套用樣式設定
                    styled_df = df_to_display.style.apply(highlight_identical, axis=1)
                    
                    # 顯示帶有顏色的表格
                    st.dataframe(styled_df, use_container_width=True)

        st.divider()
        
        # 實驗結果回寫區
        with st.expander("📝 將真實實驗結果寫入資料庫 (做完實驗後填寫)", expanded=False):
            rc1, rc2, rc3 = st.columns(3)
            real_snag = rc1.number_input("真實 Snag Cu", value=0.0, step=0.01)
            real_cu_ni = rc2.number_input("真實 Cu Ni", value=0.0, step=0.01)
            real_time = rc3.number_input("蝕刻時間 (sec)", value=0, step=1)
            real_result = st.text_area("實驗備註 (Result)", placeholder="吃得很乾淨、藥水有點濁...")
            
            if st.button("💾 儲存真實數據並重訓三大模型"):
                new_row = {col: 0.0 for col in st.session_state.df.columns}
                new_row['temp'], new_row['region'] = temp, region
                new_row['H2O_weight'], new_row['H3PO4_weight'], new_row['H2O2_weight'] = h2o, h3po4, h2o2
                for chem, val in chem_inputs.items():
                    new_row[chem] = val
                    
                new_row['snag_cu_undercut_um'] = real_snag
                new_row['cu_ni_undercut_um'] = real_cu_ni
                new_row['etch_time_value_sec'] = real_time
                new_row['result'] = real_result
                new_row['date_folder'] = datetime.now().strftime("%Y%m%d")
                new_row['item'] = "NEW"
                
                st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
                train_models(st.session_state.df)
                st.success("✅ 資料已新增！模型已吸收新經驗，變得更聰明了。")

# ------------------------------------------
# 分頁三：逆向最佳配方探索 (全自訂原料組合)
# ------------------------------------------
with tab3:
    st.header("🎯 逆向尋找最佳配方 (XGBoost 萬次模擬引擎)")
    if st.session_state.df is None:
        st.info("請先至「資料載入」分頁上傳 CSV 檔案。")
    else:
        st.write("設定您的目標與【欲使用的原料】，AI 會在背景生成 10,000 組不同重量的組合，並找出最接近目標的黃金比例。")
        
        # 1. 目標設定區
        st.markdown("#### 1. 設定預測目標")
        tcol1, tcol2 = st.columns(2)
        target_snag = tcol1.number_input("🎯 目標 Snag Cu (um)", value=0.100, step=0.01, format="%.3f")
        target_cu_ni = tcol2.number_input("🎯 目標 Cu Ni (um)", value=0.100, step=0.01, format="%.3f")
        
        st.divider()
        
        # 2. 原料勾選區
        st.markdown("#### 2. 勾選欲使用的配方原料")
        st.caption("您可以自由勾選欲使用的基底與添加物。AI 會根據您勾選的項目，尋找最佳的重量比例。")
        
        tab3_selected_bases = []
        tab3_selected_chems = []
        
        # --- 新增：核心基底獨立分類 (預設打勾) ---
        base_chems = ['H2O_weight', 'H3PO4_weight', 'H2O2_weight']
        with st.expander("💧 核心基底 (預設使用)", expanded=True):
            b_col1, b_col2, b_col3 = st.columns(3)
            # 使用 value=True 讓它們預設被勾選
            if b_col1.checkbox("H2O", value=True, key="t3_H2O"):
                tab3_selected_bases.append('H2O_weight')
            if b_col2.checkbox("H3PO4", value=True, key="t3_H3PO4"):
                tab3_selected_bases.append('H3PO4_weight')
            if b_col3.checkbox("H2O2", value=True, key="t3_H2O2"):
                tab3_selected_bases.append('H2O2_weight')

        # --- 原有添加物分類 ---
        optional_chems = [c for c in st.session_state.feature_cols if c not in ['temp', 'region'] + base_chems]
        categorized_chems = [chem for sublist in CHEMICAL_CATEGORIES.values() for chem in sublist]
        uncategorized_chems = [c for c in optional_chems if c not in categorized_chems]
        
        # 使用 4 欄並排顯示分類
        cat_cols = st.columns(4)
        col_idx = 0
        
        for cat_name, chems_in_cat in CHEMICAL_CATEGORIES.items():
            valid_chems = [c for c in chems_in_cat if c in optional_chems]
            if valid_chems:
                with cat_cols[col_idx % 4].expander(cat_name):
                    for chem in valid_chems:
                        if st.checkbox(chem.replace('_weight', ''), key=f"t3_{chem}"):
                            tab3_selected_chems.append(chem)
                col_idx += 1
                
        if uncategorized_chems:
            with cat_cols[col_idx % 4].expander("📦 其他未分類添加物"):
                for chem in uncategorized_chems:
                    if st.checkbox(chem.replace('_weight', ''), key=f"t3_{chem}"):
                        tab3_selected_chems.append(chem)

        st.divider()
        
        # 3. 執行探索
        if st.button("🔍 開始 10,000 次平行宇宙模擬探索", type="primary", use_container_width=True):
            # 防呆：確保使用者至少有勾選一種原料
            if len(tab3_selected_bases) + len(tab3_selected_chems) == 0:
                st.warning("⚠️ 請至少勾選一種原料（基底或添加物），AI 才有辦法調配配方！")
            else:
                with st.spinner("AI 正在為您勾選的原料測試 10,000 種重量組合，請稍候..."):
                    N_SIMULATIONS = 10000
                    
                    sim_data_1 = {col: np.zeros(N_SIMULATIONS) for col in st.session_state.feature_cols}
                    sim_data_0 = {col: np.zeros(N_SIMULATIONS) for col in st.session_state.feature_cols}
                    
                    # 環境設定
                    for sim_data in [sim_data_1, sim_data_0]:
                        sim_data['temp'] = np.full(N_SIMULATIONS, 25.0)
                    sim_data_1['region'] = np.full(N_SIMULATIONS, 1) # 密區
                    sim_data_0['region'] = np.full(N_SIMULATIONS, 0) # 疏區
                    
                    # --- 動態處理基底的隨機重量 ---
                    # 有打勾才給亂數探索，沒打勾就是 0
                    h2o_w = np.random.uniform(40.0, 90.0, N_SIMULATIONS) if 'H2O_weight' in tab3_selected_bases else np.zeros(N_SIMULATIONS)
                    h3po4_w = np.random.uniform(1.0, 25.0, N_SIMULATIONS) if 'H3PO4_weight' in tab3_selected_bases else np.zeros(N_SIMULATIONS)
                    h2o2_w = np.random.uniform(1.0, 25.0, N_SIMULATIONS) if 'H2O2_weight' in tab3_selected_bases else np.zeros(N_SIMULATIONS)
                    
                    for sim_data in [sim_data_1, sim_data_0]:
                        sim_data['H2O_weight'] = h2o_w
                        sim_data['H3PO4_weight'] = h3po4_w
                        sim_data['H2O2_weight'] = h2o2_w
                        
                    # 為使用者勾選的添加物生成隨機重量 (範圍 0.1 ~ 8.0)
                    for chem in tab3_selected_chems:
                        rand_w = np.random.uniform(0.1, 8.0, N_SIMULATIONS)
                        sim_data_1[chem] = rand_w
                        sim_data_0[chem] = rand_w
                        
                    df_1 = pd.DataFrame(sim_data_1)
                    df_0 = pd.DataFrame(sim_data_0)
                    
                    # XGBoost 預測
                    pred_snag_1 = st.session_state.models['xgb_snag'].predict(df_1)
                    pred_cuni_1 = st.session_state.models['xgb_cu_ni'].predict(df_1)
                    
                    pred_snag_0 = st.session_state.models['xgb_snag'].predict(df_0)
                    pred_cuni_0 = st.session_state.models['xgb_cu_ni'].predict(df_0)
                    
                    # 綜合誤差計算
                    error_1 = np.abs(pred_snag_1 - target_snag) + np.abs(pred_cuni_1 - target_cu_ni)
                    error_0 = np.abs(pred_snag_0 - target_snag) + np.abs(pred_cuni_0 - target_cu_ni)
                    total_error = error_1 + error_0
                    
                    # 整理輸出用的 DataFrame，只保留有被選到的欄位
                    results_dict = {
                        'error': total_error,
                        'pred_snag_1': pred_snag_1, 'pred_cuni_1': pred_cuni_1,
                        'pred_snag_0': pred_snag_0, 'pred_cuni_0': pred_cuni_0
                    }
                    if 'H2O_weight' in tab3_selected_bases: results_dict['H2O_weight'] = h2o_w
                    if 'H3PO4_weight' in tab3_selected_bases: results_dict['H3PO4_weight'] = h3po4_w
                    if 'H2O2_weight' in tab3_selected_bases: results_dict['H2O2_weight'] = h2o2_w
                    
                    for chem in tab3_selected_chems:
                        results_dict[chem] = df_1[chem]
                        
                    results_df = pd.DataFrame(results_dict)
                    
                    # 取出綜合誤差最小的 Top 3
                    top3 = results_df.sort_values('error').head(3)
                    
                st.success("🎉 探索完成！以下是 AI 在萬次模擬中，為您找出綜合表現最佳的 3 組比例：")
                
                # 顯示結果
                for idx, row in top3.reset_index(drop=True).iterrows():
                    st.markdown(f"### 🏆 推薦配方 Top {idx + 1}")
                    
                    res_c1, res_c2 = st.columns(2)
                    with res_c1:
                        st.info("**【密區 (1)】預測結果**")
                        st.metric("Snag Cu", f"{row['pred_snag_1']:.3f} um", delta=f"{row['pred_snag_1'] - target_snag:.3f}", delta_color="inverse")
                        st.metric("Cu Ni", f"{row['pred_cuni_1']:.3f} um", delta=f"{row['pred_cuni_1'] - target_cu_ni:.3f}", delta_color="inverse")
                    with res_c2:
                        st.success("**【疏區 (0)】預測結果**")
                        st.metric("Snag Cu", f"{row['pred_snag_0']:.3f} um", delta=f"{row['pred_snag_0'] - target_snag:.3f}", delta_color="inverse")
                        st.metric("Cu Ni", f"{row['pred_cuni_0']:.3f} um", delta=f"{row['pred_cuni_0'] - target_cu_ni:.3f}", delta_color="inverse")
                    
                    # 動態產出配方字串 (只包含真正打勾的項目)
                    formula_parts = []
                    weight_parts = []
                    
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
