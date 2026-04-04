import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import io

# ==========================================
# 1. 頁面基本設定與分類字典 (請在這裡修改您的化學品分類)
# ==========================================
st.set_page_config(page_title="蝕刻配方推測系統", page_icon="🧪", layout="wide")
st.title("🧪 智慧化學蝕刻配方推測系統")

# 定義不進入特徵運算的欄位與目標欄位
NON_FEATURE_COLS = ['date_folder', 'item', 'chemical_formula', 'chemical_weights', 'result', 'etch_time_value_sec', 'etch_time_note']
TARGET_COLS = ['snag_cu_undercut_um', 'cu_ni_undercut_um']

# 💡 【重要】在此處定義您的化學品分類！
# 請確保這裡的名稱與 CSV 檔案中的表頭完全一致（包含 _weight 字眼）
CHEMICAL_CATEGORIES = {
    "🛑 抑制劑 (Inhibitors)": [
        "1,2,4-三氮唑_weight", "5-甲基-1H-苯並三唑_weight", "BTA_weight", 
        "1H-1,2,3-三氮唑_weight", "5-ATZ_weight", "5-氯苯並三氮唑_weight"
    ],
    "🧬 聚合物與表面活性劑 (Polymers/Surfactants)": [
        "0.1%之PVP#3500水溶液_weight", "0.1%的聚丙烯醯胺水溶液_weight",
        "PEG #8000_weight", "500ppmPVP_weight", "PVP #3500_weight"
    ],
    "⚖️ pH與酸鹼調節 (pH Adjusters)": [
        "45%KOH_weight", "H2SO4_weight", "DGA_weight"
    ],
    "🔄 助劑與其他有機物 (Promoters/Organics)": [
        "2,3,5-三甲基吡嗪_weight", "2-甲基吡嗪_weight", "ATMP_weight"
    ]
}

# 初始化 Session State
if "df" not in st.session_state:
    st.session_state.df = None
if "model_snag" not in st.session_state:
    st.session_state.model_snag = None
if "model_cu_ni" not in st.session_state:
    st.session_state.model_cu_ni = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = []

# ==========================================
# 2. 核心函式定義
# ==========================================
def train_models(df):
    feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS + TARGET_COLS]
    st.session_state.feature_cols = feature_cols
    X = df[feature_cols].fillna(0)
    y_snag = df['snag_cu_undercut_um'].fillna(0)
    y_cu_ni = df['cu_ni_undercut_um'].fillna(0)
    
    model_snag = RandomForestRegressor(n_estimators=100, random_state=42)
    model_cu_ni = RandomForestRegressor(n_estimators=100, random_state=42)
    model_snag.fit(X, y_snag)
    model_cu_ni.fit(X, y_cu_ni)
    
    st.session_state.model_snag = model_snag
    st.session_state.model_cu_ni = model_cu_ni
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
        st.success("✅ AI 模型準備就緒")
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
# 分頁一：資料載入 (不變)
# ------------------------------------------
with tab1:
    uploaded_file = st.file_uploader("請上傳清洗後的寬表 CSV 檔案", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        train_models(df)
        st.success("資料讀取與模型訓練成功！")

# ------------------------------------------
# 分頁二：正向推測與紀錄 (大幅修改排版與分類)
# ------------------------------------------
with tab2:
    if st.session_state.df is None:
        st.info("請先至「資料載入」分頁上傳 CSV 檔案。")
    else:
        # 使用 4 個較窄的 column 來壓縮畫面，讓格子不要太寬
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
            
            # 建立一個字典來收集使用者輸入的化學品重量
            chem_inputs = {}
            selected_count = 0
            
            # 找出尚未被分類的化學品
            categorized_chems = [chem for sublist in CHEMICAL_CATEGORIES.values() for chem in sublist]
            uncategorized_chems = [c for c in optional_chems if c not in categorized_chems]
            
            # 繪製分類手風琴 (Expander)
            for cat_name, chems_in_cat in CHEMICAL_CATEGORIES.items():
                # 只顯示存在於目前 CSV 中的化學品
                valid_chems = [c for c in chems_in_cat if c in optional_chems]
                if valid_chems:
                    with st.expander(cat_name):
                        for chem in valid_chems:
                            use_chem = st.checkbox(chem.replace('_weight', ''), key=f"chk_{chem}")
                            if use_chem:
                                selected_count += 1
                                chem_inputs[chem] = st.number_input(f"重量", value=1.0, step=0.1, key=f"num_{chem}")
            
            # 其他未分類的化學品
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
            if st.button("🚀 執行 AI 推測", use_container_width=True, type="primary"):
                if selected_count <= 10:
                    input_data = {col: 0.0 for col in st.session_state.feature_cols}
                    input_data['temp'], input_data['region'] = temp, region
                    input_data['H2O_weight'], input_data['H3PO4_weight'], input_data['H2O2_weight'] = h2o, h3po4, h2o2
                    for chem, val in chem_inputs.items():
                        input_data[chem] = val
                        
                    input_df = pd.DataFrame([input_data])
                    pred_snag = st.session_state.model_snag.predict(input_df)[0]
                    pred_cu_ni = st.session_state.model_cu_ni.predict(input_df)[0]
                    
                    st.success("推測完成！")
                    st.metric("預測 Snag Cu (um)", f"{pred_snag:.3f}")
                    st.metric("預測 Cu Ni (um)", f"{pred_cu_ni:.3f}")

        st.divider()
        
        # 實驗結果回寫區
        with st.expander("📝 將真實實驗結果寫入資料庫 (點擊展開)", expanded=False):
            rc1, rc2, rc3 = st.columns(3)
            real_snag = rc1.number_input("真實 Snag Cu", value=0.0, step=0.01)
            real_cu_ni = rc2.number_input("真實 Cu Ni", value=0.0, step=0.01)
            real_time = rc3.number_input("蝕刻時間 (sec)", value=0, step=1)
            real_result = st.text_area("實驗備註 (Result)", placeholder="吃得很乾淨、藥水有點濁...")
            
            if st.button("💾 儲存真實數據並更新模型"):
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
# 分頁三：逆向最佳配方探索 (不變，沿用上一版的邏輯)
# ------------------------------------------
with tab3:
    st.write("此處保留為逆向探索功能區塊 (同上一版程式碼)")
    # (為了節省空間，請將上一版 Tab 3 的內容貼在這裡)
