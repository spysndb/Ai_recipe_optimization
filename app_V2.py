import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import io

# ==========================================
# 1. 頁面基本設定
# ==========================================
st.set_page_config(page_title="蝕刻配方推測系統", page_icon="🧪", layout="wide")
st.title("🧪 智慧化學蝕刻配方推測系統")

# 定義不進入特徵運算的欄位與目標欄位
NON_FEATURE_COLS = ['date_folder', 'item', 'chemical_formula', 'chemical_weights', 'result', 'etch_time_value_sec', 'etch_time_note']
TARGET_COLS = ['snag_cu_undercut_um', 'cu_ni_undercut_um']

# 初始化 Session State (用來在分頁間保留資料)
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
    """訓練兩個隨機森林模型"""
    # 找出所有特徵欄位 (排除非特徵與目標)
    feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS + TARGET_COLS]
    st.session_state.feature_cols = feature_cols
    
    X = df[feature_cols].fillna(0)
    y_snag = df['snag_cu_undercut_um'].fillna(0)
    y_cu_ni = df['cu_ni_undercut_um'].fillna(0)
    
    # 建立並訓練模型
    model_snag = RandomForestRegressor(n_estimators=100, random_state=42)
    model_cu_ni = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model_snag.fit(X, y_snag)
    model_cu_ni.fit(X, y_cu_ni)
    
    st.session_state.model_snag = model_snag
    st.session_state.model_cu_ni = model_cu_ni
    return feature_cols

def convert_df_to_csv(df):
    """將 DataFrame 轉為 CSV 格式供下載"""
    return df.to_csv(index=False).encode('utf-8-sig')

# ==========================================
# 3. 側邊欄設計
# ==========================================
with st.sidebar:
    st.header("系統狀態")
    if st.session_state.df is not None:
        st.success(f"✅ 資料庫已載入 ({len(st.session_state.df)} 筆)")
        st.success("✅ AI 模型訓練完成")
        st.divider()
        st.write("📥 備份最新資料庫")
        csv_data = convert_df_to_csv(st.session_state.df)
        st.download_button(
            label="下載更新後的 CSV",
            data=csv_data,
            file_name=f"Etch_Recipe_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("⚠️ 尚未載入資料")

# ==========================================
# 4. 主畫面分頁設定
# ==========================================
tab1, tab2, tab3 = st.tabs(["📂 資料載入", "🧪 正向推測與紀錄", "🎯 逆向最佳配方探索"])

# ------------------------------------------
# 分頁一：資料載入
# ------------------------------------------
with tab1:
    st.header("第一步：載入配方資料庫")
    uploaded_file = st.file_uploader("請上傳清洗後的寬表 CSV 檔案", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            train_models(df)
            st.success("資料讀取與模型訓練成功！您可以前往『正向推測』或『逆向探索』分頁。")
            with st.expander("預覽資料庫內容"):
                st.dataframe(df.head(10))
        except Exception as e:
            st.error(f"讀取失敗，請確認檔案格式。錯誤訊息：{e}")

# ------------------------------------------
# 分頁二：正向推測與紀錄
# ------------------------------------------
with tab2:
    st.header("🧪 測試新配方成效")
    if st.session_state.df is None:
        st.info("請先至「資料載入」分頁上傳 CSV 檔案。")
    else:
        # 分兩欄排版
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. 環境與基底設定")
            temp = st.number_input("溫度 (temp)", value=25.0, step=1.0)
            region = st.selectbox("區域 (region)", options=[1, 0], format_func=lambda x: "密區 (1)" if x == 1 else "疏區 (0)")
            h2o = st.number_input("H2O 重量", value=60.0, step=1.0)
            h3po4 = st.number_input("H3PO4 重量", value=10.0, step=1.0)
            h2o2 = st.number_input("H2O2 重量", value=15.0, step=1.0)
        
        with col2:
            st.subheader("2. 添加物設定 (最多10種)")
            # 挑出所有添加物清單 (排除環境變數與三大基底)
            base_features = ['temp', 'region', 'H2O_weight', 'H3PO4_weight', 'H2O2_weight']
            optional_chems = [c for c in st.session_state.feature_cols if c not in base_features]
            
            selected_chems = st.multiselect("選擇添加物", optional_chems, max_selections=10)
            
            chem_inputs = {}
            for chem in selected_chems:
                chem_inputs[chem] = st.number_input(f"{chem} 重量", value=1.0, step=0.1)

        st.divider()
        
        # 預測按鈕
        if st.button("🚀 執行 AI 推測", use_container_width=True, type="primary"):
            # 建立單筆特徵資料
            input_data = {col: 0.0 for col in st.session_state.feature_cols} # 全部先補 0
            input_data['temp'] = temp
            input_data['region'] = region
            input_data['H2O_weight'] = h2o
            input_data['H3PO4_weight'] = h3po4
            input_data['H2O2_weight'] = h2o2
            for chem, val in chem_inputs.items():
                input_data[chem] = val
                
            input_df = pd.DataFrame([input_data])
            
            # 預測
            pred_snag = st.session_state.model_snag.predict(input_df)[0]
            pred_cu_ni = st.session_state.model_cu_ni.predict(input_df)[0]
            
            st.success("推測完成！")
            mcol1, mcol2 = st.columns(2)
            mcol1.metric("預測 Snag Cu Undercut (um)", f"{pred_snag:.3f}")
            mcol2.metric("預測 Cu Ni Undercut (um)", f"{pred_cu_ni:.3f}")

        st.divider()
        
        # 實驗結果回寫區
        with st.expander("📝 將真實實驗結果寫入資料庫", expanded=False):
            st.write("做完實驗後，將真實數據填入並儲存，系統會自動重新訓練變聰明！")
            r_col1, r_col2, r_col3 = st.columns(3)
            real_snag = r_col1.number_input("真實 Snag Cu", value=0.0, step=0.01)
            real_cu_ni = r_col2.number_input("真實 Cu Ni", value=0.0, step=0.01)
            real_time = r_col3.number_input("蝕刻時間 (sec)", value=0, step=1)
            real_result = st.text_area("實驗備註 (Result)", placeholder="吃得很乾淨、藥水有點濁...")
            
            if st.button("💾 儲存並更新模型"):
                # 準備新資料列
                new_row = {col: 0.0 for col in st.session_state.df.columns}
                
                # 填入輸入特徵
                new_row['temp'] = temp
                new_row['region'] = region
                new_row['H2O_weight'] = h2o
                new_row['H3PO4_weight'] = h3po4
                new_row['H2O2_weight'] = h2o2
                for chem, val in chem_inputs.items():
                    new_row[chem] = val
                    
                # 填入目標與備註
                new_row['snag_cu_undercut_um'] = real_snag
                new_row['cu_ni_undercut_um'] = real_cu_ni
                new_row['etch_time_value_sec'] = real_time
                new_row['result'] = real_result
                new_row['date_folder'] = datetime.now().strftime("%Y%m%d")
                new_row['item'] = "NEW"
                
                # Append 並重訓
                st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
                train_models(st.session_state.df)
                st.success("資料已新增！模型已重新訓練。您可以到左側欄下載最新 CSV。")

# ------------------------------------------
# 分頁三：逆向最佳配方探索
# ------------------------------------------
with tab3:
    st.header("🎯 逆向尋找最佳配方 (蒙地卡羅法)")
    if st.session_state.df is None:
        st.info("請先至「資料載入」分頁上傳 CSV 檔案。")
    else:
        st.write("輸入您的目標，AI 會在背景生成 10,000 組隨機配方進行測試，並挑出最接近目標的組合。")
        
        tcol1, tcol2, tcol3 = st.columns(3)
        target_snag = tcol1.number_input("目標 Snag Cu (um)", value=0.1, step=0.01)
        target_cu_ni = tcol2.number_input("目標 Cu Ni (um)", value=0.1, step=0.01)
        target_region = tcol3.selectbox("針對區域最佳化", options=[1, 0], format_func=lambda x: "密區 (1)" if x == 1 else "疏區 (0)")
        
        st.divider()
        if st.button("🔍 開始萬次模擬探索", type="primary"):
            with st.spinner("AI 正在平行宇宙中測試 10,000 種配方，請稍候..."):
                N_SIMULATIONS = 10000
                
                # 1. 建立空矩陣
                sim_data = {col: np.zeros(N_SIMULATIONS) for col in st.session_state.feature_cols}
                
                # 2. 填入環境與基底隨機值 (範圍可依您的經驗在這邊修改程式碼)
                sim_data['temp'] = np.full(N_SIMULATIONS, 25.0)
                sim_data['region'] = np.full(N_SIMULATIONS, target_region)
                sim_data['H2O_weight'] = np.random.uniform(50, 80, N_SIMULATIONS)
                sim_data['H3PO4_weight'] = np.random.uniform(5, 20, N_SIMULATIONS)
                sim_data['H2O2_weight'] = np.random.uniform(5, 20, N_SIMULATIONS)
                
                # 3. 隨機加入添加物
                optional_chems = [c for c in st.session_state.feature_cols if c not in ['temp', 'region', 'H2O_weight', 'H3PO4_weight', 'H2O2_weight']]
                
                for i in range(N_SIMULATIONS):
                    # 隨機挑選 1 到 5 種添加物
                    num_additives = np.random.randint(1, 6) 
                    chosen_chems = np.random.choice(optional_chems, num_additives, replace=False)
                    for chem in chosen_chems:
                        # 隨機給予 0.1 ~ 5.0 的重量
                        sim_data[chem][i] = np.random.uniform(0.1, 5.0) 
                
                sim_df = pd.DataFrame(sim_data)
                
                # 4. 進行雙目標預測
                pred_snag_all = st.session_state.model_snag.predict(sim_df)
                pred_cu_ni_all = st.session_state.model_cu_ni.predict(sim_df)
                
                # 5. 計算誤差 (這裡採用絕對誤差總和，越小越好)
                error = np.abs(pred_snag_all - target_snag) + np.abs(pred_cu_ni_all - target_cu_ni)
                sim_df['error'] = error
                sim_df['pred_snag'] = pred_snag_all
                sim_df['pred_cu_ni'] = pred_cu_ni_all
                
                # 6. 取出 Top 3
                top3 = sim_df.sort_values('error').head(3)
                
            st.success("探索完成！以下是最接近您目標的 3 組配方：")
            
            # 顯示結果
            for idx, row in top3.reset_index(drop=True).iterrows():
                st.subheader(f"🏆 推薦配方 {idx + 1}")
                rcol1, rcol2 = st.columns(2)
                rcol1.metric("預測 Snag Cu", f"{row['pred_snag']:.3f} um")
                rcol2.metric("預測 Cu Ni", f"{row['pred_cu_ni']:.3f} um")
                
                # 只列出有添加的成分
                recipe_text = f"H2O: {row['H2O_weight']:.1f} | H3PO4: {row['H3PO4_weight']:.1f} | H2O2: {row['H2O2_weight']:.1f}"
                for chem in optional_chems:
                    if row[chem] > 0:
                        recipe_text += f" | {chem.replace('_weight', '')}: {row[chem]:.2f}"
                st.info(f"建議配方組成：\n{recipe_text}")
                st.divider()
