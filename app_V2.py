import streamlit as st
import pandas as pd
import numpy as np

# === 新增演算法相關的引入 ===
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor # 用來讓單輸出模型(如SVR)支援多輸出

# ==========================================
# 頁面設定
# ==========================================
st.set_page_config(page_title="AI 配方優化大師", layout="wide")
st.title("🧪 AI 配方實驗室 - 智慧預測系統")

# 初始化 Session State (用來記憶跨分頁的資料)
if 'input_count' not in st.session_state:
    st.session_state['input_count'] = 5
if 'output_count' not in st.session_state:
    st.session_state['output_count'] = 5
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame()

# ==========================================
# 建立三個分頁
# ==========================================
tab1, tab2, tab3 = st.tabs(["⚙️ 1. 設定 (Settings)", "📝 2. 已知輸入 (Data)", "🎯 3. AI 預測 (Prediction)"])

# ==========================================
# 分頁 1: 設定輸入與輸出數量
# ==========================================
with tab1:
    st.header("設定實驗參數數量")
    
    col1, col2 = st.columns(2)
    with col1:
        temp_in = st.number_input("輸入格數 (原料/變數)", min_value=1, value=st.session_state['input_count'])
    with col2:
        temp_out = st.number_input("輸出格數 (測量結果)", min_value=1, value=st.session_state['output_count'])

    if st.button("確認並重置表格", type="primary"):
        st.session_state['input_count'] = temp_in
        st.session_state['output_count'] = temp_out
        
        # 產生欄位名稱
        input_cols = [f"輸入_{i+1:02d}" for i in range(temp_in)]
        output_cols = [f"輸出_{i+1:02d}" for i in range(temp_out)]
        all_cols = input_cols + output_cols
        
        # 重置 DataFrame
        st.session_state['data'] = pd.DataFrame(columns=all_cols)
        st.success(f"✅ 已設定：{temp_in} 個輸入, {temp_out} 個輸出。請前往分頁 2 輸入數據。")

# ==========================================
# 分頁 2: 輸入已知數據
# ==========================================
with tab2:
    st.header("輸入歷史實驗數據")
    st.caption("請在此貼上或輸入你的實驗紀錄。AI 會學習這些資料。")

    # 取得目前的欄位設定
    input_cols = [f"輸入_{i+1:02d}" for i in range(st.session_state['input_count'])]
    output_cols = [f"輸出_{i+1:02d}" for i in range(st.session_state['output_count'])]
    
    if len(st.session_state['data'].columns) == 0:
        st.warning("⚠️ 請先在「分頁 1」設定並按下確認按鈕。")
    else:
        # 顯示可編輯的表格 (Data Editor)
        edited_df = st.data_editor(
            st.session_state['data'],
            num_rows="dynamic",  # 允許使用者新增/刪除行
            use_container_width=True,
            height=400
        )
        
        # 當表格有變動時，自動存回 session_state
        if not edited_df.equals(st.session_state['data']):
            st.session_state['data'] = edited_df

# ==========================================
# 分頁 3: AI 預測與優化
# ==========================================
with tab3:
    st.header("設定目標與預測")
    
    # 檢查是否有足夠數據
    if len(st.session_state['data']) < 2:
        st.warning("⚠️ 資料不足！請先在「分頁 2」輸入至少 2 筆完整的實驗數據。")
    else:
        # 準備訓練資料
        df = st.session_state['data'].dropna() # 移除空值
        input_cols = [f"輸入_{i+1:02d}" for i in range(st.session_state['input_count'])]
        output_cols = [f"輸出_{i+1:02d}" for i in range(st.session_state['output_count'])]
        
        X = df[input_cols]
        y = df[output_cols]
        
        # 1. 設定目標區域
        st.subheader("1. 設定你的目標 (Output Targets)")
        st.caption("想要哪個結果達到特定數值？不填代表不限制。")
        
        targets = {}
        cols = st.columns(5)
        for i, col_name in enumerate(output_cols):
            with cols[i % 5]:
                val = st.text_input(f"{col_name} 目標", key=f"target_{col_name}")
                if val:
                    try:
                        targets[col_name] = float(val)
                    except ValueError:
                        st.error("請輸入數字")

        # 2. 設定限制區域
        st.subheader("2. 設定原料限制 (Input Constraints)")
        st.caption("有哪些原料想要固定用量？不填代表讓 AI 自由發揮。")
        
        constraints = {}
        cols_in = st.columns(5)
        for i, col_name in enumerate(input_cols):
            with cols_in[i % 5]:
                val = st.text_input(f"{col_name} 固定", key=f"const_{col_name}")
                if val:
                    try:
                        constraints[col_name] = float(val)
                    except ValueError:
                        st.error("請輸入數字")

        st.divider()
        
        # === 新增功能：模型選擇區 ===
        st.subheader("3. 選擇 AI 核心演算法")
        col_algo, col_btn = st.columns([1, 1])
        
        with col_algo:
            model_option = st.selectbox(
                "請選擇演算法模型：",
                [
                    "Random Forest (隨機森林 - 預設)",
                    "Gradient Boosting - 預測單一目標(類XGBoost)",
                    "SVR - 預測單一目標(支援向量機)",
                    "Gaussian Process (高斯過程)"
                ]
            )

        # 3. 執行按鈕
        with col_btn:
            st.write("") # 排版用，讓按鈕往下對齊
            st.write("") 
            run_btn = st.button("🚀 開始 AI 運算 (生成推薦配方)", type="primary", use_container_width=True)

        if run_btn:
            if not targets:
                st.error("❌ 請至少設定一個「目標」數值！")
            else:
                with st.spinner(f'正在使用 {model_option} 進行 50,000 次虛擬實驗...'):
                    
                    # 強制資料轉型
                    try:
                        for col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        X = df[input_cols]
                        y = df[output_cols]
                    except Exception as e:
                        st.error(f"資料轉換失敗: {e}")
                        st.stop()

                    # === 核心修改：根據選單建立對應的模型 ===
                    if "Random Forest" in model_option:
                        # 隨機森林原本就支援多輸出
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    
                    elif "Gradient Boosting" in model_option:
                        # GradientBoosting 預設只支援單輸出，需用 MultiOutputRegressor 包裝
                        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
                        model = MultiOutputRegressor(gb)
                    
                    elif "SVR" in model_option:
                        # SVR 需要標準化(StandardScaler)且需包裝成 MultiOutput
                        svr_pipeline = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
                        model = MultiOutputRegressor(svr_pipeline)
                    
                    elif "Gaussian Process" in model_option:
                        # 高斯過程非常需要標準化
                        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
                        # GP 支援原生的多輸出，但用 Pipeline 處理標準化比較安全
                        model = make_pipeline(
                            StandardScaler(), 
                            GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.1)
                        )

                    # 訓練模型
                    model.fit(X, y)
                    
                    # 虛擬實驗模擬 (產生 50,000 組)
                    random_recipes = []
                    # 預先計算好範圍，加速迴圈
                    bounds = {}
                    for col in input_cols:
                         bounds[col] = (df[col].min(), df[col].max())

                    for _ in range(50000):
                        recipe = {}
                        for col in input_cols:
                            if col in constraints:
                                recipe[col] = constraints[col] # 鎖定
                            else:
                                min_v, max_v = bounds[col]
                                if max_v == 0:
                                    recipe[col] = 0.0
                                else:
                                    low = min_v * 0.8
                                    high = max_v * 1.2
                                    if low < 0: low = 0
                                    recipe[col] = np.random.uniform(low, high)
                        random_recipes.append(recipe)
                    
                    virtual_df = pd.DataFrame(random_recipes)
                    
                    # 預測
                    predictions = model.predict(virtual_df)
                    
                    # 處理預測結果格式
                    pred_df = pd.DataFrame(predictions, columns=output_cols)
                    virtual_df = pd.concat([virtual_df, pred_df], axis=1)
                    
                    # 計算誤差
                    virtual_df['Total_Error'] = 0
                    for t_col, t_val in targets.items():
                        virtual_df['Total_Error'] += abs(virtual_df[t_col] - float(t_val))
                    
                    # 取前 3 名
                    best_results = virtual_df.sort_values('Total_Error').head(3)
                    
                    # --- 顯示結果 ---
                    st.success(f"運算完成！使用 [{model_option}] 推薦的最佳 3 組配方：")
                    
                    for idx, (index, row) in enumerate(best_results.iterrows()):
                        with st.expander(f"🏆 推薦配方 #{idx+1} (誤差: {row['Total_Error']:.4f})", expanded=True):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("### 🧪 建議配方 (Input)")
                                input_show = {k: round(v, 3) for k, v in row[input_cols].items() if v > 0.001 or k in constraints}
                                st.json(input_show)
                            with c2:
                                st.markdown("### 📉 預測結果 (Output)")
                                output_show = {}
                                for k, v in row[output_cols].items():
                                    target_str = f"(目標 {targets[k]})" if k in targets else ""
                                    output_show[f"{k} {target_str}"] = round(v, 4)
                                st.json(output_show)