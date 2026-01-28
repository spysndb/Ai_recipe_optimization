import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

# 嘗試匯入 XGBoost，如果沒安裝就不啟用該選項
try:
    from xgboost import XGBRegressor
    has_xgboost = True
except ImportError:
    has_xgboost = False

# ==========================================
# 頁面設定
# ==========================================
st.set_page_config(page_title="AI 配方優化大師 V4", layout="wide")
st.title("🧪 AI 配方實驗室 - 智慧預測系統")

# ==========================================
# Session State 初始化
# ==========================================
if 'input_count' not in st.session_state:
    st.session_state['input_count'] = 5
if 'output_count' not in st.session_state:
    st.session_state['output_count'] = 5

if 'col_map' not in st.session_state:
    st.session_state['col_map'] = {} 

if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame()

# 輔助函式：取得顯示名稱
def get_name(prefix, index, type_key):
    key = f"{type_key}_{index}" 
    default_name = f"{prefix}_{index+1:02d}" 
    return st.session_state['col_map'].get(key, default_name)

# ==========================================
# 建立四個分頁
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ 1. 設定 (Settings)", 
    "📝 2. 已知輸入 (Data)", 
    "🏷️ 3. 欄位命名 (Naming)", 
    "🎯 4. AI 預測 (Prediction)"
])

# ==========================================
# 分頁 1: 設定數量
# ==========================================
with tab1:
    st.header("1. 設定實驗參數數量")
    
    col1, col2 = st.columns(2)
    with col1:
        temp_in = st.number_input("輸入格數 (原料/變數)", min_value=1, value=st.session_state['input_count'])
    with col2:
        temp_out = st.number_input("輸出格數 (測量結果)", min_value=1, value=st.session_state['output_count'])

    if st.button("確認並重置表格", type="primary"):
        st.session_state['input_count'] = temp_in
        st.session_state['output_count'] = temp_out
        
        cols = [f"input_{i}" for i in range(temp_in)] + [f"output_{i}" for i in range(temp_out)]
        st.session_state['data'] = pd.DataFrame(columns=cols)
        st.session_state['col_map'] = {}
        
        st.success(f"✅ 已重置：{temp_in} 個輸入, {temp_out} 個輸出。")

# ==========================================
# 分頁 2: 輸入資料
# ==========================================
with tab2:
    st.header("2. 輸入歷史實驗數據")
    
    df_display = st.session_state['data'].copy()
    
    rename_dict = {}
    for i in range(st.session_state['input_count']):
        rename_dict[f"input_{i}"] = get_name("輸入", i, "input")
    for i in range(st.session_state['output_count']):
        rename_dict[f"output_{i}"] = get_name("輸出", i, "output")
        
    df_display = df_display.rename(columns=rename_dict)

    edited_df = st.data_editor(
        df_display,
        num_rows="dynamic",
        use_container_width=True,
        height=400
    )

    if not edited_df.equals(df_display):
        reverse_rename = {v: k for k, v in rename_dict.items()}
        st.session_state['data'] = edited_df.rename(columns=reverse_rename)

# ==========================================
# 分頁 3: 欄位命名
# ==========================================
with tab3:
    st.header("3. 自訂欄位名稱")
    
    st.subheader("1. 設定目標名稱 (Outputs Name)")
    cols = st.columns(5)
    for i in range(st.session_state['output_count']):
        key_id = f"output_{i}"
        current_name = get_name("輸出", i, "output")
        with cols[i % 5]:
            new_name = st.text_input(f"輸出_{i+1:02d} 名稱", value=current_name, key=f"rename_out_{i}")
            if new_name:
                st.session_state['col_map'][key_id] = new_name

    st.divider()

    st.subheader("2. 設定原料名稱 (Inputs Name)")
    cols_in = st.columns(5)
    for i in range(st.session_state['input_count']):
        key_id = f"input_{i}"
        current_name = get_name("輸入", i, "input")
        with cols_in[i % 5]:
            new_name = st.text_input(f"輸入_{i+1:02d} 名稱", value=current_name, key=f"rename_in_{i}")
            if new_name:
                st.session_state['col_map'][key_id] = new_name

# ==========================================
# 分頁 4: AI 預測 (含 4 種演算法)
# ==========================================
with tab4:
    st.header("4. 設定目標與預測")
    
    df = st.session_state['data']
    
    if len(df) < 2:
        st.warning("⚠️ 資料不足！請先在「分頁 2」輸入至少 2 筆數據。")
    else:
        # --- 1. 設定目標 ---
        st.subheader("1. 設定你的目標 (Output Targets)")
        targets = {}
        cols = st.columns(5)
        for i in range(st.session_state['output_count']):
            display_name = get_name("輸出", i, "output") 
            col_id = f"output_{i}"
            with cols[i % 5]:
                val = st.text_input(f"{display_name} 目標", key=f"target_{i}")
                if val:
                    try:
                        targets[col_id] = float(val)
                    except:
                        pass

        st.divider()

        # --- 2. 設定原料限制 + 自動補 0 按鈕 ---
        c_title, c_btn = st.columns([5, 1])
        with c_title:
            st.subheader("2. 設定原料限制 (Input Constraints)")
        with c_btn:
            if st.button("🧹 將空白處補 0"):
                for i in range(st.session_state['input_count']):
                    key = f"const_{i}"
                    if key not in st.session_state or st.session_state[key] == "":
                        st.session_state[key] = "0"
                st.rerun()

        constraints = {}
        cols_in = st.columns(5)
        for i in range(st.session_state['input_count']):
            display_name = get_name("輸入", i, "input")
            col_id = f"input_{i}"
            with cols_in[i % 5]:
                val = st.text_input(f"{display_name} 固定", key=f"const_{i}")
                if val:
                    try:
                        constraints[col_id] = float(val)
                    except:
                        pass
        
        st.divider()

        # --- 3. 選擇演算法 ---
        st.subheader("3. 選擇 AI 演算法 (Model Selection)")
        
        algo_options = ["隨機森林 (Random Forest)", "支持向量機 (SVR)", "線性回歸 (Linear Regression)"]
        if has_xgboost:
            algo_options.append("XGBoost")
        else:
            st.caption("ℹ️ 尚未安裝 XGBoost 函式庫，因此無法選擇該選項。")

        selected_algo = st.selectbox("請選擇運算模型：", algo_options)

        # --- 4. 執行按鈕 ---
        st.markdown("---")
        if st.button("🚀 開始 AI 運算 (生成推薦配方)", type="primary"):
            if not targets:
                st.error("❌ 請至少設定一個「目標」數值！")
            else:
                with st.spinner(f'正在使用 {selected_algo} 進行運算...'):
                    try:
                        work_df = df.copy()
                        for col in work_df.columns:
                            work_df[col] = pd.to_numeric(work_df[col], errors='coerce').fillna(0)
                        
                        input_ids = [f"input_{i}" for i in range(st.session_state['input_count'])]
                        output_ids = [f"output_{i}" for i in range(st.session_state['output_count'])]
                        
                        X = work_df[input_ids]
                        y = work_df[output_ids]
                    except Exception as e:
                        st.error(f"資料錯誤: {e}")
                        st.stop()

                    # 選擇模型
                    if "Random Forest" in selected_algo:
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    elif "SVR" in selected_algo:
                        # SVR 預設不支援多輸出，需用 MultiOutput 或是分別訓練，這裡用 sklearn 的自動適配(部分版本)
                        # 為了保險，這裡針對多目標輸出做簡單處理：如果 y 是多欄位，RandomForest 原生支援，
                        # 但 SVR 和 LinearRegression 建議對每一欄位分別預測。
                        # 為了簡化程式碼邏輯，這裡我們使用 RandomForestWrapper 的概念讓所有模型都能跑多輸出
                        # 但為了不讓程式碼太複雜，我們先切換回 Wrapper 方式
                        from sklearn.multioutput import MultiOutputRegressor
                        model = MultiOutputRegressor(SVR())
                    elif "Linear Regression" in selected_algo:
                        model = LinearRegression()
                    elif "XGBoost" in selected_algo:
                         from sklearn.multioutput import MultiOutputRegressor
                         model = MultiOutputRegressor(XGBRegressor(n_estimators=100, random_state=42))

                    # 訓練模型
                    model.fit(X, y)
                    
                    # 模擬配方
                    random_recipes = []
                    for _ in range(50000):
                        recipe = {}
                        for col_id in input_ids:
                            if col_id in constraints:
                                recipe[col_id] = constraints[col_id]
                            else:
                                min_v = work_df[col_id].min()
                                max_v = work_df[col_id].max()
                                if max_v == 0:
                                    recipe[col_id] = 0.0
                                else:
                                    # 擴大一點搜尋範圍
                                    low = min_v * 0.8
                                    high = max_v * 1.2
                                    if low < 0: low = 0
                                    recipe[col_id] = np.random.uniform(low, high)
                        random_recipes.append(recipe)
                    
                    virtual_df = pd.DataFrame(random_recipes)
                    preds = model.predict(virtual_df)
                    
                    pred_df = pd.DataFrame(preds, columns=output_ids)
                    virtual_df = pd.concat([virtual_df, pred_df], axis=1)
                    
                    # 計算誤差
                    virtual_df['Total_Error'] = 0
                    for t_id, t_val in targets.items():
                        virtual_df['Total_Error'] += abs(virtual_df[t_id] - t_val)
                    
                    best_results = virtual_df.sort_values('Total_Error').head(3)
                    
                    st.success(f"運算完成！使用模型：{selected_algo}")
                    
                    for idx, (_, row) in enumerate(best_results.iterrows()):
                        with st.expander(f"🏆 推薦配方 #{idx+1} (誤差: {row['Total_Error']:.4f})", expanded=True):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("### 🧪 建議配方")
                                input_show = {}
                                for i in range(st.session_state['input_count']):
                                    key = f"input_{i}"
                                    val = row[key]
                                    if val > 0.001 or key in constraints:
                                        name = get_name("輸入", i, "input")
                                        input_show[name] = round(val, 3)
                                st.json(input_show)
                            with c2:
                                st.markdown("### 📉 預測結果")
                                output_show = {}
                                for i in range(st.session_state['output_count']):
                                    key = f"output_{i}"
                                    val = row[key]
                                    name = get_name("輸出", i, "output")
                                    target_str = f"(目標 {targets[key]})" if key in targets else ""
                                    output_show[f"{name} {target_str}"] = round(val, 4)
                                st.json(output_show)