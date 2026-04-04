import hashlib
import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


# 設定頁面基本資訊，讓 Streamlit Cloud 開啟時就有清楚的標題與版面。
st.set_page_config(
    page_title="V20 化學配方預測 App",
    page_icon="🧪",
    layout="wide",
)


# V20 必要欄位。
# 只要缺少其中任一欄，後面的訓練、預測與資料更新都會失去一致性，
# 因此在讀檔後會先檢查。
REQUIRED_COLUMNS = {
    "temp",
    "region",
    "H2O_weight",
    "H3PO4_weight",
    "H2O2_weight",
    "snag_cu_undercut_um",
    "cu_ni_undercut_um",
    "date_folder",
    "item",
    "chemical_formula",
    "chemical_weights",
    "etch_time_note",
    "etch_time_value_sec",
    "result",
}


# 這些欄位不會進入模型特徵。
# 原因是它們要嘛是純文字、要嘛是人工備註、要嘛是實驗結果，
# 若放進模型會造成資料洩漏或格式不相容。
NON_FEATURE_COLUMNS = {
    "date_folder",
    "item",
    "chemical_formula",
    "chemical_weights",
    "result",
    "etch_time_value_sec",
    "etch_time_note",
    "snag_cu_undercut_um",
    "cu_ni_undercut_um",
}


# 兩個要分開訓練的目標欄位。
TARGET_COLUMNS = [
    "snag_cu_undercut_um",
    "cu_ni_undercut_um",
]


# 使用者介面中的必填欄位。
# 這三個重量欄位與 temp、region 會直接顯示在主輸入區。
MANDATORY_INPUT_COLUMNS = [
    "temp",
    "H2O_weight",
    "H3PO4_weight",
    "H2O2_weight",
]


def init_session_state() -> None:
    """初始化需要跨 rerun 保存的資料。"""
    defaults = {
        "uploaded_file_hash": None,
        "uploaded_file_name": None,
        "uploaded_file_encoding": None,
        "raw_df": None,
        "working_df": None,
        "feature_columns": [],
        "chemical_columns": [],
        "optional_chemical_columns": [],
        "model_snag": None,
        "model_cu_ni": None,
        "training_row_count": 0,
        "dropped_row_count": 0,
        "last_prediction_input": None,
        "last_prediction_result": None,
        "last_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def read_csv_with_fallbacks(file_bytes: bytes) -> tuple[pd.DataFrame, str]:
    """
    讀取上傳的 CSV。

    使用者有可能是用不同編碼匯出 CSV，例如：
    - utf-8-sig
    - utf-8
    - cp950 / big5

    因此這裡依序嘗試多種編碼，盡量降低中文欄位亂碼或讀檔失敗的機率。
    """
    encodings = ["utf-8-sig", "utf-8", "cp950", "big5"]
    last_error = None

    for encoding in encodings:
        try:
            dataframe = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
            return dataframe, encoding
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise ValueError(f"CSV 無法讀取，請確認檔案格式或編碼。原始錯誤: {last_error}") from last_error


def validate_required_columns(dataframe: pd.DataFrame) -> list[str]:
    """回傳缺少的必要欄位名稱。"""
    return sorted(REQUIRED_COLUMNS - set(dataframe.columns))


def get_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    """
    根據 V20 規則取得模型特徵欄位。

    規則是：
    - 先排除展示用途欄位與目標欄位
    - 剩下的欄位全部都當作 X
    - 保留原始欄位順序，避免訓練與預測欄位順序不一致
    """
    return [column for column in dataframe.columns if column not in NON_FEATURE_COLUMNS]


def get_chemical_columns(feature_columns: list[str]) -> list[str]:
    """
    從特徵欄位中找出所有化學成分欄位。

    這裡用 `_weight` 結尾來辨識化學重量欄位，因為 V20 的化學欄位皆符合此命名方式。
    """
    return [column for column in feature_columns if column.endswith("_weight")]


def train_models(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[RandomForestRegressor, RandomForestRegressor, int, int]:
    """
    將 DataFrame 轉成可訓練的數值資料，並訓練兩個隨機森林模型。

    流程如下：
    1. 只取特徵欄位與目標欄位
    2. 用 `pd.to_numeric(errors="coerce")` 強制轉成數值
    3. 若有任一必要欄位轉成 NaN，該列資料就不參與訓練
    4. 各自訓練 `snag` 與 `cu_ni` 的模型
    """
    x_numeric = dataframe[feature_columns].apply(pd.to_numeric, errors="coerce")
    y_snag = pd.to_numeric(dataframe["snag_cu_undercut_um"], errors="coerce")
    y_cu_ni = pd.to_numeric(dataframe["cu_ni_undercut_um"], errors="coerce")

    valid_mask = x_numeric.notna().all(axis=1) & y_snag.notna() & y_cu_ni.notna()
    x_train = x_numeric.loc[valid_mask]
    y_snag_train = y_snag.loc[valid_mask]
    y_cu_ni_train = y_cu_ni.loc[valid_mask]

    if len(x_train) < 2:
        raise ValueError("可用於訓練的資料筆數不足，至少需要 2 筆完整數值資料。")

    model_snag = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model_cu_ni = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    model_snag.fit(x_train, y_snag_train)
    model_cu_ni.fit(x_train, y_cu_ni_train)

    dropped_row_count = int((~valid_mask).sum())
    training_row_count = int(valid_mask.sum())
    return model_snag, model_cu_ni, training_row_count, dropped_row_count


def refresh_models_from_dataframe(dataframe: pd.DataFrame) -> None:
    """
    重新根據目前記憶體中的 DataFrame 建立欄位資訊與模型。

    這個函式會在兩種情況被呼叫：
    - 使用者剛上傳 CSV
    - 使用者按下「更新資料」並 append 新資料之後
    """
    feature_columns = get_feature_columns(dataframe)
    chemical_columns = get_chemical_columns(feature_columns)
    optional_chemical_columns = [
        column
        for column in chemical_columns
        if column not in {"H2O_weight", "H3PO4_weight", "H2O2_weight"}
    ]

    model_snag, model_cu_ni, training_row_count, dropped_row_count = train_models(
        dataframe=dataframe,
        feature_columns=feature_columns,
    )

    st.session_state["working_df"] = dataframe
    st.session_state["feature_columns"] = feature_columns
    st.session_state["chemical_columns"] = chemical_columns
    st.session_state["optional_chemical_columns"] = optional_chemical_columns
    st.session_state["model_snag"] = model_snag
    st.session_state["model_cu_ni"] = model_cu_ni
    st.session_state["training_row_count"] = training_row_count
    st.session_state["dropped_row_count"] = dropped_row_count


def load_uploaded_file(uploaded_file) -> None:
    """
    將使用者上傳的 CSV 載入 session_state。

    為了避免每次畫面 rerun 都重新讀檔、重新訓練，
    這裡會先比對檔案內容的 SHA256 雜湊值。
    若是同一份檔案，就沿用現有 session_state。
    """
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state["uploaded_file_hash"] == file_hash:
        return

    dataframe, encoding = read_csv_with_fallbacks(file_bytes)
    missing_columns = validate_required_columns(dataframe)
    if missing_columns:
        missing_text = "、".join(missing_columns)
        raise ValueError(f"上傳的 CSV 缺少必要欄位：{missing_text}")

    st.session_state["uploaded_file_hash"] = file_hash
    st.session_state["uploaded_file_name"] = uploaded_file.name
    st.session_state["uploaded_file_encoding"] = encoding
    st.session_state["raw_df"] = dataframe.copy()
    st.session_state["last_prediction_input"] = None
    st.session_state["last_prediction_result"] = None

    refresh_models_from_dataframe(dataframe.copy())


def format_number(value: float) -> str:
    """把數字格式化成適合寫回 CSV 的字串，避免太多尾數。"""
    text = f"{float(value):.12f}".rstrip("0").rstrip(".")
    return text if text else "0"


def build_prediction_input(
    feature_columns: list[str],
    temp_value: float,
    region_value: int,
    h2o_value: float,
    h3po4_value: float,
    h2o2_value: float,
    extra_values: dict[str, float],
) -> pd.DataFrame:
    """
    建立單筆要送進模型的特徵列。

    核心做法是：
    - 先把所有特徵欄位補成 0
    - 再把使用者有輸入的欄位覆蓋上去
    這樣就能確保未選的化學成分都會自動補 0，且欄位順序與訓練一致。
    """
    row = {column: 0.0 for column in feature_columns}

    row["temp"] = float(temp_value)
    row["region"] = int(region_value)
    row["H2O_weight"] = float(h2o_value)
    row["H3PO4_weight"] = float(h3po4_value)
    row["H2O2_weight"] = float(h2o2_value)

    for column, value in extra_values.items():
        row[column] = float(value)

    return pd.DataFrame([row], columns=feature_columns)


def build_formula_strings(feature_row: dict[str, float], chemical_columns: list[str]) -> tuple[str, str]:
    """
    根據非 0 的化學欄位回推出 `chemical_formula` 與 `chemical_weights`。

    例如：
    - H2O_weight = 77.8
    - H3PO4_weight = 14
    - H2O2_weight = 0.2

    會組成：
    - chemical_formula = H2O+H3PO4+H2O2
    - chemical_weights = 77.8+14+0.2
    """
    chemical_names = []
    chemical_weights = []

    for column in chemical_columns:
        value = float(feature_row.get(column, 0.0))
        if abs(value) > 0:
            chemical_names.append(column.removesuffix("_weight"))
            chemical_weights.append(format_number(value))

    return "+".join(chemical_names), "+".join(chemical_weights)


def build_new_record(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    chemical_columns: list[str],
    prediction_input_df: pd.DataFrame,
    actual_snag: float,
    actual_cu_ni: float,
    etch_time_value_sec: float,
    result_text: str,
) -> pd.DataFrame:
    """
    將目前輸入組成一筆完整的 V20 資料列，並 append 到 DataFrame。

    重點是維持 V20 原欄位順序與欄位集合不變，這樣下載後的 CSV 才能與原始資料一致。
    """
    feature_row = prediction_input_df.iloc[0].to_dict()
    chemical_formula, chemical_weights = build_formula_strings(feature_row, chemical_columns)

    new_row = {}
    for column in dataframe.columns:
        if column in feature_columns:
            new_row[column] = feature_row.get(column, 0.0)
        elif column == "snag_cu_undercut_um":
            new_row[column] = float(actual_snag)
        elif column == "cu_ni_undercut_um":
            new_row[column] = float(actual_cu_ni)
        elif column == "etch_time_value_sec":
            new_row[column] = float(etch_time_value_sec)
        elif column == "date_folder":
            new_row[column] = datetime.now().strftime("%Y%m%d")
        elif column == "item":
            new_row[column] = ""
        elif column == "chemical_formula":
            new_row[column] = chemical_formula
        elif column == "chemical_weights":
            new_row[column] = chemical_weights
        elif column == "etch_time_note":
            new_row[column] = ""
        elif column == "result":
            new_row[column] = result_text
        else:
            # 若 V20 後續又新增欄位，這裡仍保留一個安全預設值：
            # 數值欄補 0，文字欄補空字串。
            if pd.api.types.is_numeric_dtype(dataframe[column]):
                new_row[column] = 0
            else:
                new_row[column] = ""

    new_row_df = pd.DataFrame([new_row], columns=dataframe.columns)
    return pd.concat([dataframe, new_row_df], ignore_index=True)


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """
    匯出成 UTF-8 with BOM。

    這樣使用者下載後若直接用 Excel 開啟，中文欄位與備註出現亂碼的機率會低很多。
    """
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def render_dataset_summary() -> None:
    """顯示目前載入資料與模型訓練狀態。"""
    working_df = st.session_state["working_df"]
    feature_columns = st.session_state["feature_columns"]
    chemical_columns = st.session_state["chemical_columns"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("資料筆數", len(working_df))
    col2.metric("特徵欄位數", len(feature_columns))
    col3.metric("化學欄位數", len(chemical_columns))
    col4.metric("有效訓練筆數", st.session_state["training_row_count"])

    if st.session_state["dropped_row_count"] > 0:
        st.warning(
            f"有 {st.session_state['dropped_row_count']} 筆資料因欄位不是完整數值，未納入模型訓練。"
        )
    else:
        st.success("所有資料列都成功納入模型訓練。")


def main() -> None:
    """主畫面。"""
    init_session_state()

    st.title("V20 CSV 上傳式化學配方預測 App")
    st.caption("上傳 V20 格式 CSV 後，系統會在記憶體中訓練兩個隨機森林模型，並提供雙目標預測與資料更新下載。")

    uploaded_file = st.file_uploader(
        "請上傳 V20 格式 CSV",
        type=["csv"],
        help="資料不會寫入伺服器資料庫，只會在目前工作階段記憶體中使用。",
    )

    if uploaded_file is None:
        st.info("請先上傳一份 V20 格式 CSV，系統才會顯示訓練、預測與更新資料功能。")
        return

    try:
        load_uploaded_file(uploaded_file)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return

    st.subheader("資料與模型狀態")
    st.write(
        f"目前檔案：`{st.session_state['uploaded_file_name']}`，"
        f"讀取編碼：`{st.session_state['uploaded_file_encoding']}`"
    )
    render_dataset_summary()

    st.divider()
    st.subheader("配方輸入與 AI 雙目標推測")

    optional_chemical_columns = st.session_state["optional_chemical_columns"]
    feature_columns = st.session_state["feature_columns"]
    chemical_columns = st.session_state["chemical_columns"]

    left_col, right_col = st.columns([1, 1])

    with left_col:
        temp_value = st.number_input(
            "temp",
            min_value=0.0,
            value=25.0,
            step=0.1,
            help="依你的需求預設為 25，但仍可手動修改。",
        )
        h2o_value = st.number_input("H2O_weight", min_value=0.0, value=0.0, step=0.1)
        h3po4_value = st.number_input("H3PO4_weight", min_value=0.0, value=0.0, step=0.1)
        h2o2_value = st.number_input("H2O2_weight", min_value=0.0, value=0.0, step=0.1)

    with right_col:
        region_label = st.radio(
            "region",
            options=["密區 (1)", "疏區 (0)"],
            index=0,
            horizontal=True,
            help="依照你的規格：密區=1，疏區=0。",
        )
        region_value = 1 if region_label == "密區 (1)" else 0

        selected_optional_chemicals = st.multiselect(
            "其餘化學成分（最多選 10 種）",
            options=optional_chemical_columns,
            default=[],
            help="未選的成分在預測時會自動補 0。",
        )

        if len(selected_optional_chemicals) > 10:
            st.error("最多只能選擇 10 種額外化學成分，請減少選項後再進行推測或更新資料。")

    extra_values = {}
    if selected_optional_chemicals:
        st.markdown("#### 額外化學成分輸入")
        extra_columns = st.columns(2)

        for index, chemical_column in enumerate(selected_optional_chemicals):
            extra_values[chemical_column] = extra_columns[index % 2].number_input(
                chemical_column,
                min_value=0.0,
                value=0.0,
                step=0.1,
                key=f"extra_input_{chemical_column}",
            )

    prediction_input_df = build_prediction_input(
        feature_columns=feature_columns,
        temp_value=temp_value,
        region_value=region_value,
        h2o_value=h2o_value,
        h3po4_value=h3po4_value,
        h2o2_value=h2o2_value,
        extra_values=extra_values,
    )

    preview_formula, preview_weights = build_formula_strings(
        prediction_input_df.iloc[0].to_dict(),
        chemical_columns,
    )
    st.caption(
        "目前配方預覽："
        f"`chemical_formula = {preview_formula or '(空白)'}`；"
        f"`chemical_weights = {preview_weights or '(空白)'}`"
    )

    if st.button("執行推測", type="primary", use_container_width=True):
        if len(selected_optional_chemicals) > 10:
            st.error("目前選擇的額外化學成分超過 10 種，請先減少後再推測。")
        else:
            snag_prediction = float(
                st.session_state["model_snag"].predict(prediction_input_df)[0]
            )
            cu_ni_prediction = float(
                st.session_state["model_cu_ni"].predict(prediction_input_df)[0]
            )

            st.session_state["last_prediction_input"] = prediction_input_df.copy()
            st.session_state["last_prediction_result"] = {
                "snag_cu_undercut_um": snag_prediction,
                "cu_ni_undercut_um": cu_ni_prediction,
            }

    if st.session_state["last_prediction_result"] is not None:
        st.markdown("#### 推測結果")
        result_col1, result_col2 = st.columns(2)
        result_col1.metric(
            "預測 snag_cu_undercut_um",
            format_number(st.session_state["last_prediction_result"]["snag_cu_undercut_um"]),
        )
        result_col2.metric(
            "預測 cu_ni_undercut_um",
            format_number(st.session_state["last_prediction_result"]["cu_ni_undercut_um"]),
        )

    st.divider()
    st.subheader("實驗結果更新與下載")
    st.write("填入真實實驗數值後，系統會將新紀錄 append 到目前記憶體中的資料，並立刻重訓模型。")

    update_col1, update_col2 = st.columns(2)
    with update_col1:
        actual_snag = st.number_input(
            "真實 snag_cu_undercut_um",
            min_value=0.0,
            value=0.0,
            step=0.001,
        )
        actual_cu_ni = st.number_input(
            "真實 cu_ni_undercut_um",
            min_value=0.0,
            value=0.0,
            step=0.001,
        )

    with update_col2:
        etch_time_value_sec = st.number_input(
            "真實 etch_time_value_sec",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )
        result_text = st.text_area(
            "文字備註 result",
            placeholder="請輸入這次實驗的備註、觀察或結論。",
            height=120,
        )

    if st.button("更新資料", use_container_width=True):
        if len(selected_optional_chemicals) > 10:
            st.error("目前選擇的額外化學成分超過 10 種，請先減少後再更新資料。")
        else:
            updated_df = build_new_record(
                dataframe=st.session_state["working_df"],
                feature_columns=feature_columns,
                chemical_columns=chemical_columns,
                prediction_input_df=prediction_input_df,
                actual_snag=actual_snag,
                actual_cu_ni=actual_cu_ni,
                etch_time_value_sec=etch_time_value_sec,
                result_text=result_text,
            )

            refresh_models_from_dataframe(updated_df)
            st.success("新資料已成功加入記憶體中的 DataFrame，並已完成模型重訓。")

    download_name = (
        f"{Path(st.session_state['uploaded_file_name']).stem}_updated.csv"
        if st.session_state["uploaded_file_name"]
        else "updated_v20.csv"
    )

    st.download_button(
        "下載更新後的 CSV",
        data=dataframe_to_csv_bytes(st.session_state["working_df"]),
        file_name=download_name,
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("查看最新 5 筆資料"):
        st.dataframe(st.session_state["working_df"].tail(5), use_container_width=True)


if __name__ == "__main__":
    main()
