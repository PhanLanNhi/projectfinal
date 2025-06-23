import streamlit as st
import pandas as pd
import joblib

# ==== Load model ====
MODEL_DIR = r"D:\dow\project_final\evaluation"
xgb_model = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
lgbm_model = joblib.load(f"{MODEL_DIR}/lgbm_model.pkl")

# ==== Đặc trưng dùng cho model ====
sensor_cols = [
    'Normalized_Temp', 'Normalized_Vibration', 'Normalized_Pressure',
    'Normalized_Voltage', 'Normalized_Current',
    'FFT_Feature1', 'FFT_Feature2', 'Anomaly_Score'
]

st.set_page_config(page_title="Dự đoán lỗi thiết bị IoT", layout="wide")
st.title("Dự đoán lỗi thiết bị IoT")

tab1, tab2 = st.tabs(["Dự đoán theo file CSV", "Nhập giá trị cảm biến"])

def color_xgb(row):
    return ['background-color: #C9F7D3' if row['XGBoost_Prediction']==row['Fault_Status'] else 'background-color: #F8D7DA']*len(row)
def color_lgbm(row):
    return ['background-color: #C9F7D3' if row['LightGBM_Prediction']==row['Fault_Status'] else 'background-color: #F8D7DA']*len(row)
def color_compare(row):
    true_label = row['Fault_Status']
    pred_xgb = row['XGBoost_Prediction']
    pred_lgbm = row['LightGBM_Prediction']
    if pred_xgb == true_label and pred_lgbm == true_label:
        return ['background-color: #C9F7D3']*len(row)  # Xanh
    elif pred_xgb != true_label and pred_lgbm != true_label:
        return ['background-color: #F8D7DA']*len(row)  # Đỏ
    else:
        return ['background-color: #FFF3CD']*len(row)  # Vàng

# =========== TAB 1: DỰ ĐOÁN FILE CSV ===========
with tab1:
    st.write("## Tải lên file CSV")
    uploaded_file = st.file_uploader("", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Kiểm tra đủ cột
        missing = [c for c in sensor_cols if c not in df.columns]
        if missing:
            st.error("Thiếu cột đặc trưng: " + ', '.join(missing))
        else:
            X = df[sensor_cols]
            xgb_pred = xgb_model.predict(X)
            lgbm_pred = lgbm_model.predict(X)
            df_xgb = df.copy()
            df_lgbm = df.copy()
            df_xgb['XGBoost_Prediction'] = xgb_pred
            df_lgbm['LightGBM_Prediction'] = lgbm_pred
            compare_df = df.copy()
            compare_df['XGBoost_Prediction'] = xgb_pred
            compare_df['LightGBM_Prediction'] = lgbm_pred
            n_show = min(1000, len(df))

            # Bảng XGBoost
            st.subheader("Kết quả dự đoán với XGBoost")
            if 'Fault_Status' in df.columns:
                st.dataframe(df_xgb.head(n_show).style.apply(color_xgb, axis=1), use_container_width=True)
            else:
                st.dataframe(df_xgb.head(n_show), use_container_width=True)
            csv_xgb = df_xgb.to_csv(index=False).encode('utf-8')
            st.download_button("Tải bảng XGBoost", data=csv_xgb, file_name="result_xgboost.csv")

            # Bảng LightGBM
            st.subheader("Kết quả dự đoán với LightGBM")
            if 'Fault_Status' in df.columns:
                st.dataframe(df_lgbm.head(n_show).style.apply(color_lgbm, axis=1), use_container_width=True)
            else:
                st.dataframe(df_lgbm.head(n_show), use_container_width=True)
            csv_lgbm = df_lgbm.to_csv(index=False).encode('utf-8')
            st.download_button("Tải bảng LightGBM", data=csv_lgbm, file_name="result_lightgbm.csv")

            # Bảng so sánh
            st.subheader("So sánh cả hai mô hình")
            if 'Fault_Status' in df.columns:
                st.dataframe(compare_df.head(n_show).style.apply(color_compare, axis=1), use_container_width=True)
            else:
                st.dataframe(compare_df.head(n_show), use_container_width=True)
            csv_compare = compare_df.to_csv(index=False).encode('utf-8')
            st.download_button("Tải bảng so sánh", data=csv_compare, file_name="result_compare.csv")

            # Độ chính xác
            # if 'Fault_Status' in df.columns:
            #     acc_xgb = (df_xgb['XGBoost_Prediction'] == df_xgb['Fault_Status']).mean()
            #     acc_lgbm = (df_lgbm['LightGBM_Prediction'] == df_lgbm['Fault_Status']).mean()
            #     st.info(f"Độ chính xác XGBoost: **{acc_xgb*100:.2f}%**")
            #     st.info(f"Độ chính xác LightGBM: **{acc_lgbm*100:.2f}%**")
            # else:
            #     st.warning("Không có cột nhãn thật (Fault_Status) trong file để đánh giá độ chính xác.")

            st.markdown("""
            <b>Ý nghĩa màu sắc:</b>
            <ul>
            <li><span style="background-color:#C9F7D3;">&nbsp;&nbsp;&nbsp;&nbsp;</span> <b>Xanh nhạt</b>: Dự đoán đúng (so với cột nhãn thật <i>Fault_Status</i>)</li>
            <li><span style="background-color:#F8D7DA;">&nbsp;&nbsp;&nbsp;&nbsp;</span> <b>Đỏ nhạt</b>: Dự đoán sai</li>
            <li><span style="background-color:#FFF3CD;">&nbsp;&nbsp;&nbsp;&nbsp;</span> <b>Vàng nhạt</b> (chỉ bảng so sánh): Chỉ 1 trong 2 mô hình dự đoán đúng</li>
            </ul>
            """, unsafe_allow_html=True)
    else:
        st.info("Upload file .csv để bắt đầu.")

# =========== TAB 2: NHẬP GIÁ TRỊ CẢM BIẾN ===========
with tab2:
    st.write("## Nhập giá trị cảm biến để dự đoán")
    input_vals = {}
    cols = st.columns(4)
    for idx, col in enumerate(sensor_cols):
        with cols[idx % 4]:
            # Dùng key cố định để track thay đổi giá trị
            input_vals[col] = st.number_input(
                col, min_value=0.0, max_value=1.0, value=0.5, 
                step=0.01, format="%.3f", key=f"input_{col}"
            )

    # Reset kết quả nếu bất kỳ input nào thay đổi
    # Ý tưởng: Tạo 1 tuple lưu toàn bộ giá trị input, nếu khác với lần trước thì reset result
    inputs_tuple = tuple(input_vals.values())
    if "last_inputs" not in st.session_state:
        st.session_state["last_inputs"] = inputs_tuple
    if "result_df" not in st.session_state:
        st.session_state["result_df"] = None

    if st.session_state["last_inputs"] != inputs_tuple:
        st.session_state["result_df"] = None
        st.session_state["last_inputs"] = inputs_tuple

    if st.button("Dự đoán"):
        df_input = pd.DataFrame([input_vals])
        pred_xgb = xgb_model.predict(df_input)[0]
        pred_lgbm = lgbm_model.predict(df_input)[0]
        st.session_state['result_df'] = pd.DataFrame({
            "Thuật toán": ["XGBoost", "LightGBM"],
            "Kết quả dự đoán": [
                "Lỗi" if pred_xgb == 1 else "Không lỗi",
                "Lỗi" if pred_lgbm == 1 else "Không lỗi"
            ],
            "Mã số dự đoán": [pred_xgb, pred_lgbm]
        })

    # Chỉ hiện bảng khi đã bấm "Dự đoán"
    if st.session_state['result_df'] is not None:
        st.write("### Kết quả dự đoán cho giá trị vừa nhập:")
        st.dataframe(st.session_state['result_df'], use_container_width=True)

