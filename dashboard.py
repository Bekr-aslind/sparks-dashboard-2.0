import os
import pandas as pd
import streamlit as st

# =========================
# DASHBOARD SETTINGS
# =========================
st.set_page_config(
    page_title="S.P.A.R.K.S Dashboard",
    layout="wide"
)

LOCAL_CSV = "data/inspection_results.csv"

# Later, when Google Sheets is ready, paste the published CSV link here
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSYv4vtoKDuaULQexNYJE-vKLS8tWYy9BxgiBr2x-4ZOMjO_5QQIlBujUCjfMXHrPoMA2hGD4DR3Daj/pub?output=csv"

REQUIRED_COLUMNS = [
    "tray_id",
    "die_id",
    "image_name",
    "base_detected",
    "die_detected",
    "base_confidence",
    "die_confidence",
    "base_bbox",
    "die_bbox",
    "x_offset_px",
    "y_offset_px",
    "offset_distance_px",
    "offset_percent",
    "overlap_percent",
    "misalignment_percent",
    "visual_status",
    "visual_score",
    "voltage_value",
    "electrical_status",
    "final_die_score",
    "tray_risk",
    "timestamp",
    "image_url",
    "qr_code_url"
]


# =========================
# LOAD DATA
# =========================
@st.cache_data(ttl=10)
def load_data():
    if SHEET_CSV_URL:
        df = pd.read_csv(SHEET_CSV_URL)
    else:
        df = pd.read_csv(LOCAL_CSV)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df


df = load_data()


# =========================
# TITLE
# =========================
st.title("S.P.A.R.K.S")
st.caption("Sampling Prediction and Risk Knowledge System")

if df.empty:
    st.warning("No inspection data available yet.")
    st.info("The dashboard is ready. Data will appear here after the main system records inspection results.")
    st.stop()


# =========================
# CLEAN NUMERIC COLUMNS
# =========================
numeric_columns = [
    "base_confidence",
    "die_confidence",
    "x_offset_px",
    "y_offset_px",
    "offset_distance_px",
    "offset_percent",
    "overlap_percent",
    "misalignment_percent",
    "visual_score",
    "voltage_value",
    "final_die_score"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.header("Filter")

tray_list = sorted(df["tray_id"].dropna().unique())

if len(tray_list) == 0:
    st.warning("No tray data available yet.")
    st.stop()

selected_tray = st.sidebar.selectbox("Select Tray ID", tray_list)

filtered_df = df[df["tray_id"] == selected_tray]


# =========================
# OVERVIEW METRICS
# =========================
total_die = len(filtered_df)
visual_fail = (filtered_df["visual_status"].astype(str).str.upper() == "FAIL").sum()
visual_warning = (filtered_df["visual_status"].astype(str).str.upper() == "WARNING").sum()
electrical_fail = (filtered_df["electrical_status"].astype(str).str.upper() == "FAIL").sum()

average_misalignment = filtered_df["misalignment_percent"].mean()
average_score = filtered_df["final_die_score"].mean()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Tray ID", selected_tray)
col2.metric("Total Die", total_die)
col3.metric("Visual WARNING", visual_warning)
col4.metric("Visual FAIL", visual_fail)
col5.metric("Electrical FAIL", electrical_fail)

col6, col7 = st.columns(2)

if pd.isna(average_misalignment):
    col6.metric("Average Misalignment", "N/A")
else:
    col6.metric("Average Misalignment", f"{average_misalignment:.2f}%")

if pd.isna(average_score):
    col7.metric("Average Final Die Score", "N/A")
else:
    col7.metric("Average Final Die Score", f"{average_score:.1f}")


# =========================
# TRAY RISK
# =========================
risk_values = filtered_df["tray_risk"].astype(str).str.upper().tolist()

if "HIGH" in risk_values:
    tray_risk = "HIGH"
elif "MEDIUM" in risk_values:
    tray_risk = "MEDIUM"
else:
    tray_risk = "LOW"

if tray_risk == "HIGH":
    st.error(f"Tray Risk Level: {tray_risk}")
elif tray_risk == "MEDIUM":
    st.warning(f"Tray Risk Level: {tray_risk}")
else:
    st.success(f"Tray Risk Level: {tray_risk}")


# =========================
# INSPECTION DATA TABLE
# =========================
st.subheader("Inspection Data")

display_columns = [
    "tray_id",
    "die_id",
    "base_detected",
    "die_detected",
    "base_confidence",
    "die_confidence",
    "x_offset_px",
    "y_offset_px",
    "offset_distance_px",
    "offset_percent",
    "overlap_percent",
    "misalignment_percent",
    "visual_status",
    "visual_score",
    "voltage_value",
    "electrical_status",
    "final_die_score",
    "tray_risk",
    "timestamp"
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True
)


# =========================
# INSPECTION PERFORMANCE SUMMARY
# =========================
st.subheader("Inspection Performance Summary")

visual_status_upper = filtered_df["visual_status"].astype(str).str.upper()
electrical_status_upper = filtered_df["electrical_status"].astype(str).str.upper()

visual_pass_count = (visual_status_upper == "PASS").sum()
electrical_pass_count = (electrical_status_upper == "PASS").sum()

visual_pass_rate = (visual_pass_count / total_die * 100) if total_die > 0 else 0
electrical_pass_rate = (electrical_pass_count / total_die * 100) if total_die > 0 else 0

average_overlap = filtered_df["overlap_percent"].mean()
average_visual_score = filtered_df["visual_score"].mean()

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

summary_col1.metric("Visual Pass Rate", f"{visual_pass_rate:.1f}%")
summary_col2.metric("Electrical Pass Rate", f"{electrical_pass_rate:.1f}%")

if pd.isna(average_misalignment):
    summary_col3.metric("Average Misalignment", "N/A")
else:
    summary_col3.metric("Average Misalignment", f"{average_misalignment:.2f}%")

if pd.isna(average_overlap):
    summary_col4.metric("Average Overlap", "N/A")
else:
    summary_col4.metric("Average Overlap", f"{average_overlap:.2f}%")


# =========================
# ALIGNMENT TREND
# =========================
st.markdown("### Alignment Trend by Die")

alignment_trend_df = filtered_df[
    [
        "die_id",
        "misalignment_percent",
        "overlap_percent",
        "offset_percent"
    ]
].copy()

alignment_trend_df = alignment_trend_df.dropna(
    subset=["misalignment_percent", "overlap_percent", "offset_percent"],
    how="all"
)

if not alignment_trend_df.empty:
    alignment_trend_df = alignment_trend_df.set_index("die_id")
    st.line_chart(alignment_trend_df)
else:
    st.info("No alignment trend data available yet.")


# =========================
# SCORE TREND
# =========================
st.markdown("### Score Trend by Die")

score_trend_df = filtered_df[
    [
        "die_id",
        "visual_score",
        "final_die_score"
    ]
].copy()

score_trend_df = score_trend_df.dropna(
    subset=["visual_score", "final_die_score"],
    how="all"
)

if not score_trend_df.empty:
    score_trend_df = score_trend_df.set_index("die_id")
    st.line_chart(score_trend_df)
else:
    st.info("No score trend data available yet.")

# =========================
# DIE DETAIL VIEW
# =========================
st.subheader("Individual Die Detail")

die_list = filtered_df["die_id"].dropna().unique()

if len(die_list) == 0:
    st.info("No die data available yet.")
    st.stop()

selected_die = st.selectbox("Select Die ID", die_list)

die_row = filtered_df[filtered_df["die_id"] == selected_die].iloc[0]

detail_col1, detail_col2, detail_col3 = st.columns(3)

detail_col1.metric("Visual Status", str(die_row["visual_status"]))
detail_col2.metric("Electrical Status", str(die_row["electrical_status"]))

final_score = die_row["final_die_score"]
if pd.isna(final_score):
    detail_col3.metric("Final Die Score", "N/A")
else:
    detail_col3.metric("Final Die Score", f"{final_score:.1f}")


st.markdown("### Alignment Measurement")

align_col1, align_col2, align_col3, align_col4 = st.columns(4)

align_col1.metric("X Offset", f"{die_row['x_offset_px']:.2f} px" if pd.notna(die_row["x_offset_px"]) else "N/A")
align_col2.metric("Y Offset", f"{die_row['y_offset_px']:.2f} px" if pd.notna(die_row["y_offset_px"]) else "N/A")
align_col3.metric("Offset Distance", f"{die_row['offset_distance_px']:.2f} px" if pd.notna(die_row["offset_distance_px"]) else "N/A")
align_col4.metric("Misalignment", f"{die_row['misalignment_percent']:.2f}%" if pd.notna(die_row["misalignment_percent"]) else "N/A")


st.markdown("### Detection Details")

detect_col1, detect_col2 = st.columns(2)

with detect_col1:
    st.write("**Base / Reference Square**")
    st.write("Detected:", die_row["base_detected"])
    st.write("Confidence:", die_row["base_confidence"])
    st.write("Bounding Box:", die_row["base_bbox"])

with detect_col2:
    st.write("**Die / Top Square**")
    st.write("Detected:", die_row["die_detected"])
    st.write("Confidence:", die_row["die_confidence"])
    st.write("Bounding Box:", die_row["die_bbox"])


# =========================
# IMAGE AND QR DISPLAY
# =========================
st.subheader("Processed Image and QR Access")

image_url = str(die_row.get("image_url", "")).strip()
qr_code_url = str(die_row.get("qr_code_url", "")).strip()

col_img, col_qr = st.columns(2)

with col_img:
    st.markdown("### Processed Die Image")

    image_name = str(die_row.get("image_name", "")).strip()
    local_processed_path = os.path.join("processed", image_name)

    if os.path.exists(local_processed_path):
        st.image(
            local_processed_path,
            caption=f"Processed image for {selected_die}",
            use_container_width=True
        )

    elif image_url and image_url.lower() != "nan":
        st.image(
            image_url,
            caption=f"Processed image for {selected_die}",
            use_container_width=True
        )
        st.markdown(f"[Open image in new tab]({image_url})")

    else:
        st.info("No image link available yet.")
        
with col_qr:
    st.markdown("### QR / Tray Access")
    if qr_code_url and qr_code_url.lower() != "nan":
        st.markdown(f"[Open Tray Inspection Record]({qr_code_url})")
    else:
        st.info("No QR code link available yet.")