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

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #111827 45%, #0f172a 100%);
        color: #e5e7eb;
    }

    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }

    .block-container {
        padding-top: 4rem;
        padding-bottom: 4rem;
    }

    .main-title {
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: 0.1em;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #22c55e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        color: #93c5fd;
        font-size: 0.95rem;
        margin-bottom: 2.4rem;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }

    .section-heading {
        font-size: 1.45rem;
        font-weight: 850;
        color: #f8fafc;
        margin-top: 1.8rem;
        margin-bottom: 1rem;
    }

    .tray-card {
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.26);
        border-radius: 22px;
        padding: 1.25rem;
        min-height: 245px;
        box-shadow: 0 18px 42px rgba(0, 0, 0, 0.28);
        margin-bottom: 1rem;
        backdrop-filter: blur(14px);
    }

    .tray-id {
        font-size: 1.45rem;
        font-weight: 900;
        color: #f8fafc;
        margin-bottom: 0.45rem;
    }

    .remark-good {
        color: #22c55e;
        font-weight: 900;
        letter-spacing: 0.12em;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }

    .remark-warning {
        color: #facc15;
        font-weight: 900;
        letter-spacing: 0.12em;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }

    .remark-defect {
        color: #ef4444;
        font-weight: 900;
        letter-spacing: 0.12em;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }

    .metric-label-custom {
        color: #94a3b8;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.85rem;
    }

    .metric-value-custom {
        color: #e5e7eb;
        font-size: 1.08rem;
        font-weight: 800;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.76);
        border: 1px solid rgba(148, 163, 184, 0.20);
        padding: 1rem;
        border-radius: 18px;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
    }

    .stSelectbox label {
        color: #cbd5e1 !important;
        font-weight: 700;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    p, span, label {
        color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# DATA SETTINGS
# =========================
LOCAL_CSV = "data/inspection_results.csv"

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
# HELPER FUNCTIONS
# =========================
def clean_value(value, unit="", decimals=2):
    if pd.isna(value):
        return "N/A"

    try:
        value = float(value)
        return f"{value:.{decimals}f}{unit}"
    except Exception:
        return str(value)


def get_tray_remark(group):
    visual_statuses = group["visual_status"].astype(str).str.upper().tolist()
    electrical_statuses = group["electrical_status"].astype(str).str.upper().tolist()
    risk_values = group["tray_risk"].astype(str).str.upper().tolist()

    if (
        "FAIL" in visual_statuses
        or "DEFECT" in visual_statuses
        or "FAIL" in electrical_statuses
        or "DEFECT" in electrical_statuses
        or "HIGH" in risk_values
    ):
        return "DEFECT"

    if (
        "WARNING" in visual_statuses
        or "WARNING" in electrical_statuses
        or "MEDIUM" in risk_values
    ):
        return "WARNING"

    return "GOOD"


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
st.markdown(
    """
    <div class="main-title">S.P.A.R.K.S</div>
    <div class="subtitle">Intelligent Inspection Dashboard · Sampling Prediction and Risk Knowledge System</div>
    """,
    unsafe_allow_html=True
)

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
# TRAY SELECTION OVERVIEW
# =========================
st.markdown(
    '<div class="section-heading">Tray Selection Overview</div>',
    unsafe_allow_html=True
)

tray_list = sorted(df["tray_id"].dropna().unique())

if len(tray_list) == 0:
    st.warning("No tray data available yet.")
    st.stop()

tray_summary_rows = []

for tray_id, group in df.groupby("tray_id"):
    tray_summary_rows.append({
        "tray_id": tray_id,
        "remark": get_tray_remark(group),
        "total_die": len(group),
        "mean_misalignment_percent": group["misalignment_percent"].mean(),
        "mean_voltage": group["voltage_value"].mean(),
        "mean_final_score": group["final_die_score"].mean()
    })

tray_summary = pd.DataFrame(tray_summary_rows)

tray_summary["mean_misalignment_percent"] = tray_summary["mean_misalignment_percent"].round(2)
tray_summary["mean_voltage"] = tray_summary["mean_voltage"].round(3)
tray_summary["mean_final_score"] = tray_summary["mean_final_score"].round(2)


# =========================
# DISPLAY TRAY CARDS
# =========================
card_cols = st.columns(3)

for index, row in tray_summary.iterrows():
    remark = str(row["remark"]).upper()

    if remark == "GOOD":
        remark_class = "remark-good"
    elif remark == "WARNING":
        remark_class = "remark-warning"
    else:
        remark_class = "remark-defect"

    card_html = (
        f'<div class="tray-card">'
        f'<div class="tray-id">{row["tray_id"]}</div>'
        f'<div class="{remark_class}">{remark}</div>'
        f'<div class="metric-label-custom">Total Die</div>'
        f'<div class="metric-value-custom">{row["total_die"]}</div>'
        f'<div class="metric-label-custom">Mean Misalignment</div>'
        f'<div class="metric-value-custom">{clean_value(row["mean_misalignment_percent"], "%", 2)}</div>'
        f'<div class="metric-label-custom">Mean Voltage</div>'
        f'<div class="metric-value-custom">{clean_value(row["mean_voltage"], " V", 3)}</div>'
        f'<div class="metric-label-custom">Mean Final Score</div>'
        f'<div class="metric-value-custom">{clean_value(row["mean_final_score"], "", 2)}</div>'
        f'</div>'
    )

    with card_cols[index % 3]:
        st.markdown(card_html, unsafe_allow_html=True)


# =========================
# SELECT TRAY TO INSPECT
# =========================
st.markdown(
    '<div class="section-heading">Select Tray to Inspect</div>',
    unsafe_allow_html=True
)

selected_tray = st.selectbox(
    "Choose a tray ID",
    tray_summary["tray_id"].tolist()
)

filtered_df = df[df["tray_id"] == selected_tray]

selected_tray_remark = tray_summary[
    tray_summary["tray_id"] == selected_tray
]["remark"].iloc[0]

if selected_tray_remark == "DEFECT":
    st.error(f"Selected Tray: {selected_tray} | Remark: DEFECT")
elif selected_tray_remark == "WARNING":
    st.warning(f"Selected Tray: {selected_tray} | Remark: WARNING")
else:
    st.success(f"Selected Tray: {selected_tray} | Remark: GOOD")


# =========================
# OVERVIEW METRICS
# =========================
total_die = len(filtered_df)

visual_status_upper = filtered_df["visual_status"].astype(str).str.upper()
electrical_status_upper = filtered_df["electrical_status"].astype(str).str.upper()

visual_fail = visual_status_upper.isin(["FAIL", "DEFECT"]).sum()
visual_warning = (visual_status_upper == "WARNING").sum()
electrical_fail = electrical_status_upper.isin(["FAIL", "DEFECT"]).sum()

average_misalignment = filtered_df["misalignment_percent"].mean()
average_score = filtered_df["final_die_score"].mean()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Tray ID", selected_tray)
col2.metric("Total Die", total_die)
col3.metric("Visual WARNING", visual_warning)
col4.metric("Visual FAIL", visual_fail)
col5.metric("Electrical FAIL", electrical_fail)

col6, col7 = st.columns(2)

col6.metric(
    "Average Misalignment",
    clean_value(average_misalignment, "%", 2)
)

col7.metric(
    "Average Final Die Score",
    clean_value(average_score, "", 1)
)


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

visual_pass_count = (visual_status_upper == "PASS").sum()
electrical_pass_count = (electrical_status_upper == "PASS").sum()

visual_pass_rate = (visual_pass_count / total_die * 100) if total_die > 0 else 0
electrical_pass_rate = (electrical_pass_count / total_die * 100) if total_die > 0 else 0

average_overlap = filtered_df["overlap_percent"].mean()

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

summary_col1.metric("Visual Pass Rate", f"{visual_pass_rate:.1f}%")
summary_col2.metric("Electrical Pass Rate", f"{electrical_pass_rate:.1f}%")
summary_col3.metric("Average Misalignment", clean_value(average_misalignment, "%", 2))
summary_col4.metric("Average Overlap", clean_value(average_overlap, "%", 2))


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
detail_col3.metric(
    "Final Die Score",
    clean_value(die_row["final_die_score"], "", 1)
)


# =========================
# ALIGNMENT MEASUREMENT
# =========================
st.markdown("### Alignment Measurement")

align_col1, align_col2, align_col3, align_col4 = st.columns(4)

align_col1.metric(
    "X Offset",
    clean_value(die_row["x_offset_px"], " px", 2)
)

align_col2.metric(
    "Y Offset",
    clean_value(die_row["y_offset_px"], " px", 2)
)

align_col3.metric(
    "Offset Distance",
    clean_value(die_row["offset_distance_px"], " px", 2)
)

align_col4.metric(
    "Misalignment",
    clean_value(die_row["misalignment_percent"], "%", 2)
)


# =========================
# DETECTION DETAILS
# =========================
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
# PROCESSED IMAGE DISPLAY
# =========================
st.subheader("Processed Die Image")

image_url = str(die_row.get("image_url", "")).strip()
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