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

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        margin-bottom: 0rem;
        background: linear-gradient(90deg, #38bdf8, #a78bfa, #22c55e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
        letter-spacing: 0.08em;
    }

    .glass-card {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 22px;
        padding: 1.3rem;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(14px);
        margin-bottom: 1rem;
    }

    .tray-card {
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 20px;
        padding: 1.2rem;
        min-height: 180px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
    }

    .tray-id {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.4rem;
    }

    .remark-good {
        color: #22c55e;
        font-weight: 800;
        letter-spacing: 0.08em;
    }

    .remark-warning {
        color: #facc15;
        font-weight: 800;
        letter-spacing: 0.08em;
    }

    .remark-defect {
        color: #ef4444;
        font-weight: 800;
        letter-spacing: 0.08em;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.75rem;
    }

    .metric-value {
        color: #e5e7eb;
        font-size: 1.05rem;
        font-weight: 700;
    }

    .section-heading {
        font-size: 1.4rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 1rem;
        border-radius: 18px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
    }

    .stSelectbox label {
        color: #cbd5e1 !important;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
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
st.markdown(
    """
    <div class="main-title">S.P.A.R.K.S</div>
    <div class="subtitle">INTELLIGENT INSPECTION DASHBOARD · SAMPLING PREDICTION AND RISK KNOWLEDGE SYSTEM</div>
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
# TRAY SELECTION PAGE
# =========================
st.markdown('<div class="section-heading">Tray Selection Overview</div>', unsafe_allow_html=True)

tray_list = sorted(df["tray_id"].dropna().unique())

if len(tray_list) == 0:
    st.warning("No tray data available yet.")
    st.stop()


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


tray_summary_rows = []

for_trays = df.groupby("tray_id")

for tray_id, group in for_trays:
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


# Display tray cards
card_cols = st.columns(3)

for index, row in tray_summary.iterrows():
    remark = str(row["remark"]).upper()

    if remark == "GOOD":
        remark_class = "remark-good"
    elif remark == "WARNING":
        remark_class = "remark-warning"
    else:
        remark_class = "remark-defect"

    with card_cols[index % 3]:
        st.markdown(
            card_html = f"""
            <div class="tray-card">
                <div class="tray-id">{row["tray_id"]}</div>
                <div class="{remark_class}">{remark}</div>

                <div class="metric-label">Total Die</div>
                <div class="metric-value">{row["total_die"]}</div>

                <div class="metric-label">Mean Misalignment</div>
                <div class="metric-value">{row["mean_misalignment_percent"]}%</div>

                <div class="metric-label">Mean Voltage</div>
                <div class="metric-value">{row["mean_voltage"]} V</div>

                <div class="metric-label">Mean Final Score</div>
                <div class="metric-value">{row["mean_final_score"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown('<div class="section-heading">Select Tray to Inspect</div>', unsafe_allow_html=True)

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