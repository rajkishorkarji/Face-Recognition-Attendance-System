import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from streamlit_calendar import calendar

# --- Helper Functions ---
def get_name_column(df):
    if df.empty: return None
    possible_names = ['name', 'student', 'student name', 'names']
    for col in df.columns:
        if col.strip().lower() in possible_names:
            return col
    return df.columns[0]

@st.cache_data
def load_attendance():
    path = "Attendance"
    os.makedirs(path, exist_ok=True)
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    if not files: return pd.DataFrame(), 0

    dfs = []
    for f in files:
        try:
            date_str = os.path.basename(f).split('_')[1].replace('.csv', '')
            df = pd.read_csv(f)
            df.rename(columns=lambda c: c.strip().title(), inplace=True)
            df['Date'] = pd.to_datetime(date_str, format="%d-%m-%Y").date()
            dfs.append(df)
        except: continue
    if not dfs: return pd.DataFrame(), 0
    master = pd.concat(dfs, ignore_index=True)
    return master, len(master['Date'].unique())

def convert_df(df): return df.to_csv(index=False).encode('utf-8')

def get_today_df(name_col):
    file = f"Attendance/Attendance_{datetime.now().strftime('%d-%m-%Y')}.csv"
    if os.path.exists(file):
        try: df = pd.read_csv(file); df.rename(columns=lambda c: c.strip().title(), inplace=True)
        except: df = pd.DataFrame()
    else: df = pd.DataFrame()
    return df

# --- Page Config ---
st.set_page_config("🎓 Attendance Dashboard", layout="wide")
st_autorefresh(interval=5000, key="refresh")

# --- Load Data ---
attendance_df, total_days = load_attendance()
if attendance_df.empty:
    st.warning("No attendance data found in 'Attendance' folder.")
    st.stop()

name_col = st.sidebar.selectbox(
    "Choose Columns (Name / Date)",
    options=[c for c in attendance_df.columns if c != 'Date'],
    index=[c for c in attendance_df.columns if c != 'Date'].index(get_name_column(attendance_df))
)

# --- Tabs for Navigation ---
tab1, tab2, tab3 = st.tabs(["Dashboard", "Student Report", "Insights"])

# --- DASHBOARD ---
with tab1:
    st.header("📊 Today's Attendance")
    all_students = sorted(attendance_df[name_col].unique())
    today_df = get_today_df(name_col)
    present = sorted(today_df[name_col].unique()) if not today_df.empty else []
    absent = sorted(list(set(all_students) - set(present)))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Students", len(all_students))
        st.metric("✅ Present Today", len(present))
        if not today_df.empty: st.dataframe(today_df, use_container_width=True)
    with col2:
        st.metric("❌ Absent Today", len(absent))
    if absent:
        st.dataframe(pd.DataFrame(absent, columns=[name_col]))
    else:
        st.info("All students are present today!")  

    st.markdown("---")
    st.subheader("📅 Filter Attendance Records")
    min_d, max_d = attendance_df['Date'].min(), attendance_df['Date'].max()
    start, end = st.date_input("Date Range", (min_d, max_d), min_value=min_d, max_value=max_d)
    filtered = attendance_df[(attendance_df['Date'] >= start) & (attendance_df['Date'] <= end)]
    st.dataframe(filtered.sort_values(['Date', name_col], ascending=[False, True]), use_container_width=True)
    st.download_button("📥 Download CSV", convert_df(filtered), f"attendance_{start}_{end}.csv", "text/csv")

# --- STUDENT REPORT ---
with tab2:
    st.header("👤 Student Report")
    student = st.selectbox("Select Student", sorted(attendance_df[name_col].unique()))
    student_df = attendance_df[attendance_df[name_col] == student]
    days_present = len(student_df)
    percentage = (days_present / total_days * 100) if total_days else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Days Present", days_present)
    c2.metric("Total Class Days", total_days)
    c3.metric("Attendance %", f"{percentage:.2f}%")

    events = [{"title":"Present","start":d.isoformat(),"end":d.isoformat(),"color":"green"} for d in student_df['Date']]
    calendar(events=events, options={"initialView":"dayGridMonth"}, key="calendar")

# --- INSIGHTS ---
with tab3:
    st.header("📈 Attendance Insights")
    avg_att = attendance_df.groupby('Date')[name_col].count().mean()
    c1, c2 = st.columns(2)
    c1.metric("Total Class Days", total_days)
    c2.metric("Average Daily Attendance", f"{avg_att:.2f}")

    st.subheader("🏆 Attendance Leaderboard")
    leaderboard = (attendance_df[name_col].value_counts() / total_days * 100).reset_index()
    leaderboard.columns = ["Student","Attendance (%)"]
    st.dataframe(leaderboard.sort_values("Attendance (%)", ascending=False).style.format({"Attendance (%)":"{:.2f}%"}))

    st.subheader("📅 Daily Attendance Trend")
    daily = attendance_df.groupby('Date')[name_col].count().reset_index().rename(columns={name_col:"Present"})
    
    fig = px.line(daily, x='Date', y='Present', title="Daily Attendance Trend", markers=True)
    
    # --- FIX: Format the x-axis to show only the date (Year-Month-Day) ---
    fig.update_xaxes(
        dtick="D1",             # Set the tick interval to 1 day
        tickformat="%Y-%m-%d"   # Format the ticks to display Year-Month-Day
    )
    # ---------------------------------------------------------------------
    
    st.plotly_chart(fig, use_container_width=True)
    # cd OneDrive\Desktop\FaceRec_Project
    # python -m streamlit run attendance_gui.py