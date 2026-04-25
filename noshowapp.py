import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="No-Show Predictor", page_icon="🏥", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0a0f1e; color: #e2e8f0; }

section[data-testid="stSidebar"] {
    background: #0d1426 !important;
    border-right: 1px solid #1e2d4a;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] .stRadio label { 
    color: #e2e8f0 !important; 
    font-size: 15px;
    padding: 8px 0;
}

.hero {
    background: linear-gradient(135deg, #0d1f3c 0%, #1a3a6b 50%, #0d2d5e 100%);
    border: 1px solid #1e3a6e;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '🏥';
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.15;
}
.hero h1 { color: #ffffff; font-size: 2rem; font-weight: 700; margin: 0 0 0.3rem 0; }
.hero p { color: #7fa8d4; margin: 0; font-size: 0.95rem; }

.card {
    background: #0d1426;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card h3 { color: #60a5fa; font-size: 0.85rem; font-weight: 600; 
           letter-spacing: 0.1em; text-transform: uppercase; margin: 0 0 1rem 0; }

.risk-high {
    background: linear-gradient(135deg, #2d0a0a, #3d1010);
    border: 2px solid #ef4444;
    border-radius: 16px; padding: 2rem; text-align: center; margin: 1.5rem 0;
}
.risk-high h1 { color: #ef4444; font-size: 2.5rem; margin: 0 0 0.5rem 0; }
.risk-high h2 { color: #ffffff; font-size: 1.8rem; margin: 0 0 0.5rem 0; }
.risk-high p { color: #fca5a5; margin: 0.3rem 0; }

.risk-medium {
    background: linear-gradient(135deg, #2d1f0a, #3d2d0d);
    border: 2px solid #f59e0b;
    border-radius: 16px; padding: 2rem; text-align: center; margin: 1.5rem 0;
}
.risk-medium h1 { color: #f59e0b; font-size: 2.5rem; margin: 0 0 0.5rem 0; }
.risk-medium h2 { color: #ffffff; font-size: 1.8rem; margin: 0 0 0.5rem 0; }
.risk-medium p { color: #fcd34d; margin: 0.3rem 0; }

.risk-low {
    background: linear-gradient(135deg, #0a2d0a, #0d3d1a);
    border: 2px solid #22c55e;
    border-radius: 16px; padding: 2rem; text-align: center; margin: 1.5rem 0;
}
.risk-low h1 { color: #22c55e; font-size: 2.5rem; margin: 0 0 0.5rem 0; }
.risk-low h2 { color: #ffffff; font-size: 1.8rem; margin: 0 0 0.5rem 0; }
.risk-low p { color: #86efac; margin: 0.3rem 0; }

.metric-box {
    background: #0d1426;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-box .label { color: #64748b; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-box .value { color: #60a5fa; font-size: 1.6rem; font-weight: 700; font-family: 'DM Mono', monospace; }

.feat-bar-wrap { margin: 0.5rem 0; }
.feat-label { display: flex; justify-content: space-between; margin-bottom: 4px; }
.feat-name { color: #cbd5e1; font-size: 0.88rem; }
.feat-pct { color: #60a5fa; font-size: 0.88rem; font-family: 'DM Mono', monospace; }
.feat-bar-bg { background: #1e2d4a; border-radius: 999px; height: 8px; }
.feat-bar-fill { height: 8px; border-radius: 999px; }

.model-winner {
    background: linear-gradient(135deg, #0d2d1a, #0a3d20);
    border: 2px solid #22c55e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
}
.model-loser {
    background: #0d1426;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    opacity: 0.7;
}

.built-by {
    text-align: center;
    padding: 2rem;
    margin-top: 2rem;
    border-top: 1px solid #1e2d4a;
    color: #475569;
    font-size: 0.88rem;
}
.built-by span { color: #60a5fa; font-weight: 600; }

div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
}

label { color: #94a3b8 !important; font-size: 0.88rem !important; }
hr { border-color: #1e2d4a !important; }
h2, h3 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 No-Show Predictor")
    st.markdown("---")
    page = st.radio("", ["🔮  Predict No-Show", "📊  Insights & Model"])
    st.markdown("---")
    st.markdown("""
    <div style='color:#475569; font-size:0.8rem; padding: 0.5rem 0;'>
    Dataset: CER Brazil<br>
    Records: 49,593<br>
    Period: 2016–2022<br>
    No-show rate: 9.7%
    </div>
    """, unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('noshow_model.pkl')

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if "Predict" in page:

    st.markdown("""
    <div class="hero">
        <h1>Medical No-Show Predictor</h1>
        <p>CER Rehabilitation Center · Southern Brazil · 2016–2022</p>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error("❌ `noshow_model.pkl` not found. Run your training script first.")
        st.code("import joblib\njoblib.dump(final_best_model, 'noshow_model.pkl')")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card"><h3>👤 Patient Info</h3>', unsafe_allow_html=True)
        age = st.number_input("Age", 2, 110, 35)
        gender = st.selectbox("Gender", ["F", "M", "I"])
        disability = st.selectbox("Disability Type", [" ", "motor", "intellectual"])
        patient_needs_companion = st.selectbox("Needs Companion?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><h3>📅 Appointment Info</h3>', unsafe_allow_html=True)
        specialty = st.selectbox("Specialty", [
            "physiotherapy", "psychotherapy", "speech therapy",
            "occupational therapy", "pedagogo", "enf", "assist", "sem especialidade"
        ])
        appointment_shift = st.selectbox("Shift", ["morning", "afternoon"])
        appointment_hour = st.slider("Hour", 6, 18, 9)
        waiting_days = st.number_input("Waiting Days", 0, 365, 7)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card"><h3>🌦️ Weather Conditions</h3>', unsafe_allow_html=True)
        average_temp_day = st.slider("Avg Temperature (°C)", 8.0, 30.0, 22.0)
        max_temp_day = st.slider("Max Temperature (°C)", 13.0, 36.0, 28.0)
        average_rain_day = st.slider("Avg Rain (mm)", 0.0, 5.0, 0.0)
        max_rain_day = st.slider("Max Rain (mm)", 0.0, 45.0, 0.0)
        rain_intensity = st.selectbox("Rain Intensity", ["no_rain", "weak", "moderate", "heavy"])
        heat_intensity = st.selectbox("Heat Intensity", ["cold", "heavy_cold", "mild", "warm", "heavy_warm"])
        storm_day_before = st.selectbox("Storm Yesterday?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        rainy_day_before = st.selectbox("Rain Yesterday?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮  Predict No-Show Risk", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame([{
            'age': age, 'patient_needs_companion': patient_needs_companion,
            'waiting_days': waiting_days, 'appointment_hour': appointment_hour,
            'average_temp_day': average_temp_day, 'average_rain_day': average_rain_day,
            'storm_day_before': storm_day_before, 'max_temp_day': max_temp_day,
            'max_rain_day': max_rain_day, 'rainy_day_before': rainy_day_before,
            'gender': gender, 'disability': disability, 'specialty': specialty,
            'appointment_shift': appointment_shift,
            'rain_intensity': rain_intensity, 'heat_intensity': heat_intensity,
        }])

        try:
            proba = model.predict_proba(input_data)[0][1]
            THRESHOLD = 0.4

            if proba >= 0.5:
                st.markdown(f"""<div class="risk-high">
                    <h1>🔴 HIGH RISK</h1>
                    <h2>No-Show Probability: {proba:.1%}</h2>
                    <p>This patient is very likely to miss their appointment.</p>
                    <p><b>Action:</b> Call patient + send SMS reminder immediately</p>
                </div>""", unsafe_allow_html=True)
            elif proba >= THRESHOLD:
                st.markdown(f"""<div class="risk-medium">
                    <h1>🟡 MEDIUM RISK</h1>
                    <h2>No-Show Probability: {proba:.1%}</h2>
                    <p>This patient may miss their appointment.</p>
                    <p><b>Action:</b> Send SMS reminder 24 hours before</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="risk-low">
                    <h1>🟢 LOW RISK</h1>
                    <h2>No-Show Probability: {proba:.1%}</h2>
                    <p>This patient is likely to attend their appointment.</p>
                    <p><b>Action:</b> No intervention needed</p>
                </div>""", unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-box"><div class="label">No-Show Prob</div><div class="value">{proba:.1%}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-box"><div class="label">Risk Level</div><div class="value">{"HIGH" if proba>=0.5 else "MED" if proba>=0.4 else "LOW"}</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown('<div class="metric-box"><div class="label">Threshold</div><div class="value">0.40</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown('<div class="metric-box"><div class="label">Model</div><div class="value">XGBoost</div></div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

    st.markdown('<div class="built-by">Built by <span>Aditya</span> · B.Tech CSE (AI/ML) · Amity University Mohali</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — INSIGHTS & MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif "Insights" in page:

    st.markdown("""
    <div class="hero">
        <h1>Insights & Model Info</h1>
        <p>What drives no-shows · How the model works · Who won</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature Importance
    st.subheader("🔍 What Predicts a No-Show?")
    st.markdown("<p style='color:#64748b; margin-top:-0.5rem; margin-bottom:1rem;'>Top features ranked by importance from XGBoost model</p>", unsafe_allow_html=True)

    features = [
        ("Gender (Male)",         6.65, "#ef4444"),
        ("Max Rain Day",          5.87, "#3b82f6"),
        ("Heat Intensity (Warm)", 5.81, "#f97316"),
        ("Disability (Motor)",    4.82, "#a855f7"),
        ("Heat Intensity (Mild)", 4.78, "#f97316"),
        ("Rain Intensity (Weak)", 4.47, "#3b82f6"),
        ("Average Rain Day",      4.36, "#3b82f6"),
        ("Max Temperature",       4.35, "#f97316"),
        ("Waiting Days",          4.34, "#22c55e"),
        ("Appointment Hour",      4.16, "#22c55e"),
        ("Age",                   3.26, "#60a5fa"),
        ("Needs Companion",       3.21, "#a855f7"),
    ]

    col1, col2 = st.columns([3, 2])

    with col1:
        for name, pct, color in features:
            bar_width = int((pct / 7.0) * 100)
            st.markdown(f"""
            <div class="feat-bar-wrap">
                <div class="feat-label">
                    <span class="feat-name">{name}</span>
                    <span class="feat-pct">{pct:.2f}%</span>
                </div>
                <div class="feat-bar-bg">
                    <div class="feat-bar-fill" style="width:{bar_width}%; background:{color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h3>🔑 Key Findings</h3>
            <p style='color:#94a3b8; font-size:0.9rem; line-height:1.8;'>
            🌧️ <b style='color:#3b82f6;'>Weather is top predictor</b> — Rain and heat appear in 5 of top 10 features.<br><br>
            👨 <b style='color:#ef4444;'>Males miss more</b> — Gender (Male) is the single strongest predictor at 6.65%.<br><br>
            ♿ <b style='color:#a855f7;'>Motor disability</b> patients have higher no-show rates.<br><br>
            ⏳ <b style='color:#22c55e;'>Longer wait = more no-shows</b> — waiting days is a key predictor.<br><br>
            🕐 <b style='color:#22c55e;'>Appointment hour matters</b> — late slots show worse attendance.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🏆 Model Comparison — Who Won?")

    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("""
        <div class="model-winner">
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <span style='color:#22c55e; font-size:0.8rem; font-weight:700; letter-spacing:0.1em;'>🏆 WINNER · DEPLOYED</span>
                    <h3 style='color:#ffffff; margin:0.3rem 0; font-size:1.3rem;'>XGBoost</h3>
                </div>
                <span style='font-size:2.5rem;'>⚡</span>
            </div>
            <hr style='border-color:#1a4d2e; margin:0.8rem 0;'>
            <table style='width:100%; font-size:0.88rem;'>
                <tr><td style='color:#64748b; padding:3px 0;'>Accuracy</td><td style='color:#e2e8f0; text-align:right; font-family:monospace;'>81.93%</td></tr>
                <tr><td style='color:#64748b; padding:3px 0;'>F1 Score</td><td style='color:#22c55e; text-align:right; font-family:monospace;'>33.64%</td></tr>
                <tr><td style='color:#64748b; padding:3px 0;'>Precision</td><td style='color:#e2e8f0; text-align:right; font-family:monospace;'>25.00%</td></tr>
                <tr><td style='color:#22c55e; padding:3px 0;'>Recall ✓</td><td style='color:#22c55e; text-align:right; font-family:monospace;'>51.43%</td></tr>
                <tr><td style='color:#64748b; padding:3px 0;'>Threshold</td><td style='color:#e2e8f0; text-align:right; font-family:monospace;'>0.40</td></tr>
            </table>
            <p style='color:#86efac; font-size:0.82rem; margin:0.8rem 0 0 0;'>
            ✓ Higher recall — catches more no-shows<br>
            ✓ scale_pos_weight handles 1:9 class imbalance<br>
            ✓ GridSearchCV + StratifiedKFold tuning
            </p>
        </div>
        """, unsafe_allow_html=True)

    with mc2:
        st.markdown("""
        <div class="model-loser">
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <span style='color:#64748b; font-size:0.8rem; font-weight:700; letter-spacing:0.1em;'>RUNNER UP</span>
                    <h3 style='color:#94a3b8; margin:0.3rem 0; font-size:1.3rem;'>Random Forest</h3>
                </div>
                <span style='font-size:2.5rem; opacity:0.4;'>🌲</span>
            </div>
            <hr style='border-color:#1e2d4a; margin:0.8rem 0;'>
            <table style='width:100%; font-size:0.88rem;'>
                <tr><td style='color:#64748b; padding:3px 0;'>Accuracy</td><td style='color:#94a3b8; text-align:right; font-family:monospace;'>89.43%</td></tr>
                <tr><td style='color:#64748b; padding:3px 0;'>F1 Score</td><td style='color:#94a3b8; text-align:right; font-family:monospace;'>30.96%</td></tr>
                <tr><td style='color:#64748b; padding:3px 0;'>Precision</td><td style='color:#94a3b8; text-align:right; font-family:monospace;'>37.37%</td></tr>
                <tr><td style='color:#ef4444; padding:3px 0;'>Recall ✗</td><td style='color:#ef4444; text-align:right; font-family:monospace;'>26.43%</td></tr>
                <tr><td style='color:#64748b; padding:3px 0;'>Threshold</td><td style='color:#94a3b8; text-align:right; font-family:monospace;'>0.40</td></tr>
            </table>
            <p style='color:#475569; font-size:0.82rem; margin:0.8rem 0 0 0;'>
            ✗ High accuracy is misleading — predicts<br>
            &nbsp;&nbsp; majority class too often<br>
            ✗ Only catches 26% of actual no-shows
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📁 About the Dataset")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("""
        <div class="card">
            <h3>📌 Source</h3>
            <p style='color:#94a3b8; font-size:0.9rem; line-height:1.9;'>
            <b style='color:#e2e8f0;'>CER Rehabilitation Center</b><br>
            University of Vale do Itajaí<br>
            Southern Brazil<br><br>
            <b style='color:#e2e8f0;'>DOI:</b> 10.17632/wm6w2fvkfj.1<br>
            <b style='color:#e2e8f0;'>Published:</b> February 2023<br>
            <b style='color:#e2e8f0;'>Version:</b> 1
            </p>
        </div>
        """, unsafe_allow_html=True)

    with d2:
        st.markdown("""
        <div class="card">
            <h3>📊 Stats</h3>
            <p style='color:#94a3b8; font-size:0.9rem; line-height:1.9;'>
            <b style='color:#60a5fa;'>49,593</b> total appointments<br>
            <b style='color:#ef4444;'>4,832</b> no-shows (9.7%)<br>
            <b style='color:#22c55e;'>44,761</b> attended (90.3%)<br><br>
            <b style='color:#e2e8f0;'>Period:</b> 2016–2022<br>
            <b style='color:#e2e8f0;'>Specialties:</b> 8 types<br>
            <b style='color:#e2e8f0;'>Features:</b> 26 columns
            </p>
        </div>
        """, unsafe_allow_html=True)

    with d3:
        st.markdown("""
        <div class="card">
            <h3>⚙️ ML Pipeline</h3>
            <p style='color:#94a3b8; font-size:0.9rem; line-height:1.9;'>
            <b style='color:#e2e8f0;'>Preprocessor:</b> ColumnTransformer<br>
            <b style='color:#e2e8f0;'>Imputer:</b> Median + Most Frequent<br>
            <b style='color:#e2e8f0;'>Encoder:</b> OneHotEncoder<br>
            <b style='color:#e2e8f0;'>CV:</b> StratifiedKFold (5-fold)<br>
            <b style='color:#e2e8f0;'>Tuner:</b> GridSearchCV<br>
            <b style='color:#e2e8f0;'>Imbalance:</b> scale_pos_weight 1:9
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="built-by">
        Built by <span>Aditya</span> · B.Tech CSE (AI/ML) · Amity University Mohali<br>
        <span style='color:#334155; font-size:0.8rem;'>Dataset: DOI 10.17632/wm6w2fvkfj.1 · CER Brazil · 2016–2022</span>
    </div>
    """, unsafe_allow_html=True)