import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ==================== ตั้งค่าหน้าเว็บ ====================
st.set_page_config(
    page_title="❤️ Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    /* พื้นหลังหลัก */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* หัวข้อหลัก */
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        font-size: 1.2rem;
        text-align: center;
        color: #f0f0f0;
        margin-bottom: 2rem;
    }
    
    /* กล่องผลลัพธ์ */
    .result-box-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .result-box-danger {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #4a6572 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* ปุ่ม */
    .stButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        font-size: 1.1rem;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==================== โหลดโมเดล ====================
@st.cache_resource
def load_model():
    model = joblib.load('heart_disease_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, feature_names

model, feature_names = load_model()

# ==================== Sidebar ====================
with st.sidebar:
    st.markdown("## 🏥 เกี่ยวกับแอป")
    st.markdown("""
    แอปพลิเคชันนี้ใช้ **Machine Learning** 
    โมเดล **Decision Tree** ในการประเมินความเสี่ยง
    การเป็นโรคหัวใจจากข้อมูลสุขภาพของคุณ
    
    ---
    **พัฒนาโดย:** AI Assistant  
    **โมเดล:** Decision Tree Classifier  
    **ความแม่นยำ:** ~80%+
    """)
    
    if st.button("🔄 รีเซ็ตค่าทั้งหมด"):
        st.rerun()

# ==================== Header ====================
st.markdown('<p class="main-title">❤️ Heart Disease Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">ระบบประเมินความเสี่ยงโรคหัวใจด้วย AI</p>', unsafe_allow_html=True)

# ==================== Input Form ====================
st.markdown("### 📝 กรุณากรอกข้อมูลสุขภาพของคุณ")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("🎂 อายุ (ปี)", min_value=20, max_value=100, value=50, step=1)
    sex = st.selectbox("⚧ เพศ", options=[1, 0], format_func=lambda x: "ชาย 👨" if x == 1 else "หญิง 👩")
    chest_pain = st.selectbox(
        "💔 ชนิดของอาการเจ็บหน้าอก",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 
                               3: "Non-anginal Pain", 4: "Asymptomatic"}[x]
    )

with col2:
    resting_bp = st.number_input("💓 ความดันโลหิตขณะพัก (mm Hg)", min_value=80, max_value=200, value=120, step=1)
    cholesterol = st.number_input("🩸 โคเลสเตอรอล (mg/dl)", min_value=0, max_value=600, value=200, step=1)
    fasting_bs = st.selectbox("🍬 น้ำตาลในเลือดหลังอดอาหาร > 120 mg/dl?", 
                              options=[0, 1], format_func=lambda x: "ใช่" if x == 1 else "ไม่ใช่")

with col3:
    resting_ecg = st.selectbox(
        "📈 ผล ECG ขณะพัก",
        options=[0, 1, 2, 3],
        format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 
                               2: "LV Hypertrophy", 3: "Possible/LV Hypertrophy"}[x]
    )
    max_hr = st.number_input("💗 อัตราการเต้นหัวใจสูงสุด", min_value=60, max_value=220, value=150, step=1)
    exercise_angina = st.selectbox("🏃 เจ็บหน้าอกขณะออกกำลังกาย?", 
                                   options=[0, 1], format_func=lambda x: "ใช่" if x == 1 else "ไม่ใช่")

col4, col5 = st.columns(2)
with col4:
    oldpeak = st.number_input("📉 ST Depression (Oldpeak)", min_value=-3.0, max_value=7.0, value=0.0, step=0.1, format="%.1f")
with col5:
    st_slope = st.selectbox(
        "📊 Slope ของ ST Segment",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x]
    )

# ==================== ปุ่มทำนาย ====================
st.markdown("---")
predict_button = st.button("🔮 ทำนายผล", use_container_width=True)

# ==================== ทำนายผล ====================
if predict_button:
    # สร้าง DataFrame จากข้อมูลผู้ใช้
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })
    
    # ทำนาย
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    st.markdown("---")
    
    # แสดงผลลัพธ์
    if prediction == 0:
        st.markdown(f"""
        <div class="result-box-safe">
            <h1 style="color:white; margin:0;">✅ ไม่มีความเสี่ยง</h1>
            <h2 style="color:white;">คุณมีแนวโน้ม <u>ไม่</u> เป็นโรคหัวใจ</h2>
            <p style="font-size:1.3rem; color:white;">
                ความมั่นใจ: <b>{probabilities[0]*100:.1f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box-danger">
            <h1 style="color:white; margin:0;">⚠️ มีความเสี่ยง</h1>
            <h2 style="color:white;">คุณมีแนวโน้ม <u>เป็น</u> โรคหัวใจ</h2>
            <p style="font-size:1.3rem; color:white;">
                ความมั่นใจ: <b>{probabilities[1]*100:.1f}%</b>
            </p>
            <p style="color:white; font-size:1rem;">
                💡 แนะนำให้ปรึกษาแพทย์เพื่อตรวจเพิ่มเติม
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # แสดงกราฟความน่าจะเป็น
    st.markdown("### 📊 ความน่าจะเป็นของการทำนาย")
    fig = go.Figure(data=[
        go.Pie(
            labels=['ไม่มีความเสี่ยง', 'มีความเสี่ยง'],
            values=probabilities,
            hole=0.4,
            marker=dict(colors=['#38ef7d', '#eb3349']),
            textinfo='percent+label',
            textfont_size=15
        )
    ])
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=14),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # แสดงข้อมูลผู้ใช้
    with st.expander("🔍 ดูข้อมูลที่คุณกรอก"):
        st.dataframe(input_data.T.rename(columns={0: 'ค่า'}), use_container_width=True)

# ==================== Footer ====================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:white; padding:1rem;">
    <p>⚕️ <b>คำเตือน:</b> แอปพลิเคชันนี้เป็นเพียงเครื่องมือประเมินเบื้องต้น 
    ไม่สามารถทดแทนการวินิจฉัยจากแพทย์ผู้เชี่ยวชาญได้</p>
    <p>© 2026 Heart Disease Predictor | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)