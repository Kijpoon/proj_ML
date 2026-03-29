import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. การตั้งค่าหน้าเว็บ (Page Config)
# ==========================================
st.set_page_config(
    page_title="Road Accident Severity Predictor",
    page_icon="🚨",
    layout="centered"
)

# ==========================================
# 2. โหลดโมเดล (Cache ไว้จะได้ไม่โหลดใหม่ทุกครั้งที่กดปุ่ม)
# ==========================================
@st.cache_resource
def load_model():
    try:
        # โหลดไฟล์โมเดลที่เซฟมาจาก Jupyter Notebook
        model = joblib.load('lgb_model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์ 'lgb_model.pkl' กรุณาตรวจสอบให้แน่ใจว่าได้อัปโหลดไฟล์โมเดลแล้ว")
        return None

model = load_model()

# ==========================================
# 3. ส่วนแสดงผล UI (User Interface)
# ==========================================
st.title("🚨 ระบบประเมินความรุนแรงของอุบัติเหตุทางถนน")
st.markdown("""
ระบบนี้ใช้แบบจำลอง **LightGBM** ร่วมกับเทคนิค **Custom Thresholding** เพื่อเพิ่มความไวในการดักจับเคสบาดเจ็บสาหัสและเสียชีวิต 
(Serious Threshold: 25%, Fatal Threshold: 15%)
""")

st.header("📋 กรอกข้อมูลอุบัติเหตุ (Top 10 Features)")
st.write("กรุณาระบุข้อมูลคุณลักษณะเด่น 10 อันดับแรกที่ได้จากการวิเคราะห์")

col1, col2 = st.columns(2)

with col1:
    feature_1 = st.number_input("1. ลักษณะถนน (Road Type)", value=0.0)
    feature_2 = st.number_input("2. สภาพอากาศ (Weather)", value=0.0)
    feature_3 = st.number_input("3. สภาพแสง (Light Condition)", value=0.0)
    feature_4 = st.number_input("4. ความเร็วรถ (Speed Limit)", value=0.0)
    feature_5 = st.number_input("5. ประเภทยานพาหนะ (Vehicle Type)", value=0.0)

with col2:
    feature_6 = st.number_input("6. อายุผู้ขับขี่ (Driver Age)", value=0.0)
    feature_7 = st.number_input("7. เพศผู้ขับขี่ (Driver Sex)", value=0.0)
    feature_8 = st.number_input("8. จุดเกิดเหตุ (Junction Detail)", value=0.0)
    feature_9 = st.number_input("9. วันในสัปดาห์ (Day of Week)", value=0.0)
    feature_10 = st.number_input("10. เวลาที่เกิดเหตุ (Time)", value=0.0)

# ==========================================
# 4. ฟังก์ชันสำหรับการทำนายแบบกำหนด Threshold
# ==========================================
def custom_predict(probs, threshold_class2=0.05, threshold_class1=0.25):
    # probs คือ array ของความน่าจะเป็น [prob_0, prob_1, prob_2]
    p = probs[0] 
    if p[2] >= threshold_class2:
        return 2, p[2] # Fatal
    elif p[1] >= threshold_class1:
        return 1, p[1] # Serious
    else:
        return 0, p[0] # Slight

# ==========================================
# 5. ปุ่มกดทำนาย (Predict Button)
# ==========================================
st.markdown("---")
if st.button("🔍 ประเมินความรุนแรง", type="primary", use_container_width=True):
    if model is not None:
        input_data = pd.DataFrame([[
            feature_1, feature_2, feature_3, feature_4, feature_5, 
            feature_6, feature_7, feature_8, feature_9, feature_10
        ]], columns=[
            'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
            'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10'
        ])
        
        # 1. ให้โมเดลคำนวณความน่าจะเป็น
        probs = model.predict_proba(input_data)
        
        # 2. ปรับ Threshold ตามที่เราตกลงกันไว้
        pred_class, confidence = custom_predict(probs, threshold_class2=0.05, threshold_class1=0.25)
        
        # 3. แสดงผลลัพธ์
        st.subheader("🎯 ผลการประเมิน:")
        
        if pred_class == 2:
            st.error(f"🔴 ระดับความรุนแรง: **เสียชีวิต (Fatal)**")
            st.write(f"ความน่าจะเป็นที่เข้าเกณฑ์: {confidence*100:.2f}%")
            st.warning("⚠️ โปรดแจ้งหน่วยกู้ชีพขั้นสูงและเตรียมเครื่องมือแพทย์ฉุกเฉินด่วนที่สุด!")
            
        elif pred_class == 1:
            st.warning(f"🟠 ระดับความรุนแรง: **บาดเจ็บสาหัส (Serious)**")
            st.write(f"ความน่าจะเป็นที่เข้าเกณฑ์: {confidence*100:.2f}%")
            st.info("🚑 โปรดเตรียมพร้อมรถพยาบาลและเจ้าหน้าที่กู้ภัย")
            
        else:
            st.success(f"🟢 ระดับความรุนแรง: **บาดเจ็บเล็กน้อย (Slight)**")
            st.write(f"ความน่าจะเป็น: {confidence*100:.2f}%")
            st.write("✅ จัดส่งทีมกู้ภัยเพื่อประเมินสถานการณ์เบื้องต้น")
            
        # แสดงตารางความน่าจะเป็นทุกคลาสแบบละเอียด
        st.markdown("---")
        st.write("📊 รายละเอียดความน่าจะเป็น (Probabilities):")
        prob_df = pd.DataFrame(probs, columns=['Slight (0)', 'Serious (1)', 'Fatal (2)'])
        # แปลงเป็น %
        prob_df = (prob_df * 100).round(2).astype(str) + '%'
        st.table(prob_df)
