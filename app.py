import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title = "Smart Placement Predictor",
    page_icon  = "🎓",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)
st.markdown("""
    <style>
    /* Selectbox dropdown — pointer cursor */
    div[data-baseweb="select"] {
        cursor: pointer !important;
    }
    div[data-baseweb="select"] * {
        cursor: pointer !important;
    }

    /* Slider — pointer cursor */
    div[data-testid="stSlider"] * {
        cursor: pointer !important;
    }

    /* Button — pointer cursor */
    div.stButton > button {
        cursor: pointer !important;
    }

    /* Sidebar selectbox options */
    ul[data-testid="stSelectboxVirtualDropdown"] li {
        cursor: pointer !important;
    }
    </style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_artifacts():
    model            = joblib.load('models/rf_model.pkl')
    scaler           = joblib.load('models/scaler.pkl')
    label_encoders   = joblib.load('models/label_encoders.pkl')
    feature_names    = joblib.load('models/feature_names.pkl')
    numeric_cols     = joblib.load('models/numeric_cols.pkl')
    categorical_cols = joblib.load('models/categorical_cols.pkl')
    targets          = joblib.load('models/improvement_targets.pkl')
    return (model, scaler, label_encoders, feature_names,
            numeric_cols, categorical_cols, targets)

(model, scaler, label_encoders, feature_names,
 numeric_cols, categorical_cols, targets) = load_artifacts()

@st.cache_resource
def load_explainer():
    return joblib.load('models/shap_explainer.pkl')
def encode_and_scale(raw_input):
    """Takes raw user input dict, encodes categories, scales numerics."""
    df_input = pd.DataFrame([raw_input])

    for col in categorical_cols:
        le = label_encoders[col]
        df_input[col] = le.transform(df_input[col].astype(str))

    df_input = df_input[feature_names]
    df_display = df_input.copy()

    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])
    return df_input, df_display


def get_probability(scaled_df):
    return model.predict_proba(scaled_df)[0][1]


def get_shap_values(scaled_df):
    explainer = load_explainer()
    sv = explainer.shap_values(scaled_df)
    if isinstance(sv, list):
        return sv[1][0]
    elif len(np.array(sv).shape) == 3:
        return np.array(sv)[0, :, 1]
    else:
        return sv[0]
    # Session state — remembers prediction across reruns
if 'predicted' not in st.session_state:
    st.session_state.predicted = False
if 'raw_input' not in st.session_state:
    st.session_state.raw_input = None
if 'prob' not in st.session_state:
    st.session_state.prob = None


def run_skill_gap(raw_input, current_prob):
    """Simulates improving each weak feature and measures probability impact."""
    recommendations = []

    for feature, target in targets.items():
        if feature not in raw_input:
            continue

        current_val = raw_input[feature]
        needs_improvement = (
            current_val > target if feature == 'backlogs'
            else current_val < target
        )

        if needs_improvement:
            improved_input          = raw_input.copy()
            improved_input[feature] = target
            imp_scaled, _           = encode_and_scale(improved_input)
            new_prob                = get_probability(imp_scaled)
            impact                  = (new_prob - current_prob) * 100

            if impact > 0.5:
                recommendations.append({
                    'feature'      : feature,
                    'current_value': abs(round(float(current_val), 2)),
                    'target_value' : target,
                    'current_prob' : round(current_prob * 100, 1),
                    'new_prob'     : round(new_prob * 100, 1),
                    'impact'       : round(impact, 1)
                })

    recommendations.sort(key=lambda x: x['impact'], reverse=True)
    return recommendations
st.title("🎓 Smart College Placement Predictor")
st.markdown("Enter student details to predict placement probability, "
            "understand the reasons, and get personalized improvement suggestions.")

with st.sidebar:
    st.header("📋 Student Profile")
    st.markdown("---")

    branch_display = st.selectbox(
        "Engineering Branch",
        ["CSE",
         "AIDS",
         "IT",
         "EEE",
         "ECE",
         "Civil",
         "Mechanical"]
    )
    branch_mapping = {
        "CSE"                  : "CSE",
        "AIDS"                 : "CSE",
        "IT"                   : "IT",
        "EEE"                  : "ECE",
        "ECE"                  : "ECE",
        "Civil"                : "CE",
        "Mechanical"           : "ME"
    }
    branch = branch_mapping[branch_display]

    st.markdown("---")
    st.subheader("🎓 Academic Details")

    cgpa = st.slider("CGPA (out of 10)", 4.0, 10.0, 7.0, 0.1,
                     help="Your current cumulative GPA")

    twelfth_percentage = st.slider("12th Percentage",
                                   40.0, 100.0, 70.0, 1.0,
                                   help="Your Class 12 board exam percentage")

    backlogs = st.slider("Number of Active Backlogs", 0, 10, 0, 1,
                         help="Number of subjects failed or pending")

    attendance_percentage = st.slider("Attendance Percentage",
                                      40.0, 100.0, 75.0, 1.0,
                                      help="Your college attendance percentage")

    st.markdown("---")
    st.subheader("💻 Skills")
    st.caption("Pick the option that honestly describes you.")

    coding_level = st.selectbox(
        "Coding Skill Level",
        ["Beginner",
         "Intermediate",
         "Advanced"]
    )
    coding_skill_rating = {
        "Beginner"      : 2,
        "Intermediate"  : 3,
        "Advanced"      : 5
    }[coding_level]

    aptitude_level = st.selectbox(
        "Aptitude & Logical Reasoning",
        ["Weak",
         "Average",
         "Strong"]
    )
    aptitude_skill_rating = {
        "Weak"       : 2,
        "Average"    : 3,
        "Strong"     : 5
    }[aptitude_level]

    communication_level = st.selectbox(
        "Communication & Presentation Skills",
        ["Basic",
         "Moderate",
         "Good"]
    )
    communication_skill_rating = {
        "Basic"        : 2,
        "Moderate"     : 3,
        "Good"         : 5
    }[communication_level]

    st.markdown("---")
    st.subheader("🏆 Experience & Activities")

    internships_completed   = st.slider("Internships Completed",   0, 5,  0, 1,
                                        help="Paid or unpaid internships")
    projects_completed      = st.slider("Projects Completed",      0, 10, 1, 1,
                                        help="Academic + personal projects")
    hackathons_participated = st.slider("Hackathons Participated", 0, 10, 0, 1)
    certifications_count    = st.slider("Online Certifications",   0, 10, 1, 1,
                                        help="Coursera, NPTEL, Udemy, etc.")
    study_hours_options = [
        "1:00", "1:15", "1:30", "1:45",
        "2:00", "2:15", "2:30", "2:45",
        "3:00", "3:15", "3:30", "3:45",
        "4:00", "4:15", "4:30", "4:45",
        "5:00", "5:15", "5:30", "5:45",
        "6:00", "6:15", "6:30", "6:45",
        "7:00", "7:15", "7:30", "7:45",
        "8:00", "8:15", "8:30", "8:45",
        "9:00", "9:15", "9:30", "9:45",
        "10:00", "10:15", "10:30", "10:45",
        "11:00", "11:15", "11:30", "11:45",
        "12:00"
    ]
    study_hours_display = st.select_slider(
        "Study Hours Per Day",
        options=study_hours_options,
        value="4:00",
        help="Select your daily study hours (Hours:Minutes)"
    )
    study_h, study_m    = map(int, study_hours_display.split(":"))
    study_hours_per_day = study_h + study_m / 60

    st.markdown("---")

    # Silent defaults for removed low-importance features
    # Model still receives all 22 features
    gender                      = "Male"
    tenth_percentage            = twelfth_percentage - 2
    family_income_level         = "Medium"
    city_tier                   = "Tier 2"
    internet_access             = "Yes"
    part_time_job               = "No"
    extracurricular_involvement = "Medium"
    sleep_hours                 = 7.0
    stress_level                = 3

    predict_btn = st.button("🔍 Predict Placement",
                            type="primary",
                            use_container_width=True)
if 'predicted' not in st.session_state:
    st.session_state.predicted = False
if 'raw_input' not in st.session_state:
    st.session_state.raw_input = None
if 'prob' not in st.session_state:
    st.session_state.prob = None

if predict_btn:
    st.session_state.predicted = True
    st.session_state.raw_input = {
        'gender'                    : gender,
        'branch'                    : branch,
        'cgpa'                      : cgpa,
        'tenth_percentage'          : tenth_percentage,
        'twelfth_percentage'        : twelfth_percentage,
        'backlogs'                  : backlogs,
        'study_hours_per_day'       : study_hours_per_day,
        'attendance_percentage'     : attendance_percentage,
        'projects_completed'        : projects_completed,
        'internships_completed'     : internships_completed,
        'coding_skill_rating'       : coding_skill_rating,
        'communication_skill_rating': communication_skill_rating,
        'aptitude_skill_rating'     : aptitude_skill_rating,
        'hackathons_participated'   : hackathons_participated,
        'certifications_count'      : certifications_count,
        'sleep_hours'               : sleep_hours,
        'stress_level'              : stress_level,
        'part_time_job'             : part_time_job,
        'family_income_level'       : family_income_level,
        'city_tier'                 : city_tier,
        'internet_access'           : internet_access,
        'extracurricular_involvement': extracurricular_involvement,
    }
    scaled_input, _          = encode_and_scale(st.session_state.raw_input)
    st.session_state.prob    = get_probability(scaled_input)

if st.session_state.predicted and st.session_state.raw_input is not None:

    raw_input    = st.session_state.raw_input
    prob         = st.session_state.prob
    scaled_input, _ = encode_and_scale(raw_input)

    # ── Top metrics ───────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Placement Probability", f"{prob:.1%}")
    with col2:
        status = "✅ Likely Placed" if prob >= 0.5 else "⚠️ At Risk"
        st.metric("Prediction", status)
    with col3:
        if prob >= 0.75 or prob <= 0.25:
            conf = "High"
        else:
            conf = "Medium"
        st.metric("Model Confidence", conf)

    st.progress(float(prob))

    if prob >= 0.75:
        st.success("Strong placement prospect! Keep maintaining your performance.")
    elif prob >= 0.5:
        st.warning("Moderate placement chance. A few improvements can boost your odds significantly.")
    else:
        st.error("High risk. Follow the skill gap recommendations below to improve your chances.")

    # ── Three tabs ────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🔍 Why this result?",
        "📈 Skill Gap Analyzer",
        "🔮 What-If Analysis"
    ])

    # ── TAB 1: SHAP explanation ───────────────────────────────
    with tab1:
        st.subheader("Feature Impact on Your Prediction")
        st.caption("Green bars push probability UP. "
                   "Red bars pull probability DOWN.")

        shap_vals = get_shap_values(scaled_input)

        fig, ax = plt.subplots(figsize=(10, 7))
        colors  = ['#2ecc71' if v > 0 else '#e74c3c' for v in shap_vals]
        sorted_idx = np.argsort(np.abs(shap_vals))
        ax.barh(
            [feature_names[i] for i in sorted_idx],
            [shap_vals[i] for i in sorted_idx],
            color=[colors[i] for i in sorted_idx]
        )
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP Value (Impact on Placement Probability)')
        ax.set_title('What is driving your placement probability?')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        neg_features = [(feature_names[i], shap_vals[i])
                        for i in range(len(shap_vals)) if shap_vals[i] < 0]
        neg_features.sort(key=lambda x: x[1])

        if neg_features:
            st.markdown("**🔴 Main factors reducing your chances:**")
            for fname, fval in neg_features[:3]:
                st.write(f"- **{fname}** is reducing your probability "
                         f"by {abs(fval)*100:.1f}%")

    # ── TAB 2: Skill Gap Analyzer ─────────────────────────────
    with tab2:
        st.subheader("Personalized Improvement Recommendations")
        st.caption("Each recommendation shows how much your probability "
                   "increases if you improve that one feature.")

        recs = run_skill_gap(raw_input, prob)

        if not recs:
            st.success("Your profile is already strong across all key areas!")
        else:
            for i, r in enumerate(recs[:6], 1):
                direction   = "Reduce" if r['feature'] == 'backlogs' else "Improve"
                curr_disp   = int(r['current_value']) \
                              if r['current_value'] == int(r['current_value']) \
                              else r['current_value']
                target_disp = int(r['target_value']) \
                              if r['target_value'] == int(r['target_value']) \
                              else r['target_value']

                with st.expander(
                    f"#{i}  {direction} **{r['feature']}** "
                    f"→ probability increases by **+{r['impact']}%**",
                    expanded=(i <= 3)
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Current Value", curr_disp)
                    c2.metric("Target Value",  target_disp)
                    c3.metric("Probability Gain", f"+{r['impact']}%")
                    st.progress(min(r['new_prob'] / 100, 1.0))
                    st.caption(f"Probability: {r['current_prob']}% "
                               f"→ {r['new_prob']}%")

    # ── TAB 3: What-If Analysis ───────────────────────────────
    with tab3:
        st.subheader("What-If Analysis")
        st.caption("Adjust the sliders to simulate improvements "
                   "and see the live probability change.")

        wc1, wc2 = st.columns(2)
        with wc1:
            wi_cgpa = st.slider(
                "Simulate CGPA",
                4.0, 10.0,
                float(st.session_state.raw_input['cgpa']),
                0.1, key='wi_cgpa'
            )
            wi_backlogs = st.slider(
                "Simulate Backlogs",
                0, 10,
                int(st.session_state.raw_input['backlogs']),
                1, key='wi_bl'
            )
            wi_coding = st.slider(
                "Simulate Coding Skill (1-5)",
                1, 5,
                int(st.session_state.raw_input['coding_skill_rating']),
                1, key='wi_code'
            )
        with wc2:
            wi_internships = st.slider(
                "Simulate Internships",
                0, 5,
                int(st.session_state.raw_input['internships_completed']),
                1, key='wi_int'
            )
            wi_aptitude = st.slider(
                "Simulate Aptitude Skill (1-5)",
                1, 5,
                int(st.session_state.raw_input['aptitude_skill_rating']),
                1, key='wi_apt'
            )
            wi_attendance = st.slider(
                "Simulate Attendance %",
                40.0, 100.0,
                float(st.session_state.raw_input['attendance_percentage']),
                0.5, key='wi_att'
            )

        wi_input = st.session_state.raw_input.copy()
        wi_input.update({
            'cgpa'                 : wi_cgpa,
            'backlogs'             : wi_backlogs,
            'coding_skill_rating'  : wi_coding,
            'internships_completed': wi_internships,
            'aptitude_skill_rating': wi_aptitude,
            'attendance_percentage': wi_attendance,
        })

        wi_scaled, _  = encode_and_scale(wi_input)
        wi_prob       = get_probability(wi_scaled)
        wi_delta      = (wi_prob - st.session_state.prob) * 100

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Probability",   f"{st.session_state.prob:.1%}")
        m2.metric("Simulated Probability", f"{wi_prob:.1%}")
        m3.metric("Change", f"{wi_delta:+.1f}%",
                  delta=f"{wi_delta:+.1f}%")

        st.progress(float(wi_prob))

        if wi_delta > 0:
            st.success(f"These changes would increase your probability "
                       f"from {st.session_state.prob:.1%} to "
                       f"{wi_prob:.1%} (+{wi_delta:.1f}%)")
        elif wi_delta < 0:
            st.warning(f"These changes would decrease your probability "
                       f"from {st.session_state.prob:.1%} to "
                       f"{wi_prob:.1%} ({wi_delta:.1f}%)")
        else:
            st.info("No change in probability with these settings.")

else:
    st.info("👈 Fill in the student details in the sidebar and "
            "click **Predict Placement** to get results.")