
import gradio as gr
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# --------------------------------
# 1. LOAD YOUR DATASET
# --------------------------------

possible_paths = [
    "student_performance_dataset.csv",
    "student_performance_dataset(3).csv",
    "/mnt/data/student_performance_dataset(3).csv"
]

DATA_PATH = next(
    (p for p in possible_paths if os.path.exists(p)),
    None
)

if DATA_PATH is None:
    raise FileNotFoundError(
        "Dataset not found. Put the CSV file in the same folder as app.py."
    )

df = pd.read_csv(DATA_PATH)

# Remove rows with missing values
df = df.dropna()

# --------------------------------
# 2. PREPARE DATA
# --------------------------------

X = df.drop(
    columns=["Student_ID", "Final_Academic_Performance"]
).copy()

# Convert Gender into numbers
X["Gender"] = X["Gender"].map({
    "Female": 0,
    "Male": 1
})

y = df["Final_Academic_Performance"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------
# 3. TRAIN MODEL
# --------------------------------

model = GaussianNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

# --------------------------------
# 4. PREDICTION FUNCTION
# --------------------------------

def predict(age, gender, attendance, gpa, logins,
            study_hours, assignment_rate, engagement):

    input_data = pd.DataFrame([[
        age,
        gender,
        attendance,
        gpa,
        logins,
        study_hours,
        assignment_rate,
        engagement
    ]], columns=X.columns)

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # Get probability corresponding to predicted class
    class_index = list(model.classes_).index(prediction)
    confidence = probabilities[class_index] * 100

    result = f"Predicted Academic Performance: {prediction}"

    confidence_text = (
        f"Model confidence: {confidence:.2f}%"
    )

    return result, confidence_text


# --------------------------------
# 5. DASHBOARD DESIGN
# --------------------------------

css = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
}

#header {
    background: linear-gradient(120deg, #14213d, #007f86);
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    color: white;
    margin-bottom: 18px;
}

#header h1 {
    color: white;
    font-size: 32px;
}

#header p {
    color: #e0f2f1;
    font-size: 15px;
}

.card {
    border: 1px solid #dce5ed !important;
    border-radius: 15px !important;
    padding: 15px !important;
}

#predict {
    background: #087e8b !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 17px !important;
    border-radius: 10px !important;
}

#result {
    border-radius: 12px !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="blue"
    ),
    css=css,
    title="Student Performance Dashboard"
) as demo:

    gr.HTML("""
    <div id="header">
        <h1>🎓 Student Performance Prediction</h1>
        <p>Machine Learning Based Academic Performance Dashboard</p>
        <p>Analyze student information and predict academic performance</p>
    </div>
    """)

    # Dataset overview
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Dataset Overview")
            gr.Markdown(
                f"**Total Records:** {len(df)}  \n"
                f"**Input Features:** {X.shape[1]}  \n"
                f"**Target Classes:** High, Medium, Low"
            )

        with gr.Column():
            gr.Markdown("### 🤖 Model Performance")
            gr.Markdown(
                f"**Algorithm:** Gaussian Naive Bayes  \n"
                f"**Training Records:** {len(X_train)}  \n"
                f"**Testing Records:** {len(X_test)}  \n"
                f"**Test Accuracy:** {accuracy:.2f}%"
            )

    gr.Markdown("---")
    gr.Markdown("## 👨‍🎓 Enter Student Details")

    with gr.Row():

        with gr.Column(elem_classes="card"):
            gr.Markdown("### Personal Information")

            age = gr.Number(
                label="Age",
                value=float(df["Age"].median())
            )

            gender = gr.Radio(
                choices=[
                    ("Female", 0),
                    ("Male", 1)
                ],
                label="Gender",
                value=0
            )

            gr.Markdown("### Academic Background")

            gpa = gr.Slider(
                float(df["Prior_GPA"].min()),
                float(df["Prior_GPA"].max()),
                value=float(df["Prior_GPA"].median()),
                label="Prior GPA"
            )

            attendance = gr.Slider(
                float(df["Attendance_Rate"].min()),
                float(df["Attendance_Rate"].max()),
                value=float(df["Attendance_Rate"].median()),
                label="Attendance Rate (%)"
            )

        with gr.Column(elem_classes="card"):
            gr.Markdown("### Learning & Engagement")

            logins = gr.Number(
                label="LMS Logins",
                value=float(df["LMS_Logins"].median())
            )

            study_hours = gr.Slider(
                float(df["Study_Hours_Per_Week"].min()),
                float(df["Study_Hours_Per_Week"].max()),
                value=float(df["Study_Hours_Per_Week"].median()),
                label="Study Hours Per Week"
            )

            assignment_rate = gr.Slider(
                float(df["Assignment_Submission_Rate"].min()),
                float(df["Assignment_Submission_Rate"].max()),
                value=float(df["Assignment_Submission_Rate"].median()),
                label="Assignment Submission Rate (%)"
            )

            engagement = gr.Slider(
                float(df["Engagement_Score"].min()),
                float(df["Engagement_Score"].max()),
                value=float(df["Engagement_Score"].median()),
                label="Engagement Score"
            )

    gr.Markdown("---")

    predict_btn = gr.Button(
        "🔍 Predict Academic Performance",
        variant="primary",
        elem_id="predict"
    )

    with gr.Column(elem_id="result"):
        gr.Markdown("## 📈 Prediction Result")

        result = gr.Textbox(
            label="Predicted Category",
            interactive=False
        )

        confidence = gr.Textbox(
            label="Prediction Probability",
            interactive=False
        )

    predict_btn.click(
        fn=predict,
        inputs=[
            age, gender, attendance, gpa,
            logins, study_hours,
            assignment_rate, engagement
        ],
        outputs=[result, confidence]
    )

    gr.Markdown("""
    ---
    <center>
    <small>
    Student Performance ML Project | Gaussian Naive Bayes
    </small>
    </center>

    **Disclaimer:** This is an educational prediction model.
    Its output is not a definitive assessment of a student's ability.
    """)


# --------------------------------
# 6. LAUNCH DASHBOARD
# --------------------------------

demo.launch(
    share=True,
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
