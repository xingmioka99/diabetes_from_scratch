import gradio as gr
import pandas as pd
import os
import sys

# point to src folder
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from classification_model import rf, X
from logistic_model import LogisticModel

# load and train logistic model
base_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_dir, '..', 'data', 'clean_diabetes.csv'))

from sklearn.model_selection import train_test_split
X_data = df.drop('class', axis=1)
y_data = df['class']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

lr_model = LogisticModel(X_train, y_train)
lr_model.train()

def predict(Age, Gender, Polyuria, Polydipsia, sudden_weight_loss, weakness,
            Polyphagia, Genital_thrush, visual_blurring, Itching, Irritability,
            delayed_healing, partial_paresis, muscle_stiffness, Alopecia, Obesity):

    if Age is None:
        Age = 35

    row = [
        float(Age),
        int(Gender or 0),
        int(Polyuria or 0),
        int(Polydipsia or 0),
        int(sudden_weight_loss or 0),
        int(weakness or 0),
        int(Polyphagia or 0),
        int(Genital_thrush or 0),
        int(visual_blurring or 0),
        int(Itching or 0),
        int(Irritability or 0),
        int(delayed_healing or 0),
        int(partial_paresis or 0),
        int(muscle_stiffness or 0),
        int(Alopecia or 0),
        int(Obesity or 0)
    ]
    input_data = pd.DataFrame([row], columns=X.columns)

    # Random Forest
    rf_result = rf.predict(input_data)[0]
    rf_prob   = rf.predict_proba(input_data)[0]
    rf_label  = "Positive" if rf_result == 1 else "Negative"

    # Logistic Regression
    lr_result = lr_model.model.predict(input_data)[0]
    lr_prob   = lr_model.model.predict_proba(input_data)[0]
    lr_label  = "Positive" if lr_result == 1 else "Negative"
    avg_positive = (rf_prob[1] + lr_prob[1])/2

    if avg_positive < 0.50:
        alert = "✅ GOOD — Low risk of diabetes. Stay healthy and maintain your lifestyle!"
    elif avg_positive < 0.75:
        alert = "⚠️ MODERATE — Moderate risk detected. Consider consulting a doctor."
    else:
        alert = "🚨 HIGH RISK — Elevated risk detected. Please consult a healthcare professional for clinical evaluation."

    return (f"{rf_label} (Negative: {rf_prob[0]:.2f}, Positive: {rf_prob[1]:.2f})",
            f"{lr_label} (Negative: {lr_prob[0]:.2f}, Positive: {lr_prob[1]:.2f})",
            alert)


inputs = [
    gr.Number(label="Age", value=35),
    gr.Radio([0, 1], label="Gender (0=Female, 1=Male)", value=0),
    gr.Radio([0, 1], label="Polyuria (Excessive or frequent urination)", value=0),
    gr.Radio([0, 1], label="Polydipsia (Excessive thirst or increased water intake)", value=0),
    gr.Radio([0, 1], label="Sudden Weight Loss (Unexplained or rapid drop in body mass)", value=0),
    gr.Radio([0, 1], label="Weakness (Generalized fatigue, lethargy, or lack of energy)", value=0),
    gr.Radio([0, 1], label="Polyphagia (Excessive hunger or significantly increased appetite)", value=0),
    gr.Radio([0, 1], label="Genital Thrush (Presence of a yeast infection)", value=0),
    gr.Radio([0, 1], label="Visual Blurring (Blurred vision caused by fluid shifts in the eye lenses)", value=0),
    gr.Radio([0, 1], label="Itching (General skin irritation or localized itching)", value=0),
    gr.Radio([0, 1], label="Irritability (Mood swings or sudden changes in temperament)", value=0),
    gr.Radio([0, 1], label="Delayed Healing (Wounds or sores that take an abnormally long time to recover)", value=0),
    gr.Radio([0, 1], label="Partial Paresis (Significant muscle weakness or partial loss of voluntary movement)", value=0),
    gr.Radio([0, 1], label="Muscle Stiffness (Tightness or inability to move muscles and joints easily)", value=0),
    gr.Radio([0, 1], label="Alopecia (Patchy or sudden hair loss)", value=0),
    gr.Radio([0, 1], label="Obesity (High Body Mass Index (BMI))", value=0),
]

outputs = [
    gr.Text(label="Random Forest Prediction"),
    gr.Text(label="Logistic Regression Prediction"),
    gr.Text(label="Risk Alert"),
]

demo = gr.Interface(fn=predict, inputs=inputs, outputs=outputs,
                    title="Diabetes Risk Prediction",
                    description="Compare Random Forest vs Logistic Regression")

if __name__ == '__main__':
    demo.launch(inbrowser=True)