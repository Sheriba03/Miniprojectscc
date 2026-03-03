from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os

app = Flask(__name__)

# -------- LOAD DATASET --------
df = pd.read_excel("disease_last_dataset_420.xlsx")

df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")

SYMPTOMS = [
    "Fever","Cold","Cough","Headache","Fatigue","Nausea","Vomiting",
    "Body_Pain","Shortness_of_Breath","Chest_Pain","Dizziness",
    "Insomnia","Sore_Throat","Palpitations","Abnormal_Movements"
]

X = df[SYMPTOMS]
le = LabelEncoder()
y = le.fit_transform(df["Disease"])

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

def risk_color(risk):
    if risk == "Mild":
        return "green"
    elif risk == "Moderate":
        return "yellow"
    else:
        return "red"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print("PREDICT BUTTON CLICKED")
    print(data)

    selected = data.get("symptoms", [])
    days = int(data.get("days", 0))

    input_data = [1 if s in selected else 0 for s in SYMPTOMS]
    input_df = pd.DataFrame([input_data], columns=SYMPTOMS)

    pred = model.predict(input_df)[0]
    disease = le.inverse_transform([pred])[0]

    row = df[df["Disease"] == disease].iloc[0]

    log = pd.DataFrame([{
        "Name": data.get("name"),
        "Email": data.get("email"),
        "Phone": data.get("phone"),
        "Symptoms": ",".join(selected),
        "Days": days,
        "Disease": disease,
        "Risk": row["Risk"],
        "Date": datetime.now()
    }])

    if os.path.exists("user_live_data.xlsx"):
        old = pd.read_excel("user_live_data.xlsx")
        pd.concat([old, log], ignore_index=True).to_excel("user_live_data.xlsx", index=False)
    else:
        log.to_excel("user_live_data.xlsx", index=False)

    return jsonify({
        "disease": disease,
        "remedy": row["Home_Remedy"],
        "risk": row["Risk"],
        "color": risk_color(row["Risk"])
    })

if __name__ == "__main__":
    app.run(debug=True)

