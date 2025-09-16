import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv("modified_data.csv")

if 'date' in data.columns:
    data = data.drop(columns=['date'])
X = data.drop(columns=['actual_productivity'])
y = data['actual_productivity']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

# Fit models
for name, model in models.items():
    model.fit(X_train, y_train)


def predict(model_name, *features):
    model = models[model_name]
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)[0]
    return f"📊 Predicted Productivity: {pred:.3f}"

def login(username, password):
    if username == "admin" and password == "1234":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(value="❌ Invalid login. Try again."), gr.update(visible=False)

with gr.Blocks() as demo:
    # Page 1: Login
    with gr.Row(visible=True) as login_page:
        with gr.Column():
            gr.Markdown("## 🔑 Login to Access Productivity Prediction App")
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            login_msg = gr.Textbox(label="Login Status")

    with gr.Row(visible=False) as app_page:
        with gr.Column():
            gr.Markdown("## 👕 Garment Worker Productivity Prediction")

            model_choice = gr.Dropdown(list(models.keys()), label="Select Model", value="Random Forest")

            inputs = []
            with gr.Accordion("Enter Feature Values", open=False):
                for col in X.columns:
                    inputs.append(gr.Number(label=col, value=float(data[col].median())))

            btn = gr.Button("Predict")
            output_text = gr.Textbox(label="Result")

    # Button Actions
    login_btn.click(fn=login, inputs=[username, password], outputs=[login_msg, app_page])
    btn.click(fn=predict, inputs=[model_choice] + inputs, outputs=[output_text])

if __name__ == "__main__":
    demo.launch()
