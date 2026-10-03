import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from utils import *
from joblib import load
from scipy.sparse import issparse
import numpy as np
# ============================================================
# 1. LOAD the FOREST MODEL
# ============================================================

c_model = load("models/car_price_classification_model.pkl")
preprocessor = load("models/car_price_classification_preprocessor.pkl")
scaler = load("models/car_price_classification_scaler.pkl")

# ============================================================
# 9. CREATE DASH APP
# ============================================================

app = dash.Dash(__name__)

app.title = "Car Price Prediction v3"


# ============================================================
# 10. DASH LAYOUT
# ============================================================

app.layout = html.Div(
    style={
        "maxWidth": "900px",
        "margin": "auto",
        "padding": "40px",
        "fontFamily": "Arial"
    },

    children=[

        html.H1(
            "Car Price Prediction v3",
            style={
                "textAlign": "center"
            }
        ),
        html.P(
            [
                "The v3 car price classification model using Multinomial Logistic Regression. Instead of predicting the exact selling price of a car as a continuous value, the original selling price is divided into four discrete price classes (0, 1, 2, and 3). The model uses the preprocessed car dataset and predicts which price class a car belongs to based on its features.",
                html.Br(),
                html.Br()
            ],
            style={
                "backgroundColor": "#f4f8ff",
                "padding": "20px",
                "borderRadius": "8px",
                "lineHeight": "1.6",
                "color": "#333"
            }
        ),
        html.Hr(),
          html.P(
            "Enter the car information below to predict its categories. And click the button to get the prediction.",
            style={
                "textAlign": "center",
                "color": "#666"
            }
        ),

        html.Hr(),

        # ----------------------------------------------------
        # NAME
        # ----------------------------------------------------

        html.Label("Car Name"),

        dcc.Input(
            id="name",
            type="text",
            placeholder="e.g. Maruti Swift",
            style={
                "width": "100%",
                "padding": "10px",
                "marginBottom": "20px"
            }
        ),

        # ----------------------------------------------------
        # YEAR
        # ----------------------------------------------------

        html.Label("Year"),

        dcc.Input(
            id="year",
            type="number",
            placeholder="e.g. 2018",
            style={
                "width": "100%",
                "padding": "10px",
                "marginBottom": "20px"
            }
        ),

        # ----------------------------------------------------
        # KM DRIVEN
        # ----------------------------------------------------

        html.Label("Kilometers Driven"),

        dcc.Input(
            id="km_driven",
            type="number",
            placeholder="e.g. 50000",
            style={
                "width": "100%",
                "padding": "10px",
                "marginBottom": "20px"
            }
        ),

        # ----------------------------------------------------
        # FUEL
        # ----------------------------------------------------

        html.Label("Fuel"),

        dcc.Dropdown(
            id="fuel",
            options=[
                {"label": "Diesel", "value": "Diesel"},
                {"label": "Petrol", "value": "Petrol"},
                {"label": "Electric", "value": "Electric"}
            ],
            placeholder="Select fuel",
            style={
                "marginBottom": "20px"
            }
        ),

        # ----------------------------------------------------
        # SELLER TYPE
        # ----------------------------------------------------

        html.Label("Seller Type"),

        dcc.Dropdown(
            id="seller_type",
            options=[
                {
                    "label": "Individual",
                    "value": "Individual"
                },
                {
                    "label": "Dealer",
                    "value": "Dealer"
                },
                {
                    "label": "Trustmark Dealer",
                    "value": "Trustmark Dealer"
                }
            ],
            placeholder="Select seller type",
            style={
                "marginBottom": "20px"
            }
        ),

        # ----------------------------------------------------
        # TRANSMISSION
        # ----------------------------------------------------

        html.Label("Transmission"),

        dcc.Dropdown(
            id="transmission",
            options=[
                {
                    "label": "Manual",
                    "value": "Manual"
                },
                {
                    "label": "Automatic",
                    "value": "Automatic"
                }
            ],
            placeholder="Select transmission",
            style={
                "marginBottom": "20px"
            }
        ),

        # ----------------------------------------------------
        # OWNER
        # ----------------------------------------------------

        html.Label("Owner"),

        dcc.Dropdown(
            id="owner",
            options=[
                {"label": "First Owner", "value": 1},
                {"label": "Second Owner", "value": 2},
                {"label": "Third Owner", "value": 3},
                {"label": "Fourth & Above Owner", "value": 4}
            ],
            placeholder="Select owner",
            style={
                "marginBottom": "20px"
            }
        ),

        # ----------------------------------------------------
        # MILEAGE
        # ----------------------------------------------------

        html.Label("Mileage (kmpl)"),

        dcc.Input(
            id="mileage",
            type="number",
            placeholder="e.g. 24.0",
            style={
                "width": "100%",
                "padding": "10px",
                "marginBottom": "20px"
            }
        ),

        # ----------------------------------------------------
        # ENGINE
        # ----------------------------------------------------

        html.Label("Engine (CC)"),

        dcc.Input(
            id="engine",
            type="number",
            placeholder="e.g. 1248",
            style={
                "width": "100%",
                "padding": "10px",
                "marginBottom": "20px"
            }
        ),

        # ----------------------------------------------------
        # MAX POWER
        # ----------------------------------------------------

        html.Label("Max Power (bhp)"),

        dcc.Input(
            id="max_power",
            type="number",
            placeholder="e.g. 74",
            style={
                "width": "100%",
                "padding": "10px",
                "marginBottom": "20px"
            }
        ),

        # ----------------------------------------------------
        # SEATS
        # ----------------------------------------------------

        html.Label("Seats"),

        dcc.Input(
            id="seats",
            type="number",
            placeholder="e.g. 5",
            style={
                "width": "100%",
                "padding": "10px",
                "marginBottom": "20px"
            }
        ),

        # ----------------------------------------------------
        # BUTTON
        # ----------------------------------------------------

        html.Button(
            "Predict Selling Price",
            id="predict-button",
            n_clicks=0,
            style={
                "width": "100%",
                "padding": "15px",
                "fontSize": "18px",
                "cursor": "pointer"
            }
        ),

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        html.Div(
            id="prediction-output",
            style={
                "marginTop": "30px",
                "textAlign": "center",
                "fontSize": "24px",
                "fontWeight": "bold"
            }
        )
    ]
)


# ============================================================
# 11. CALLBACK
# ============================================================

@app.callback(
    Output(
        "prediction-output",
        "children"
    ),

    Input(
        "predict-button",
        "n_clicks"
    ),

    State("name", "value"),
    State("year", "value"),
    State("km_driven", "value"),
    State("fuel", "value"),
    State("seller_type", "value"),
    State("transmission", "value"),
    State("owner", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("max_power", "value"),
    State("seats", "value")
)
def predict_price(
    n_clicks,
    name,
    year,
    km_driven,
    fuel,
    seller_type,
    transmission,
    owner,
    mileage,
    engine,
    max_power,
    seats
):

    # Don't predict before button is clicked
    if n_clicks == 0:
        return ""

    # Check for missing values
    values = [
        name,
        year,
        km_driven,
        fuel,
        seller_type,
        transmission,
        owner,
        mileage,
        engine,
        max_power,
        seats
    ]

    if any(value is None for value in values):

        return html.Div(
            "Please fill in all fields.",
            style={
                "color": "red"
            }
        )

    # --------------------------------------------------------
    # CREATE DATAFRAME FROM USER INPUT
    # --------------------------------------------------------

    new_car = pd.DataFrame({
        "name": [name],
        "year": [year],
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner],
        "mileage": [mileage],
        "engine": [engine],
        "max_power": [max_power],
        "seats": [seats]
    })

    # --------------------------------------------------------
    # PREDICT CAR PRICE CATEGORY
    # --------------------------------------------------------
    
    processed_car = preprocessor.transform(new_car)

    if issparse(processed_car):
        processed_car = processed_car.toarray()

    scaled_car = scaler.transform(processed_car)

    predicted_price_category = c_model.predict(scaled_car)


    price_ranges = {
        0: (29999, 250000),
        1: (250999, 500000),
        2: (501000, 1000000),
        3: (1019999, 10000000)
    }

    category = int(predicted_price_category[0])
    min_price, max_price = price_ranges[category]

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    return html.Div([
    html.Div(
        "Predicted Selling Price Range"
    ),

    html.Div(
        f"₹{min_price:,.0f} – ₹{max_price:,.0f}",
        style={
            "fontSize": "32px",
            "marginTop": "10px"
        }
    ),

    html.Div(
        f"Price Category: {category}",
        style={
            "fontSize": "18px",
            "marginTop": "10px"
        }
    )
    ])


# ============================================================
# 12. RUN APP
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)