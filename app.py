from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Baca data
data = pd.read_csv("dataset_cuaca3 (1).csv")
X = data[['relative_humidity']]
y = data['air_temp']

# Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            humidity = float(request.form["humidity"])
            prediction = round(model.predict([[humidity]])[0], 2)
        except:
            prediction = "Input tidak valid"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
