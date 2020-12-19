# Importing libraries
from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

# Initialising flask object
app = Flask(__name__)

# HOME PAGE --------------------------------------------------------------------------------------
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Creating Home page
@app.route("/")
def home():
    return render_template("home_page.html")

# PREDICTIONS using rfc_mim.pkl -----------------------------------------------------------------
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@app.route("/home_rfc_mim")
def home_rfc_mim():
    return render_template("home_rfc_mim.html")

@app.route("/predict_rfc_mim", methods=["POST"])
def predict_rfc_mim():
    # ======================================================================================
    # Input Example:
    # [start_date, end_date, 'access', 'discussion', 'navigate', 'problem', 'wiki', 'present_days', 'effective_time', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'holidays', 'course_enroll', 'user_enroll', 'course_drop_rate']
    # ["14-12-2020", "19-12-2020", 74, 14, 12, 22, 2, 4, 8.62222, 1, 1, 1, 0, 1, 0, 0, 0, 1481, 2, 0.823991]
    # ======================================================================================

    # Initialising Pickle file
    model = pickle.load(open("pkl_rfc_mim.pkl", "rb"))

    start_date = request.form['start_date']
    end_date = request.form['end_date']
    duration_in_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days+1
    cols = ['access', 'discussion', 'navigate', 'problem', 'wiki', 'present_days', 'effective_time', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'holidays', 'course_enroll', 'user_enroll', 'course_drop_rate']
    arr = [float(request.form[col]) for col in cols]
    in_features = [duration_in_days] + arr

    pred_val = model.predict(np.array([in_features]))
    if pred_val[0] == 1:
        result = "1 :- Student will drop from the course"
    else:
        result = "0 :- Student will not drop from the course"
    return render_template("result_page.html", data=result)

# PREDICTIONS using rf_model_feature_10.pkl ------------------------------------------------------
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@app.route("/home_rf_model_feature_10")
def home_rf_model_feature_10():
    return render_template("home_rf_model_feature_10.html")

@app.route("/predict_rf_model_feature_10", methods = ["POST"])
def predict_rf_model_feature_10():
    # ======================================================================================
    # Input Example:
    # [start_date, end_date, access, discussion, navigate, page_close, problem, video, wiki]
    # ["14-12-2020", "19-12-2020", 74, 14, 12, 22, 2, 1, 2]
    # ======================================================================================

    # Initialising Pickle file
    model = pickle.load(open("pkl_rf_model_feature_10.pkl", "rb"))

    # Extracting Start Date and End Date
    start_date = pd.to_datetime(request.form["start_date"])
    end_date = pd.to_datetime(request.form["end_date"])

    present_days = (end_date - start_date).days + 1
    start_day = start_date.toordinal()
    end_day = end_date.toordinal()

    cols = ['access', 'discussion', 'navigate', 'page_close', 'problem', 'video', 'wiki']
    in_features = [int(request.form[col]) for col in cols]
    in_features = [start_day, end_day] + in_features + [present_days]

    # Standardisation of data
    scale = StandardScaler()
    data = pd.DataFrame(in_features)
    scaled_values = scale.fit_transform(data).reshape(1, -1)

    # Model Prediction
    pred_val = model.predict(scaled_values)

    if pred_val[0] == "0":
        result = "0 - Student will not Drop from the course"
    else:
        result = "1 - Student will Drop from the course"

    return render_template("result_page.html", data=result)

if __name__ == "__main__":
    app.run(debug=True, port=5101)
