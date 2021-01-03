# Importing libraries
from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from datetime import datetime
from glob import glob
import os

# Work flow
# ===================================
# 1. Extracting data from the form
#    1. Single Record
#    2. Single File
#    3. Batch Files
# 2. Preprocessing data
#    1. Conversion Dates into ordinals
#    2. Extracting New feature
#    3. Scaling Data and File
# 3. Predicting Result
#    1. Predicting Record
#    2. Predicting File
# 4. Update data into Data-base
# ===================================

# Initialising flask object
app = Flask(__name__)

# HOME PAGE --------------------------------------------------------------------------------------
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Home page
@app.route("/")
def home():
    return render_template("home_page.html")

@app.route("/home_type", methods = ["POST"])
def home_type():
    input_type = request.form['input_type']
    if input_type == 'Single_Data':
        return render_template('home_rf_model_feature_10_single_data.html')
    elif input_type == 'Single_File':
        return render_template('home_rf_model_feature_10_single_file.html')
    elif input_type == 'Batch_Files':
        return render_template("home_rf_model_feature_10_batch_files.html")

# Single Student Details :  ---------------------------------------------------------------------
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@app.route("/predict_rf_model_feature_10_single_data", methods = ["POST"])
def predict_rf_model_feature_10_single_data():
    # ======================================================================================
    # Input Example:
    # [start_date, end_date, access, discussion, navigate, page_close, problem, video, wiki]
    # ["14-12-2020", "19-12-2020", 74, 14, 12, 22, 2, 1, 2]
    # ======================================================================================
    # --------------------- Extracting values from the form  ---------------------
    cols = ['start_date', 'end_date', 'access', 'discussion', 'navigate', 'page_close', 'problem', 'video', 'wiki']
    in_features = [[request.form[col] for col in cols]]
    df = pd.DataFrame(np.array(in_features), columns=cols)    # Creating Data Frame with input values
    X = df.copy()   # Copying Input values for purpose of exporting into MongoDB
    # -------------------------- Preprocessing the data --------------------------
    from preprocessing import Preprocessing
    pre_process = Preprocessing()
    df = pre_process.processing(df)
    # -------------------------- Predicting the result --------------------------
    X['result'] = predict_df(df)
    # ----------------------- Storing the data in MongoDB -----------------------
    from database import Database
    db = Database()
    DbMessage = db.update_record(X)
    # ---------------------------- Display the result ----------------------------
    if X['result'][0] == "0":
        result = "0 - Student will not Drop from the course"
    else:
        result = "1 - Student will Drop from the course"

    return render_template("result_page.html", result=result, DbMessage=DbMessage, path="None")

# Using Single File : ---------------------------------------------------------------------------
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@app.route("/predict_rf_model_feature_10_single_file", methods=["POST"])
def predict_rf_model_feature_10_single_file():

    # ------------------- Loading the data frame from and save in specific path -------------------
    in_file = request.files['in_file']
    in_file.save(os.path.join('Data/Single_File', in_file.filename))

    # ---------------------- Loading data from specific path into data frame ----------------------
    df = pd.read_csv(os.path.join('Data/Single_File', in_file.filename))
    path = "C:\\Users\pc\Desktop\Data science\02 INEURON\Projects\S_Dropout\Data\Single_File"

    # ---------------------------------------------------------------------------------------------
    if df.columns[-1]=='result':
        return render_template("result_page.html", data="This data was already Predicted", path=path)
    else:
        X = df.copy()  # Copying Input values for purpose of exporting into MongoDB
        cols = ['start_date', 'end_date', 'access', 'discussion', 'navigate', 'page_close', 'problem', 'video', 'wiki']
        df = df[cols]
        # -------------------------- Preprocessing the data --------------------------
        from preprocessing import Preprocessing
        preprocess = Preprocessing()
        df = preprocess.processing(df)
        # -------------------------- Predicting the result --------------------------
        X['result'] = predict_df(df)
        # ------------------ Storing the result into specific path ------------------
        X.to_csv(os.path.join('Data/Single_File', in_file.filename), index=False)
    return render_template("result_page.html", result="Successfully Predicted", path=path)

# Using Batch Files : ---------------------------------------------------------------------------
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@app.route("/predict_rf_model_feature_10_batch_files", methods=["POST"])
def predict_rf_model_feature_10_batch_files():
    # Loading the path from the source
    MyCsvDir = request.form['path']
    CsvFiles = glob(os.path.join(MyCsvDir, '*.csv'))
    skipped_files = []
    predicted_files = []
    for i in range(len(CsvFiles)):
        df = pd.read_csv(CsvFiles[i])   # Creating Data Frame
                     # Calling function for predicting
        # file_name =  CsvFiles[i].split(MyCsvDir)[1]
        if df.columns[-1] == 'result':
            skipped_files.append(CsvFiles[i].split(MyCsvDir)[1])
        else:
            predicted_files.append(CsvFiles[i].split(MyCsvDir)[1])
            X = df.copy()  # Copying Input values for purpose of exporting into MongoDB
            cols = ['start_date', 'end_date', 'access', 'discussion', 'navigate', 'page_close', 'problem', 'video', 'wiki']
            df = df[cols]
            # -------------------------- Preprocessing the data --------------------------
            from preprocessing import Preprocessing
            preprocess = Preprocessing()
            df = preprocess.processing(df)
            # -------------------------- Predicting the result --------------------------
            X['result'] = predict_df(df)
            # ----------------------- Storing the resultant files -----------------------
            X.to_csv(CsvFiles[i], index=False)

    return render_template("result_page.html", result=f"Skipped Files: {skipped_files} \n Predicted Files: {predicted_files}", path=MyCsvDir)

# Function to predict the result for one data frame ---------------------------------------------
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def predict_df(df):
    model = pickle.load(open("pkl_rf_model_feature_10.pkl", "rb"))
    # Model Prediction
    pred_val = model.predict(df)
    return pred_val

if __name__ == "__main__":
    app.run(debug=True, port=5101)
