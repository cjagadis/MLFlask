import numpy as np
from flask import Flask, abort, jsonify, request
import pandas as pd
import os
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
import pickle


app = Flask(__name__)

@app.route("/regPredict", methods=['POST'])
def regPredict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            years_of_experience = float(data["yearsOfExperience"])

            lin_reg = joblib.load("./linear_regression_model.pkl")
        except ValueError:
            return jsonify("Please enter a number.")

        return jsonify(lin_reg.predict(years_of_experience).tolist())

@app.route("/regRetrain", methods=['POST'])
def regRetrain():
    if request.method == 'POST':
        data = request.get_json()

        try:
            training_set = joblib.load("./training_data.pkl")
            training_labels = joblib.load("./training_labels.pkl")

            df = pd.read_json(data)

            df_training_set = df.drop(["Salary"], axis=1)
            df_training_labels = df["Salary"]

            df_training_set = pd.concat([training_set, df_training_set])
            df_training_labels = pd.concat([training_labels, df_training_labels])

            new_lin_reg = LinearRegression()
            new_lin_reg.fit(df_training_set, df_training_labels)

            os.remove("./linear_regression_model.pkl")
            os.remove("./training_data.pkl")
            os.remove("./training_labels.pkl")

            joblib.dump(new_lin_reg, "linear_regression_model.pkl")
            joblib.dump(df_training_set, "training_data.pkl")
            joblib.dump(df_training_labels, "training_labels.pkl")

            lin_reg = joblib.load("./linear_regression_model.pkl")
        except ValueError as e:
            return jsonify("Error when retraining - {}".format(e))

        return jsonify("Retrained model successfully.")


@app.route("/currentDetails", methods=['GET'])
def current_details():
    if request.method == 'GET':
        try:
            lr = joblib.load("./linear_regression_model.pkl")
            training_set = joblib.load("./training_data.pkl")
            labels = joblib.load("./training_labels.pkl")

            return jsonify({"score": lr.score(training_set, labels),
                            "coefficients": lr.coef_.tolist(), "intercepts": lr.intercept_})
        except (ValueError, TypeError) as e:
            return jsonify("Error when getting details - {}".format(e))


@app.route('/rforest', methods=['POST'])
def rforestPredict():
    my_random_forest = pickle.load(open("iris_rfc.pk1","rb"))
    data = request.get_json(force=True)
    predict_request = [data['sl'],data['sw'],data['pl'],data['pw']]
    predict_request = np.array(predict_request).reshape(1,4)
    print("printing predict request")
    print(predict_request.shape)
    print(predict_request)
    y_hat = my_random_forest.predict(predict_request)
    print(y_hat[0])
    output = [y_hat[0]]
    
    print(output)
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port=5000,debug=True)
