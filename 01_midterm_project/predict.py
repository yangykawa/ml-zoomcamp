import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'xgb_final_model.bin'

with open(model_file, 'rb') as f_in:
    dv, model= pickle.load(f_in)

app = Flask('loan_status')
@app.route('/predict', methods=['POST'])

def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict(X) 
    y_pred_prob = model.predict_proba(X)[:, 1]

    result = {
        'loan_status': bool(y_pred),
        'loan_predict_probability': float(y_pred_prob[0])
    }


    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)