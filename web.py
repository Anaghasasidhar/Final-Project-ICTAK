import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)
scaler = StandardScaler()
model=pickle.load(open('model.pkl','rb'))
le=pickle.load(open('le.pkl','rb'))
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_eligible():
    loan_amnt = request.form.get("loan_amnt")
    int_rate = request.form.get("int_rate")
    annual_inc = request.form.get("annual_inc")
    dti = request.form.get("dti")
    open_acc = request.form.get("open_acc")
    pub_rec = request.form.get("pub_rec")
    revol_bal = request.form.get("revol_bal")
    mort_acc = request.form.get("mort_acc")

    input_data = np.array([[loan_amnt, int_rate, annual_inc, dti, open_acc, pub_rec, revol_bal, mort_acc]]).reshape(1,8)

    # Scale the input data
    scaled_data = scaler.transform(input_data)

 

    result = model.predict(scaled_data)

    if (result[0] == 1) or (result[0] == 'Fully Paid'):
        result = "Likely to pay off the loan"
    else:
        result = "Likely to default the loan"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
