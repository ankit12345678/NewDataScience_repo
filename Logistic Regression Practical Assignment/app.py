from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle

app = Flask(__name__)

model = pickle.load(open('Logistic_Regression.pkl', 'rb'))


@app.route("/", methods=['GET', 'POST'])
@cross_origin()
def home_page():
    try:
        return render_template('index.html')
    except Exception as e:
        raise Exception(f"(home) - Could Not Find The index.html Page \n" + str(e))


@app.route("/report", methods=['GET', 'POST'])
@cross_origin()
def report():
    try:
        return render_template('affaris_report.html')
    except Exception as e:
        raise Exception(f"(report) - Could Not Find The affaris_report.html Page \n" + str(e))


@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():

    if request.method == "POST":

        try:
            Intercept = int(request.form['Intercept'])

            rate_marriage = request.form['rate_marriage']

            age = int(request.form['age'])

            yrs_married = int(request.form['yrs_married'])

            children = int(request.form['children'])

            religious = request.form['religious']

            educ = request.form['educ']

            occupation = request.form['occupation']
            if occupation == 1:
                occ_2 = 0
                occ_3 = 0
                occ_4 = 0
                occ_5 = 0
                occ_6 = 0
            elif occupation == 2:
                occ_2 = 1
                occ_3 = 0
                occ_4 = 0
                occ_5 = 0
                occ_6 = 0
            elif occupation == 3:
                occ_2 = 0
                occ_3 = 1
                occ_4 = 0
                occ_5 = 0
                occ_6 = 0
            elif occupation == 4:
                occ_2 = 0
                occ_3 = 0
                occ_4 = 1
                occ_5 = 0
                occ_6 = 0
            elif occupation == 5:
                occ_2 = 0
                occ_3 = 0
                occ_4 = 0
                occ_5 = 1
                occ_6 = 0
            else:
                occ_2 = 0
                occ_3 = 0
                occ_4 = 0
                occ_5 = 0
                occ_6 = 1

            occupation_husb = request.form['occupation_husb']
            if occupation_husb == 1:
                occ_husb_2 = 0
                occ_husb_3 = 0
                occ_husb_4 = 0
                occ_husb_5 = 0
                occ_husb_6 = 0

            elif occupation == 2:
                occ_husb_2 = 1
                occ_husb_3 = 0
                occ_husb_4 = 0
                occ_husb_5 = 0
                occ_husb_6 = 0

            elif occupation == 3:
                occ_husb_2 = 0
                occ_husb_3 = 1
                occ_husb_4 = 0
                occ_husb_5 = 0
                occ_husb_6 = 0

            elif occupation == 4:
                occ_husb_2 = 0
                occ_husb_3 = 0
                occ_husb_4 = 1
                occ_husb_5 = 0
                occ_husb_6 = 0

            elif occupation == 5:
                occ_husb_2 = 0
                occ_husb_3 = 0
                occ_husb_4 = 0
                occ_husb_5 = 1
                occ_husb_6 = 0

            else:
                occ_husb_2 = 0
                occ_husb_3 = 0
                occ_husb_4 = 0
                occ_husb_5 = 0
                occ_husb_6 = 1

            prediction = model.predict([[Intercept, rate_marriage, age, yrs_married, children, religious, educ, occ_2,
                                         occ_3, occ_4, occ_5, occ_6, occ_husb_2, occ_husb_3, occ_husb_4, occ_husb_5,
                                         occ_husb_6]])
            print(prediction)
            if prediction[0] == 0:
                result = 'The Women Has Not An Extra Marital Affair'
            else:
                result = 'The Women Has An Extra Marital Affair'
            
            return render_template('index.html', prediction_text=result)

        except Exception as e:
            raise Exception(f"(predict) - Their Is Something Wrong About Predict \n" + str(e))

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
