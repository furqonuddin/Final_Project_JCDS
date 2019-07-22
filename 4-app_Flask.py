from flask import Flask, render_template, jsonify, make_response, request, send_from_directory
import json
import requests
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt


app = Flask(__name__)

# =============================================================
@app.route('/')
def welcome():
    return render_template('home.html')

@app.route('/hasil', methods = ['POST', 'GET'])
def hasil():
    try:
        Name = request.form['name'].upper()
        Pregnancies = int(request.form['pregnancies'])
        Glucose = float(request.form['glucose'])
        BloodPressure = float(request.form['bloodPressure'])
        SkinThickness = float(request.form['skinThickness'])
        Insulin = float(request.form['insulin'])
        BMI = float(request.form['bmi'])
        DiabetesPedigreeFunction = float(request.form['diabetesPedigreeFunction'])
        Age = int(request.form['age'])
        
        print(Pregnancies)
        print(Glucose)
        print(type(Glucose))

        df = pd.read_csv('diabetes_new.csv')
        model = joblib.load('modelML')
        
        if Name=="" or Pregnancies=="" or Glucose=="" or BloodPressure=="" or SkinThickness=="" or Insulin=="" or BMI=="" or DiabetesPedigreeFunction=="" or Age=="":
            return render_template('error.html')
        else:
            # prediksi Logistic Regression
            prediksi = model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
            
            if prediksi[0]==1:
                hasil = 'Diabetes'
            else:
                hasil = 'Normal'

            # Grafik pasien dengan nilai rata2 orang normal
            age1 = df[df['Outcome']==0]['Age'].mean()
            bmi1 = df[df['Outcome']==0]['BMI'].mean()
            preg1 = df[df['Outcome']==0]['Pregnancies'].mean()
            blood1 = df[df['Outcome']==0]['BloodPressure'].mean()
            glu1 = df[df['Outcome']==0]['Glucose'].mean()
            insu1 = df[df['Outcome']==0]['Insulin'].mean()
            dpf1 = df[df['Outcome']==0]['DiabetesPedigreeFunction'].mean()
            skn1 = df[df['Outcome']==0]['SkinThickness'].mean()

            x = ['rata2', 'me']
            age2 = [age1, Age]
            bmi2 = [bmi1, BMI]
            preg2 = [preg1, Pregnancies]
            blood2 = [blood1, BloodPressure]
            glu2 = [glu1, Glucose]
            insu2 = [insu1, Insulin]
            dpf2 = [dpf1, DiabetesPedigreeFunction]
            skn2 = [skn1, SkinThickness]

            plt.figure(figsize=(12,6))
            plt.subplot(241)
            plt.bar(x, age2, color=['blue', 'green'])
            plt.title('Umur')

            plt.subplot(242)
            plt.bar(x, bmi2, color=['blue', 'green'])
            plt.title('BMI')

            plt.subplot(243)
            plt.bar(x, preg2, color=['blue', 'green'])
            plt.title('Hamil')

            plt.subplot(244)
            plt.bar(x, blood2, color=['blue', 'green'])
            plt.title('T.Darah')

            plt.subplot(245)
            plt.bar(x, glu2, color=['blue', 'green'])
            plt.title('Glucose')

            plt.subplot(246)
            plt.bar(x, insu2, color=['blue', 'green'])
            plt.title('Insulin')

            plt.subplot(247)
            plt.bar(x, dpf2, color=['blue', 'green'])
            plt.title('DPF')

            plt.subplot(248)
            plt.bar(x, skn2, color=['blue', 'green'])
            plt.title('T.Kulit')

            i = 0
            while True:
                i += 1
                newname = '%s%s.png'%('filename', str(i))
                if os.path.exists('./storage/'+ newname):
                    continue
                plt.savefig('./storage/'+ newname)
                break

            grafik = 'http://localhost:5000/storage/'+ newname

            profil = {
                'name' : Name,
                'hasil' : hasil,
                'grafik': grafik
            }

            return render_template(
                'hasil.html',
                profil = profil
            )

    except:
        return render_template('error.html')        


@app.route('/storage/<namafile>')
def storage(namafile):
    return send_from_directory('./storage',namafile)


# not found display
@app.errorhandler(404)
def tidakfound(error):                                                 
    return make_response('<h1>NOT FOUND (404)</h1>')


if __name__ == '__main__':
    app.run(debug = True) 