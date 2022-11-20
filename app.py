#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import jsonify
import requests
import joblib
import numpy as np
import sklearn
app = Flask(__name__)
model = joblib.load("pipeline_jlib", 'r+')
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        NewCreditCustomer = request.form['NewCreditCustomer']
        if(NewCreditCustomer==True):
            NewCreditCustomer=1
                
        else:
            NewCreditCustomer=0      
        VerificationType=float(request.form['VerificationType'])
        LanguageCode=int(request.form['LanguageCode'])
        Gender=float(request.form['Gender'])
        Education=int(request.form['Education'])
        MaritalStatus=int(request.form['MaritalStatus'])
        EmploymentStatus=int(request.form['EmploymentStatus'])
        EmploymentDurationCurrentEmployer = int(request.form['EmploymentDurationCurrentEmployer'])
        OccupationArea=int(request.form['OccupationArea'])
        
        Restructured=request.form['Restructured']
        if(Restructured==True):
            Restructured=1
                
        else:
            Restructured=0
            
        CreditScoreEsMicroL=int(request.form['CreditScoreEsMicroL'])
        Status=request.form['Status']


        prediction=model.predict([[NewCreditCustomer,VerificationType,LanguageCode,Gender,Education,MaritalStatus,EmploymentStatus,EmploymentDurationCurrentEmployer,OccupationArea,Restructured,CreditScoreEsMicroL,Status,PC1,PC2,PC3,PC4,PC5]])
        output=round(prediction[0],1)
        if output<0:
            return render_template('index.html',prediction_texts="defaulted")
        else:
            return render_template('index.html',prediction_text="Not defaulted")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

