from flask import Flask,render_template,session,url_for,redirect
import numpy as np 
from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField
from wtforms.validators import NumberRange
from tensorflow.keras.models import load_model
import joblib 


app = Flask(__name__)
app.config['SECRET_KEY'] = "algumaKey"


class FlowerForm(FlaskForm):
    SepalLengthCm = StringField('Sepal Length', validators=[NumberRange(min=0, max=10)])
    SepalWidthCm = StringField('Sepal Width', validators=[NumberRange(min=0, max=10)])
    PetalLengthCm = StringField('Petal Length', validators=[NumberRange(min=0, max=10)])
    PetalWidthCm = StringField('Petal Width', validators=[NumberRange(min=0, max=10)])
    submit = SubmitField('Analyze')
    
    

flower_model = load_model('DPL\API\iris_model.h5')
flower_scaler = joblib.load('DPL\API\iris_scaler.pkl')

def return_prediction(model, scaler, sample_json):
    
    classes = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    
    s_len = sample_json['SepalLengthCm']
    s_wid = sample_json['SepalWidthCm']
    p_len = sample_json['PetalLengthCm']
    p_wid = sample_json['PetalWidthCm']
    
    flower = [[s_len, s_wid, p_len, p_wid]]
    
    flower = scaler.transform(flower)
    
    class_ind = model.predict(flower)
    class_x_ind = np.argmax(class_ind, axis=1)
    
    return classes[class_x_ind][0]

@app.route('/prediction')
def prediction():
    content = {}
    
    content['SepalLengthCm'] = float(session['SepalLengthCm'])
    content['SepalWidthCm'] = float(session['SepalWidthCm'])
    content['PetalLengthCm'] = float(session['PetalLengthCm'])
    content['PetalWidthCm'] = float(session['PetalWidthCm'])
    
    results = return_prediction(model=flower_model, scaler=flower_scaler, sample_json=content)
    return render_template('prediction.html', results=results)

@app.route('/')
def index():
    form = FlowerForm()

    if form.validate_on_submit():
        session['SepalLengthCm'] = form.SepalLengthCm.data
        session['SepalWidthCm'] = form.SepalWidthCm.data
        session['PetalLengthCm'] = form.PetalLengthCm.data
        session['PetalWidthCm'] = form.PetalWidthCm.data
        
        return redirect(url_for('prediction'))
    return render_template('home.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)