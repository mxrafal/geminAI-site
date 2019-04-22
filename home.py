from flask import Flask, render_template, request
from mrmodel import get_model_api
from textgenrnn import textgenrnn  
import os
import re


model = get_model_api()

# mrmodel = MrModel()

app = Flask(__name__)
def init():
    global model 
    model = textgenrnn(weights_path='zodiac_weights.hdf5',
                        vocab_path='zodiac_vocab.json',
                        config_path='zodiac_config.json')

@app.route("/")
def home():
    return render_template("geminai_web.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("geminai_contact.html")

@app.route("/predict")
def predict():
    #selected_sign = request.args.get('type')
    #print(selected_sign)
    selected_sign = request.args.get('sign')
    #print(selected_sign)
    # res = mrmodel.generate()
    res = model()
    signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    for sign in signs:
        res = res.replace(sign, selected_sign)
    return render_template("zodiac_predict.html", horoscope=res)

if __name__ == "__main__":
    if 'ENV' in os.environ:
        if os.environ['ENV'] == 'production':
            app.debug = True
            port = int(os.environ.get("PORT", 5000))
            app.run(host='0.0.0.0', port=port)
    else:
        app.run(debug=True, threaded=False)
