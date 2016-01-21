from flask import Flask, request, redirect, url_for, render_template, session
import cPickle as pickle
import pandas as pd
import os, random
from algorithm import plot_cloud, extract_data

#get model
with open('model.pkl') as f:
     model = pickle.load(f)
#get scaler
with open('scaler.pkl') as f:
     scaler = pickle.load(f)

app = Flask(__name__)
app.secret_key = 'not_so_secret'
#app.config.from_object('config')
# home page
@app.route('/')
def home():
    return render_template('homepage.html')

#redirect(url_for('submit'))

@app.route('/game/<level>')
def game(level):
    #randomly select filename
    file = random.choice(os.listdir("../data/test set"))
    filename = "../data/test set/" + file
    ans_dict = {'cr':'car','cv':'convertible','pl':'airplane','mt':'motorcycle',
                'tr':'train','hl':'helicopter',
                '0':'motorcycle','1':'car','2':'airplane','3':'convertible','4':'train',
                '5':'helicopter'}
    session['answer'] = ans_dict[file[:2]]
    features = scaler.transform(extract_data(filename, sampling = level))
    predict = model.predict(features)[0]
    session['prediction'] = ans_dict[predict]
    front_view, side_view, top_view = plot_cloud(filename, file, sampling = level)
    return render_template('game_page.html', level = level,
                            front_view = front_view, side_view = side_view,
                            top_view = top_view)
                            
@app.route('/answer/<response>/<level>')
def answer(level,response):
    response_dict = {'car': 'car','con':'convertible', 'heli':'helicopter',
                'pl':'airplane', 'motor': 'motorcycle', 'train':'train'}
    guess = response_dict[response]
    answer = session['answer']
    prediction = session['prediction']
    if (guess == answer) & (prediction == answer):
        msg = 'Correct! You matched the algorithm.'
    elif (guess == answer) & (prediction != answer):
        msg = 'Correct! And you beat the algorithm!'
    elif (guess != answer) & (prediction == answer):
        msg = 'Nope! The algorthim did better, try again!'
    elif (guess != answer) & (prediction != answer):
        msg = 'Nope! No one got this right. Maybe it was too hard.'
    # solution =
    return render_template('answer_page.html', level = level, answer = answer,
                            guess = guess, prediction = prediction, msg = msg)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
