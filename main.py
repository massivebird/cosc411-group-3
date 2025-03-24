import decision_tree
from flask import Flask, url_for, render_template, request

model = decision_tree.model();

app = Flask(__name__)

def file_contents(pathname):
    f = open(pathname, "r")
    return f.read()

@app.route("/")
def main():
    return file_contents("index.html")

# https://flask.palletsprojects.com/en/stable/quickstart/#the-request-object
@app.route("/result")
def result():
    career = model.predict([[
        request.args.get("open"),
        request.args.get("cons"),
        request.args.get("extra"),
        request.args.get("agree"),
        request.args.get("neuro"),
        request.args.get("numer"),
        request.args.get("spatial"),
        request.args.get("percep"),
        request.args.get("abstr"),
        request.args.get("verb"),
    ]])

    return render_template('result.html', result=career[0])

@app.route("/style.css")
def styles():
    f = open("style.css", "r")
    return f.read()

with app.test_request_context():
    url_for('static', filename='style.css')
    url_for('static', filename='favicon.png')
