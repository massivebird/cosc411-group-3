from flask import Flask, url_for

app = Flask(__name__)

def file_contents(pathname):
    f = open(pathname, "r")
    return f.read()

@app.route("/")
def main():
    return file_contents("index.html")

@app.route("/style.css")
def styles():
    f = open("style.css", "r")
    return f.read()

with app.test_request_context():
    url_for('static', filename='style.css')
