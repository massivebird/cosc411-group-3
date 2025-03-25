import random_forest
from flask import Flask, url_for, render_template, request

model = random_forest.model();

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
    linguistic = request.args.get("linguistic")
    musical = request.args.get("musical")
    bodily = request.args.get("bodily")
    logicalMath = request.args.get("logicalMath")
    spatialVis = request.args.get("spatialVis")
    interpersonal = request.args.get("interpersonal")
    intrapersonal = request.args.get("intrapersonal")
    naturalist = request.args.get("naturalist")

    # Normalization
    linguistic = (int(linguistic)-5)/15
    musical = (int(musical)-5)/15
    bodily = (int(bodily)-5)/15
    logicalMath = (int(logicalMath)-5)/15
    spatialVis = (int(spatialVis)-5)/15
    interpersonal = (int(interpersonal)-2)/18
    intrapersonal = (int(intrapersonal)-5)/15
    naturalist = (int(naturalist))/20

    career = model.predict([[
        linguistic,
        musical,
        bodily,
        logicalMath,
        spatialVis,
        interpersonal,
        intrapersonal,
        naturalist,
    ]])

    return render_template('result.html', result=career[0])

@app.route("/style.css")
def styles():
    f = open("style.css", "r")
    return f.read()

with app.test_request_context():
    url_for('static', filename='style.css')
    url_for('static', filename='favicon.png')
