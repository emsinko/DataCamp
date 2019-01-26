# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 21:56:12 2019

@author: marku
"""

from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"


#$ pip install Flask
#$ FLASK_APP=hello.py flask run
#* Running on http://localhost:5000/