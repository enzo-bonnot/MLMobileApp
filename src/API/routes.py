from flask import Flask, jsonify
from src.API import app


@app.route("/", methods=['GET'])
def home():
    return 'Home route'
