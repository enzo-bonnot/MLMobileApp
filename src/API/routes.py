from flask import Flask, jsonify, request
from src.API import app
import cv2
import numpy as np
from src.API.utils import preproc as pp


@app.route("/predict", methods=['GET'])
def home():
    input_size = (1024, 128, 1)
    file = request.files['image']
    img = np.fromfile(file, np.uint8)
    image = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    print(np.asarray(image).shape)
    image = pp.preprocess(image, input_size=input_size)
    norm = pp.normalization([image])

    predicts, probabilities = app.model.predict(norm, ctc_decode=True)
    predicts = [[app.tokenizer.decode(x) for x in y] for y in predicts]

    res = predicts[0][0]
    return res
