from flask import Flask, render_template, request, jsonify
import mobilenet as Model
import numpy as np
import cv2 as cv


app = Flask(__name__)




@app.route('/image_upload',methods=['POST'])
def home2():
    #r = request
    print("Files: ",request.files)
    x = request.files
    y = x["Image"].read()
    
    #converting the image sent in string to numpy array
    npimg = np.fromstring(y, np.uint8)
    
    #decoding the image uploaded
    image = cv.imdecode(npimg, cv.IMREAD_COLOR)
    
    
    #saving the image to the Deploy folder
    cv.imwrite('server_image.jpg', image)
    
    #running the tflite model here
    Class = Model.predict_the_image_class()
    print(Class)
    
    return jsonify({"Predicted_Class":Class})

