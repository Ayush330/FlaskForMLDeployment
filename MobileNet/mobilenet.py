# Copyright 2021 Ayush Kumar Anand
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#importing all the necessary modules
import numpy as np
import tensorflow as tf
import cv2 as cv

def predict_the_image_class():
    #importing the interpreter
    interpreter = tf.lite.Interpreter(model_path="model.tflite")

    #allocating tensors to the interpreter(Because tflite uses tensors)
    interpreter.allocate_tensors()

    #getting the input and output paramaters of the tflite model, helps in getting familiar with the model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    #print(input_details)
    #print(output_details)

    #finding the image from folder(same as the folder in which the code is residing) and reading it(converting to numpy array)
    image= cv.imread("server_image.jpg", cv.IMREAD_COLOR)

    #resizing the image size as per the model requirements(Can be obtained from input details)
    image = cv.resize(image,(224,224))


    #converting the data type of the image array as per thr model requirements(Also can be obtained from input details)
    image = image.astype(np.uint8)

    #print(image.shape,image.dtype)

    #including the batch size to our image array
    image = image[np.newaxis,:]
    #print(image.shape)
    #print(image[0])


    #setting the tensors to the mosel
    interpreter.set_tensor(input_details[0]['index'], image)

    #invoking the model here
    interpreter.invoke()

    #getting the output predicted by the model as tensors
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #print(output_data[0])
    #print(np.amax(output_data[0]))

    #Output is a 2D array, so we are finding the maximum element(Each element represents the probability of the class)
    result = np.where(output_data[0] == np.amax(output_data[0]))
    #print(result[0][0])
    #print(output_data[0][818])

    #Here we are storing all the possible classes that the model can predict in a list
    data = []
    file1 = open('label.txt', 'r')
    Lines = file1.readlines()
    for line in Lines:
        data.append(line)
    file1.close()  

    #printing the predicted class  
    print("The predicted class is: ",data[result[0][0]])
    return data[result[0][0]]