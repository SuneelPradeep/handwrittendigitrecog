#  HERE USING THE PKL FILE AS MODEL 

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from joblib import load
import sys
import random
import base64
import pickle
import io
import numpy as np
from sklearn import svm


app = Flask(__name__)
CORS(app)
@app.route('/',methods=["GET"])
def hello():
    return jsonify({'data' : 'welcome to API'})

#  Load your trained model
with open('netlify/functions/HandWritten_Digit_Recognition', 'rb') as f:
    clf = pickle.load(f)
# pickle_in = open('HandWritten_Digit_Recognition','rb')
# clf = load('Handdd.joblib')
# clf = pickle.load(pickle_in)
print('model loaded successfully')

# old_stdout = sys.stdout
# log_file = open("pic.log","w")
# sys.stdout = log_file  

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json()
    # print('the data is',data)
    try:
        if 'image' not in data :
            return jsonify({"error": "No image provided or format is wrong"}),400
    
    # Assuming the image is sent as a flattened array
        
        sample_data = data['image'].split(',')[1]
        # print('reached here')
        image_find = Image.open(io.BytesIO(base64.b64decode(sample_data)))
        image_final = image_find.resize((28,28))
        # print('final image after resize to 28 ,28 ',image_final)
        # image_todo = image_final
        image_final = image_final.convert("L")
        image_todo = np.array(image_final)
        image_final.save('resized_image.png')
        image_todo = image_todo /255.0
        image_data = image_todo.flatten()
        # print('sample data pic final image',image_data)
        image_data = image_todo.flatten()
        # print('sample data pic final image',image_data)
        # image_data.save('flattened_pic.png')
        if image_data.shape != (784,):
            return jsonify({"error": "Image data is not is shape"}),503
        
    
        image_data = image_data.reshape(1, 784)  # Adjust shape if needed
        # print('the reshape image is',image_data)
        prediction = clf.predict(image_data)
        prediction_probability = clf.decision_function(image_data)
        confidence = np.max(prediction_probability)
        # print('Prediction is',prediction)
        # print('reached here22',image_find)
        # sample_data= np.array(data['image'])
        # image_find = Image.eval(image_find, lambda x: 255 - x)
        
        
        # image_todo = Image.eval(image_todo, lambda x: 255 - x)
       
        # image_todo = image_final
        
        # image_find.save('afterConvertingpic.png')
        # print('image todo is',image_todo)
        # image_todo = image_todo /255.0
         # Flattens to 1D array
        # image_data = image_todo / 255.0
        # threshold =200
        # binary_image = np.where(image_todo[:,:,0] > threshold,1,0).astype(np.float32)
        # image_todo = binary_image
        # image_data = image_todo.flatten()
        # print('sample data pic final image',image_data)
        # # image_data.save('flattened_pic.png')
        # if image_data.shape != (784,):
        #     return jsonify({"error": "Image data is not is shape"}),503
        
    
        # image_data = image_data.reshape(1, 784)  # Adjust shape if needed
        # print('the reshape image is',image_data)
        # prediction = clf.predict(image_data)
        print('Prediction is',prediction)
        return jsonify({'prediction': int(prediction[0]),
        "confidence" : round(float(confidence),2) })
    except Exception as e:
        print(f'Error processing image data',{e})
        return jsonify({"error": "Failed to process image data"}), 500

from MNIST_Dataset_Loader.mnist_loader import MNIST
data = MNIST('./MNIST_Dataset_Loader/dataset/')
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)

@app.route('/random-predict',methods=["GET"])
def random_predict():
    # random index
    random_index = random.randint(0,len(test_img)-1)

    # getting that image & label 
    original_image = test_img[random_index]
    original_label = test_labels[random_index]

    # preparing for prediction
    image_data = original_image.reshape(1,784)
    prediction = clf.predict(image_data)
    predicted_label = int(prediction[0])

    # converting image to display
    two_d_image = (np.reshape(original_image, (28,28))* 255).astype(np.uint8)

    # save to visual
    img = Image.fromarray(two_d_image)
    img.save('Random_Predicted_Image.png')

    return jsonify({ "original_label" : int(original_label),
    "predicted_label" : predicted_label,
    "random_index" : random_index
    })



if __name__ == '__main__':
    app.run(debug=True)

