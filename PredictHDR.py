 # HERE USING THE PKL FILE AS MODEL 

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from PIL import Image
# import base64
# import pickle
# import io
# import numpy as np
# from sklearn import svm


# app = Flask(__name__)
# CORS(app)
# @app.route('/',methods=["GET"])
# def hello():
#     return jsonify({'data' : 'welcome to API'})

 #Load your trained model
 # with open('HandMNIST_SVM.pickle', 'rb') as f:
 #     clf = pickle.load(f)

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the data from the request
#     data = request.get_json()
#     print('the data is',data)
#     try:
#         if 'image' not in data :
#             return jsonify({"error": "No image provided or format is wrong"}),400
    
#     # Assuming the image is sent as a flattened array
        
#         sample_data = data['image'].split(',')[1]
#         print('reached here')
#         image_find = Image.open(io.BytesIO(base64.b64decode(sample_data)))
#         print('reached here22',image_find)
#         # sample_data= np.array(data['image'])
        
#         image_final = image_find.resize((28,28))
#         image_todo = np.array(image_final.convert('L'))

#          # Flattens to 1D array
#         image_data = image_todo.flatten()
#         print('sample data',image_data)
#         if image_data.shape != (784,):
#             return jsonify({"error": "Image data is not is shape"}),503
        
    
#         image_data = image_data.reshape(1, 784)  # Adjust shape if needed
#       
          #prediction = clf.predict(image_data)
#         print('Prediction is',prediction)
#         return jsonify({'prediction': int(prediction[0])})
#     except Exception as e:
#         print(f'Error processing image data',{e})
#         return jsonify({"error": "Failed to process image data"}), 500

# if __name__ == '__main__':
#     app.run(debug=True)



# HERE USING TENSORFLOW MODEL TO GET DATA


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from PIL import Image
# import base64
# import pickle
# import io
# import numpy as np
# from sklearn import svm
# from tensorflow.keras.models import load_model
# Other necessary imports

# app = Flask(__name__)
# CORS(app)
# @app.route('/',methods=["GET"])
# def hello():
#     return jsonify({'data' : 'welcome to API'})

# #Load your trained model
# # with open('HandMNIST_SVM.pickle', 'rb') as f:
# #     clf = pickle.load(f)
# model = load_model("digits_recognition_cnn.h5")

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the data from the request
#     data = request.get_json()
#     # print('the data is',data)
#     try:
#         if 'image' not in data :
#             return jsonify({"error": "No image provided or format is wrong"}),400
    
#     # Assuming the image is sent as a flattened array
        
#         sample_data = data['image'].split(',')[1]
#         print('reached here')
#         image_find = Image.open(io.BytesIO(base64.b64decode(sample_data)))
#         print('reached here22',image_find)
#         # sample_data= np.array(data['image'])
        
#         image_final = image_find.resize((28,28))
#         image_todo = np.array(image_final.convert('L'))/255.0

#          # Flattens to 1D array
#         image_data = image_todo.reshape(1,28,28,1)
#         # print('sample data',image_data)
#         # if image_data.shape != (784,):
#         #     return jsonify({"error": "Image data is not is shape"}),503
        
    
#         # image_data = image_data.reshape(1, 784)  # Adjust shape if needed
#         prediction = model.predict(image_data)

#         # prediction = clf.predict(image_data)
#         print('Prediction is',prediction)
#         predicted_class = np.argmax(prediction[0])
#         return jsonify({'prediction': int(predicted_class)})
#     except Exception as e:
#         print(f'Error processing image data',{e})
#         return jsonify({"error": "Failed to process image data"}), 500

# if __name__ == '__main__':
#     app.run(debug=True)




# model in PB so using this 

import tensorflow as tf
from flask_cors import CORS
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import os
import base64

app = Flask(__name__)
CORS(app)
# Load the TensorFlow model
model_dir = os.path.join(os.getcwd(), 'inception-v3')
model = tf.saved_model.load(model_dir)  # Replace with your model path
# print('model signature is',model.signatures)
infer = model.signatures['serving_default']
# Define a prediction function
# @tf.function
# def predict_fn(input_data):
#     return model(input_data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # print('Received data:', data)

    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    # Decode the Base64 image
    try:
        image_data = data['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        # Resize and preprocess the image
        # image = image.resize((28, 28))
        #  image_data = np.array(image.convert('L')) / 255.0  # Normalize to [0, 1]
        # Reshape for the model
        # image_data = image_data.reshape(1, 28, 28, 1)  # Shape: (1, 28, 28, 1)
        # predictions = predict_fn(tf.convert_to_tensor(image_data, dtype=tf.float32))
        # predicted_class = tf.argmax(predictions, axis=1).numpy()[0]  # Get the predicted class index


        image = image.resize((299,299))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_data = np.array(image) / 255.0
        image_data = image_data.reshape(1,299,299,3)
       
        predictions = infer(tf.convert_to_tensor(image_data,dtype=tf.float32))
        # predictions = model(image_data)
        print('predictions are ',predictions)
        predicted_class = tf.argmax( predictions['logits'],axis=1).numpy()[0]
        # Make prediction
        
        print('Predicted digit is', predicted_class)
        return jsonify({'prediction': int(predicted_class)})
    except Exception as e:
        print(f'Error processing image data: {e}')
        return jsonify({'error': 'Failed to process image data'}), 400

if __name__ == '__main__':
    app.run(debug=True)
