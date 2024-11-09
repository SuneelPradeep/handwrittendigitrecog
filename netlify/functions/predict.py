

from PIL import Image
from joblib import load
import sys
import random
import base64
import pickle
import io
import numpy as np
from sklearn import svm


with open('HandWritten_Digit_Recognition', 'rb') as f:
    clf = pickle.load(f)


def handler(event, context):
    try:
    data = request.get_json()
    try:
        if 'image' not in data:
            return {"statusCode": 400, "body": json.dumps({"error": "No image provided or format is wrong"})}

     
        sample_data = data['image'].split(',')[1]
        image_find = Image.open(io.BytesIO(base64.b64decode(sample_data)))
        image_final = image_find.resize((28,28))
        image_final = image_final.convert("L")
        image_todo = np.array(image_final)
        image_final.save('resized_image.png')
        image_todo = image_todo /255.0
        image_data = image_todo.flatten()
        image_data = image_todo.flatten()
        if image_data.shape != (784,):
            return {"statusCode": 503, "body": json.dumps({"error": "Image data is not in shape"})}

        image_data = image_data.reshape(1, 784) 
        prediction = clf.predict(image_data)
        prediction_probability = clf.decision_function(image_data)
        confidence = np.max(prediction_probability)
        
  
        print('Prediction is',prediction)
        return {
            "statusCode": 200,
            "body": json.dumps({
                'prediction': int(prediction[0]),
                "confidence": round(float(confidence), 2)
            })
        }
    except Exception as e:
        print(f'Error processing image data',{e})
        return jsonify({"error": "Failed to process image data"}), 500
