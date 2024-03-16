from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import base64
import io
from string import Template
from ultralytics import YOLO
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain  

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"]= openai_api_key

app = Flask(__name__)
CORS(app)


model = YOLO("assets/Yolov8n_b32_e100.pt")
botanist_bot = None

def inference(image):
    results = model(image, conf=0.4)
    infer = np.zeros(image.shape, dtype=np.uint8)
    classes = dict()
    namesInfer = []

    for r in results:
        infer = r.plot()
        classes = r.names
        namesInfer = r.boxes.cls.tolist()

    return infer, classes, namesInfer

def detect(image):
    inferencedImage, classesInDataset, classesInImage = inference(image)
    imageClassesList = list(set(classesInImage))
    label = ""

    for x in range(len(imageClassesList)):
        if x>=len(imageClassesList) - 1:
            label = label + str(classesInDataset[imageClassesList[x]])
        else:    
            label = label + str(classesInDataset[imageClassesList[x]]) + ", "

    global labels 
    labels = imageClassesList
    global classes 
    classes = classesInDataset
    
    return inferencedImage, label


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        img = cv2.imdecode(data, color_image_flag)
        detected_image, label = detect(img)
        _, img_encoded = cv2.imencode('.jpg', detected_image)
        image_as_text = base64.b64encode(img_encoded).decode('utf-8')        
        
        botanist_bot = ChatOpenAI()
        first_query = Template("""
            Disease Detection Bot: Mulberry Plant Disease Detection Information
                        
            Disease: $name
                        
            - Describe the visual symptoms associated with this disease.
            - Explain how this disease affects the overall health and growth of mulberry plant.
            - Provide recommendations or strategies for addressing and correcting this specific disease, which can help improve mulberry plant health and productivity.
            
            Thank you!

            """
        )
        global conversation
        conversation = ConversationChain(llm=botanist_bot)  
        info = conversation.run(first_query.substitute(name=label))

        return jsonify({'image': image_as_text, 'label': label, 'info': info})
    
    return jsonify({'error': 'Failed to process the image'}), 500


@app.route('/chat', methods=['POST'])
def chat():
    if not request.json or 'message' not in request.json:
        return jsonify({'error': 'No message provided'}), 400
    
    data = request.get_json()
    user_message = data['message']

    bot_response = conversation.run(user_message)
    return jsonify({'response': bot_response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)


    

