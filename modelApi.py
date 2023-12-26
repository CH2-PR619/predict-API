from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Muat model yang disimpan
model = load_model("cataract_model.h5")
resulty = []

def _predict(model, img):
    img = np.array(img.resize((94, 55)))
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    return 'normal' if pred[0] > 0.5 else 'cataract'

@app.route('/', methods=['GET'])
def tes():
    return "hello this is api flask model"

@app.route('/predicts', methods=['POST'])
def predict():
    try:
        # Menerima data JSON dari Express.js
        data = request.get_json()

        # Mendapatkan gambar dari data JSON dan mengonversinya kembali ke format biner
        image_data = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(image_data))

        # Melakukan prediksi menggunakan model
        result = _predict(model, img)
        resulty.append({ 'result': result})
        return jsonify({'prediction': result})
    except Exception as e:
        print(e)
        return jsonify({'error': 'Terjadi kesalahan saat memproses gambar'}), 500
    
@app.route('/predictions', methods=['GET'])
def prediction():
     if resulty:
        return resulty[-1]
     else:
        return jsonify({'error': 'Belum ada prediksi yang tersedia'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

# Endpoint 
# @app.route("/predicts", methods=["GET"])
# def predict():
#    model = pickle.load(open("model.pkl","rb"))
#    
