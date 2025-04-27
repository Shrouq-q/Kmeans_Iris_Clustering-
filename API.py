from flask import Flask, request, jsonify
from Iris import predict_cluster

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the KMeans Iris Clustering API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_features = data.get('input_features')

    if not input_features or len(input_features) != 4:
        return jsonify({'error': 'Input must be a list of 4 numeric features'}), 400

    cluster = predict_cluster(input_features)

    return jsonify({'cluster': cluster})

if __name__ == '__main__':
    app.run(debug=True)
