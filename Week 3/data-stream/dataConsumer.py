# websocket_client.py

import asyncio
import websockets
import json
import numpy as np
from datetime import datetime
import tensorflow as tf

from model import extract_features


async def receive_data():
    uri = "ws://localhost:9999"
    # Load the saved model
    model_filename = 'model.keras'
    model = tf.keras.models.load_model(model_filename)
    print(f"Model loaded from {model_filename}")

    # Initialize known users
    known_users = set()

    async with websockets.connect(uri) as websocket:
        print("Connected to the WebSocket server")
        try:
            while True:
                data = await websocket.recv()
                transaction = json.loads(data)
                # Extract features
                features_dict = extract_features(transaction, known_users)
                features = np.array([list(features_dict.values())])
                # Make prediction
                prediction = model.predict(features)
                is_fraud = prediction[0][0] > 0.5  # Threshold can be adjusted
                # Print the result
                print(f"Predicted Fraud: {'Yes' if is_fraud else 'No'} (Probability: {prediction[0][0]:.2f})")
                print(f"Transaction Details: {transaction}\n")
                # Update known users
                known_users.add(transaction['userID'])
        except websockets.ConnectionClosed:
            print("Connection closed")

if __name__ == "__main__":
    asyncio.run(receive_data())