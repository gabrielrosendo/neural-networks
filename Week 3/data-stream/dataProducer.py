import asyncio
import websockets
import json
import random
from datetime import datetime

async def send_transaction_data(websocket, path):
    print(f"New client connected from {websocket.remote_address}")
    transaction_id = 1  # Starting transaction ID
    try:
        while True:
            # Generate transaction data
            transaction_data = {
                "transactionID": transaction_id,
                "userID": random.randint(1, 50),  # Random user ID between 1 and 10
                "amount": round(random.uniform(10.0, 1000.0), 2),  # Random amount between $10.00 and $1000.00
                "itemID": random.randint(1, 100)
            }
            await websocket.send(json.dumps(transaction_data))
            print(f"Sent: {transaction_data}")
            transaction_id += 1  # Increment the transaction ID
            await asyncio.sleep(2)  # Send data every 2 seconds
    except websockets.ConnectionClosed:
        print(f"Client disconnected from {websocket.remote_address}")

async def main():
    async with websockets.serve(send_transaction_data, "localhost", 9999):
        print("WebSocket server started on ws://localhost:9999")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
