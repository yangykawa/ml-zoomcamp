import requests

url = "http://127.0.0.1:9696/predict"  
data = {
    "texts": [
    "The movie was absolutely amazing, from the plot to the performances!",
    "I don’t understand why everyone loves this game, it’s so glitchy and unplayable, totally a waste of money.",
    "The restaurant was okay, nothing special but not bad either.",
    "Just finished my morning workout, feeling great!",
    "This new phone is amazing, the camera quality is perfect, and it runs so smoothly.",
    "Horrible customer service! They never respond to my emails, and I’m still waiting for a refund.",
    "The movie was alright, just average, neither great nor terrible.",
    "Does anyone else love the smell of fresh rain? It’s so calming!",
    "Great service at the hotel, staff was friendly and welcoming, definitely coming back.",
    "Tried the new burger at this restaurant, and it was undercooked and tasteless. Not coming back."
  ]

}

try:
    response = requests.post(url, json=data)
    if response.status_code == 200:
        predictions = response.json().get("predictions", [])
        if len(predictions) == len(data["texts"]):
            for text, prediction in zip(data["texts"], predictions):
                print("-----------------------------")
                print(f"Text: {text}")
                print(f"Prediction: {prediction}")
        else:
            print("Error: Predictions do not match the number of texts.")
    else:
        print(f"Error response: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
