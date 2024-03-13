import json

import requests

text = "this is very good wow ! ??"

# create a connection to the mlflow server and select the best model


def get_prediction(text):
    headers = {"content-type": "application/json;charset=UTF-8"}
    data = json.dumps({"signature_name": "serving_default", "instances": [text]})
    json_response = requests.post(
        "http://localhost:8501/v1/models/reviews_preds:predict",
        data=data,
        headers=headers,
    )
    predictions = json.loads(json_response.text)["predictions"][0]
    if predictions[0] > 0.5:
        predicted_sentiment = "positive"
    else:
        predicted_sentiment = "negative"
    return predicted_sentiment


if __name__ == "__main__":
    print(get_prediction(text))
