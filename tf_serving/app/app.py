import streamlit as st
import requests
import json

MODEL_URI='http://tf_serving:8501/v1/models/reviews_preds:predict'

def main():

    st.title("Reviews Classification")
    message = st.text_input('Enter Review to Classify')

    if st.button('Predict'):
        payload = {
            "text": message
        }
        print(message)
        print(payload)
        headers = {"content-type": "application/json;charset=UTF-8"}
        data = json.dumps({"signature_name": "serving_default", "instances": [message]})
        res = requests.post(MODEL_URI,data=data, headers=headers)
        predictions = json.loads(res.text)['predictions'][0]
        if predictions[0] > 0.5:
            predicted_sentiment = 'positive'
        else:
            predicted_sentiment = 'negative'
        with st.spinner('Classifying, please wait....'):
            st.write(predicted_sentiment)




if __name__ == '__main__':
    main()