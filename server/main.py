import json

import spacy
import torch
from flask import Flask, request

import RnnNetworkModel
import common_resources as cr

token_lang = spacy.blank("en")
device = "cpu"
app = Flask(__name__)
train, test, text_spliter_class, label_spliter_class = cr.get_train_test_data_set()
out_dict = {
    "open": "1",
    "paused": "2",
    "resolved": "3",
    "critical": "$4",
    "closed": "4",
    "deleted": "5",
    "low":"$1",
    "high": "$3",
    "resolved & closed": "8",
    "medium": "$2"
}

def load_model():
    model1 = RnnNetworkModel.NETWORK(
        input_dim=len(text_spliter_class.vocab),
        embedding_dim=128,
        hidden_dim=256,
        output_dim=10,
    )
    # map_location=torch.device('cpu')
    map_location = torch.device("cpu")
    model1 = model1.to(device="cpu")
    # Load the trained weights
    model_weights_path = "/home/uvapriyavignesh/toothless/model_ini_up.pth"
    state_dict = torch.load(model_weights_path, map_location="cpu")
    model1.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
    return model1


toothless_model = load_model()


@app.post('/chatbot')
def get_response():
    data = request.data.decode("utf-8")
    print(data)
    dict = json.loads(data)
    user_input = dict.get("user_input")

    out = get_data(str(user_input))
    temp=out_dict.get(out)
    temp1='{ "functionName": "globalSearch", "response": ['
    response= "{\"priorities\":["+temp.replace("$","")+"]}" if temp.__contains__("$") else "{\"statusList\":["+temp+"]}"
    sam=temp1+response+' ] }'
    return sam


def test_data(model, sentence):
    model.eval()

    with torch.no_grad():
        tokenized = [tok.text for tok in token_lang.tokenizer(sentence)]
        indexed = [text_spliter_class.vocab.stoi[t] for t in tokenized]
        length = [len(indexed)]
        tensor = torch.LongTensor(indexed).to(device=device)
        tensor = tensor.unsqueeze(1)
        length_tensor = torch.LongTensor(length)
        predict_probas = torch.nn.functional.softmax(
            model(tensor, length_tensor), dim=1
        )
        predicted_label_index = torch.argmax(predict_probas)
        predicted_label_proba = torch.max(predict_probas)
        return predicted_label_index.item(), predicted_label_proba.item()


def get_data(data):
    if data is not None:
        data=data.lower()
        val = test_data(toothless_model, data)
        class_mapping = label_spliter_class.vocab.stoi
        inverse_class_mapping = {v: k for k, v in class_mapping.items()}
        print(f'{data} --- output: {inverse_class_mapping.get(val[0])} --- probablity: {val[1]}')
        return inverse_class_mapping.get(val[0])


# app.add_url_rule("/", "chatbot", get_response, methods=["POST"])

# print(get_data("get open status"))
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
    # print(f'predicted value : {inverse_class_mapping.get(val[0])} \nprobablity:{val[1]}')
