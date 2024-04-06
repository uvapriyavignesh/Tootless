import torch
import constant as co
import torchtext
import random

def get_torch_data():
    torch.manual_seed(co.RANDOM_SEED)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch,DEVICE

def get_text_label_splitter_from_torch_text():
    text_spliter_class = torchtext.data.Field(
    tokenize='spacy',
    tokenizer_language='en_core_web_sm',
    include_lengths=True
)

    label_spliter_class = torchtext.data.LabelField(dtype=torch.long)
    return text_spliter_class,label_spliter_class

def get_train_test_data_set():
    text_spliter_class,label_spliter_class=get_text_label_splitter_from_torch_text()
    fields = [(co.TEXT_COLUMN_NAME, text_spliter_class), (co.LABEL_COLUMN_NAME, label_spliter_class)]
    dataset = torchtext.data.TabularDataset(
    path='C:\\Toothless\\data\\filter_date.csv', format='csv',
    skip_header=True, fields=fields)

    

    train,test= dataset.split(
    split_ratio=[0.8, 0.2],
    random_state=random.seed(co.RANDOM_SEED))

    text_spliter_class.build_vocab(train, max_size=co.VOCABULARY_SIZE)
    label_spliter_class.build_vocab(train)
    return train,test,text_spliter_class,label_spliter_class

def get_data_loader(train,test):
    _,device=get_torch_data()
    train_loader, test_loader = torchtext.data.BucketIterator.splits(
        (train, test), 
        batch_size=co.BATCH_SIZE,
        sort_within_batch=True, 
        sort_key=lambda x: len(x.text),
        device=device
)
    return train_loader,test_loader