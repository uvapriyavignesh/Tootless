import common_resources
torch,device=common_resources.get_torch_data()

class NETWORK(torch.nn.Module):
    
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
     
        self.rnn = torch.nn.LSTM(embedding_dim,
                                 hidden_dim)   
             
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        

    def forward(self, text, text_length):
        
        embedded = self.embedding(text)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_length.to('cpu'))
        packed_output, (hidden, cell) = self.rnn(packed)
        
        hidden.squeeze_(0)
        
        output = self.fc(hidden)
        return output