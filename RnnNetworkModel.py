import common_resources
torch,device=common_resources.get_torch_data()

class NETWORK(torch.nn.Module):
    
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
     
        self.rnn = torch.nn.LSTM(embedding_dim,
                                 hidden_dim)   
        self.rnn1 = torch.nn.LSTM(hidden_dim,
                                 300) 
        
        self.rnn2 = torch.nn.LSTM(300,
                                 200) 
             
        self.rnn3 = torch.nn.LSTM(200,
                                 hidden_dim) 
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        

    def forward(self, text, text_length):
        
        embedded = self.embedding(text)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_length.to('cpu'))
        packed_output, _ = self.rnn(packed)
        packed_output1, _ = self.rnn1(packed_output)
        packed_output2, _ = self.rnn2(packed_output1)
        packed_output3, (hidden, cell) = self.rnn3(packed_output2)
        # print(packed_output3.data.size())
        # print("before",hidden.size())
        hidden.squeeze_(0)
        # print("after",hidden.size())
        output = self.fc(hidden)
        return output