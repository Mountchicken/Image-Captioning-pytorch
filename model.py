import torch
import torch.nn as nn
import torchvision.models as models
from get_loader import get_loader

# def load_embeddings():
#     _,dataset=get_loader(
#         root_folder="archive/Images",
#         annotation_file="archive/captions.txt",
#         transform=None,
#         batch_size=128,
#         num_workers=0
#     )
#     vocab_size=len(dataset.vocab)
#     word2idx = dataset.vocab.stoi
#     idx2word = dataset.vocab.itos
#     model =load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
#     word_vectors = torch.randn([vocab_size, 300])
#     for i in range(0,vocab_size):
#         word = idx2word[i]
#         if word in model:
#             vector = model[word]
#             word_vectors[i, :]= torch.from_numpy(vector)
#     embedding = nn.Embedding.from_pretrained(word_vectors)
#     embedding.weight.requires_grad = False
#     return embedding

class encoderCNN(nn.Module):
    def __init__(self,embed_size, train_CNN=False):
        super(encoderCNN, self).__init__()
        self.train_CNN=train_CNN
        self.inception=models.inception_v3(pretrained=True,aux_logits=False) # aux_logits: special parameters of inception
        self.inception.fc=nn.Linear(self.inception.fc.in_features,embed_size)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.5)

    def forward(self,images):
        features=self.inception(images)
        return self.dropout(self.relu(features))

class decoderRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super(decoderRNN,self).__init__()
        self.embed= nn.Embedding(vocab_size, embed_size)
        self.lstm=nn.LSTM(embed_size,hidden_size,num_layers)
        self.linear=nn.Linear(hidden_size,vocab_size)
        self.dropout=nn.Dropout(0.5)

    def forward(self,features,captions):
        embeddings=self.dropout(self.embed(captions))
        embeddings=torch.cat((features.unsqueeze(0),embeddings),dim=0) #把features作为lstm的第一个输入
        hiddens,_=self.lstm(embeddings)
        outputs=self.linear(hiddens)
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super(CNNtoRNN,self).__init__()
        self.encoderCNN=encoderCNN(embed_size)
        self.decoderRNN=decoderRNN(embed_size,hidden_size,vocab_size,num_layers)

    def forward(self,images,captions):
        features=self.encoderCNN(images)
        outputs=self.decoderRNN(features,captions)
        return outputs

    def caption_image(self, image, vocabulary ,max_length=50): # for inference
        result_caption=[]
        with torch.no_grad():
            x=self.encoderCNN(image).unsqueeze(0) #add the batch dimension
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x,states) #image tensor feed into the first lstm
                output=self.decoderRNN.linear(hiddens.squeeze(0))
                predicted=output.argmax(1)
                result_caption.append(predicted.item())

                x=self.decoderRNN.embed(predicted).unsqueeze(0) #将预测值作为下一次的输入值
                if vocabulary.itos[predicted.item()]=="<EOS>": #最长长度设置为50，遇到EOS就停止
                    break
        return [vocabulary.itos[idx] for idx in result_caption]
