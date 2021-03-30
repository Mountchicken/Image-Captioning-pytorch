import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import load_checkpoint, save_checkpoint, print_examples
from get_loader import get_loader
from model import CNNtoRNN
from tqdm import tqdm
def train():
    transform = transforms.Compose(
        [
            transforms.Resize((240,240)),
            transforms.RandomCrop((224,224)), #the input size of inception network
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ]
    )
    train_loader, dataset=get_loader(
        root_folder="archive/Images",
        annotation_file="archive/captions.txt",
        transform=transform,
        batch_size=128,
        num_workers=0
    )
    #Set some hyperparamters
    torch.backends.cudnn.benchmark = True #Speed up the training process
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    load_model=False
    save_model=False
    train_CNN=False
    embed_size=256
    hidden_size=256
    vocab_size=len(dataset.vocab)
    num_layers=1
    learning_rate=3e-4
    num_epochs=100
    #for tensorboard
    writer = SummaryWriter("runs/flickr")
    step=0
    #initialize model, loss etc
    model=CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

    # Only finetune the CNN
    for name, param in model.EncoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"),model,optimizer)

    criterion=nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])  #对于"<PAD>"的词语不需要计算损失
    optimizer= optim.Adam(filter(lambda p : p.requires_grad, model.parameters()),lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,140])
    model.train()
    print('Begins')
    imgs,captions=next(iter(train_loader))
    for epoch in range(num_epochs):
        print_examples(model, device, dataset, save_path='result.txt')
        if save_model:
            checkpoint={
                "state_dict": model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "step" :step
            }
            save_checkpoint(checkpoint)
        # loop = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
        total_loss=0
        # for idx, (imgs,captions) in loop:
        imgs = imgs.to(device)
        captions = captions.to(device)

        outputs= model(imgs, captions[:-1]) #EOS标志不需要送进网络训练，我们希望他能自己训练出来
        # outputs :(seq_len, batch_size, vocabulary_size), 但是交叉熵损失接受二维的tensor
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
        step+=1
        optimizer.zero_grad()
        loss.backward(loss)
        total_loss+=loss.item()
        optimizer.step()
        print(total_loss)
        # loop.set_description(f'Epoch[{epoch}/{num_epochs}]')
        # loop.set_postfix(total_loss=total_loss)
        # writer.add_scalar("Training loss", total_loss, epoch)
        # scheduler.step()
if __name__=="__main__":
    train()
