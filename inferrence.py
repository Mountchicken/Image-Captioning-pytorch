import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNNtoRNN
from get_loader import get_loader

def inferrence(model, dataset, image):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    image = transform(image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
    image_predict=model.caption_image(image, dataset.vocab)
    print("Predicted :" + " ".join(image_predict))

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_img=Image.open("test_examples/footable.jpg").convert("RGB")
    _, dataset=get_loader(
        root_folder="archive/Images",
        annotation_file="archive/captions.txt",
        transform=None,
        batch_size=64,
        num_workers=0
    )
    embed_size=256
    hidden_size=256
    vocab_size=len(dataset.vocab)
    num_layers=1
    model=CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    model.load_state_dict(torch.load("my_checkpoint.pth.tar")["state_dict"])
    model.eval()
    inferrence(model,dataset,test_img)