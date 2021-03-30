import torch
import torchvision.transforms as transforms
from PIL import Image


def print_examples(model, device, dataset, save_path=None):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open("test_examples/cat.jpg").convert("RGB")).unsqueeze(0)
    test_img1_predict=model.caption_image(test_img1.to(device), dataset.vocab)
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print(
        "Example 1 OUTPUT: "
        + " ".join(test_img1_predict)
    )
    test_img2 = transform(Image.open("test_examples/dog.jpg").convert("RGB")).unsqueeze(0)
    test_img2_predict=model.caption_image(test_img2.to(device), dataset.vocab)
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    print(
        "Example 2 OUTPUT: "
        + " ".join(test_img2_predict)
    )
    test_img3 = transform(Image.open("test_examples/tree.jpg").convert("RGB")).unsqueeze(0)
    test_img3_predict=model.caption_image(test_img3.to(device), dataset.vocab)
    print("Example 3 CORRECT: Bus driving by parked cars")
    print(
        "Example 3 OUTPUT: "
        + " ".join(test_img3_predict)
    )
    test_img4 = transform(Image.open("test_examples/mountains.jpg").convert("RGB")).unsqueeze(0)
    test_img4_predict=model.caption_image(test_img4.to(device), dataset.vocab)
    print("Example 4 CORRECT: A small boat in the ocean")
    print(
        "Example 4 OUTPUT: "
        + " ".join(test_img4_predict)
    )
    # test_img5 = transform(Image.open("test_examples/horse.png").convert("RGB")).unsqueeze(0)
    # test_img5_predict=model.caption_image(test_img5.to(device), dataset.vocab)
    # print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    # print(
    #     "Example 5 OUTPUT: "
    #     + " ".join(test_img5_predict)
    # )
    if save_path is not None:
        with open(save_path,'a') as f:
            f.write(
                "Example 1 OUTPUT: " + " ".join(test_img1_predict)+'\n'+
                "Example 2 OUTPUT: " + " ".join(test_img2_predict)+'\n'+
                "Example 3 OUTPUT: " + " ".join(test_img3_predict)+'\n'+
                "Example 4 OUTPUT: " + " ".join(test_img4_predict)+'\n'+'\n'
            )
    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step