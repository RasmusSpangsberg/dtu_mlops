import torch
import torchvision
from torchvision import transforms
import torchvision.models as models

def main():
    model = models.resnet18(pretrained=True)

    script_model = torch.jit.script(model)
    #script_model.save("deployable_model.pt")

    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([300, 300])
    ])

    imagenet_data = torchvision.datasets.Caltech101("data/", download=True, transform=t)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=0)

    images, labels = next(iter(data_loader))

    output = model.forward(images)
    _, unscripted_top5_indices = torch.topk(output, 5)

    output = script_model.forward(images)
    _, scripted_top5_indices = torch.topk(output, 5)

    assert torch.allclose(unscripted_top5_indices, scripted_top5_indices)


if __name__ == "__main__":
    main()
