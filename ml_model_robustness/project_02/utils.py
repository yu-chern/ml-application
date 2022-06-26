import torchvision.datasets as datasets
import torchvision


def get_mnist_data(train=True):
    return datasets.MNIST(root='./data', train=train, download=True, transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                 ]))


