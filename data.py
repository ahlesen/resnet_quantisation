
import torchvision.datasets as dset
import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_data():
    transform_train = T.Compose([
                T.RandomCrop(32, padding=4), 
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

    transform_test = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    
    cifar10_train = dset.CIFAR10('./datasets', train=True, download=True,
                             transform=transform_train)

    loader_train = DataLoader(cifar10_train, batch_size=128, shuffle=True)

    cifar10_test = dset.CIFAR10('./datasets', train=False, download=True, 
                                transform=transform_test)
    loader_test = DataLoader(cifar10_test, batch_size=100, shuffle=False)

    return loader_train, loader_test