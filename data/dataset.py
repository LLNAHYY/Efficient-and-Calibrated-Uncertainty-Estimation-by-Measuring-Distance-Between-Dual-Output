import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN,MNIST, FashionMNIST
import torchvision.transforms as transforms
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import random_split

class CIFAR10CDataset(Dataset):
    def __init__(self, data_dir, corruption_type, severity, transform=None):
        self.data_dir = data_dir
        self.corruption_type = corruption_type
        self.severity = severity
        self.transform = transform
        
        file_path = os.path.join(self.data_dir, f'{self.corruption_type}.npy')
        self.data = np.load(file_path)
        self.labels = np.load(os.path.join(self.data_dir, 'labels.npy'))

        # Only take the specified severity level (each severity level contains 10,000 images)
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        self.data = self.data[start_idx:end_idx]
        self.labels = self.labels[start_idx:end_idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, self.corruption_type, self.severity



class CustomDataset(Dataset):
    def __init__(self, images, labels, severity_levels):
        self.images = images
        self.labels = labels
        self.severity_levels = severity_levels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        severity_level = self.severity_levels[idx]
        return image, label, severity_level
    


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='/mnt/e/study_dataset/cifar10/cifar', cifar_c_dir='/mnt/e/study_dataset/cifar10/cifar-c', batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.cifar_c_dir = cifar_c_dir
        self.batch_size = batch_size

        # Define transforms for training, validation, and testing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        # Download CIFAR-10, CIFAR-100, and SVHN datasets
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)
        CIFAR100(root=self.data_dir, train=False, download=True)
        SVHN(root=self.data_dir, split='test', download=True)

    def setup(self, stage=None):
        # Load CIFAR-10 dataset
        cifar10_train = CIFAR10(root=self.data_dir, train=True, transform=self.transform)
        #cifar10_val = CIFAR10(root=self.data_dir, train=False, transform=self.transform)


        # 추가
        cifar10_train = CIFAR10(root=self.data_dir, train=True, transform=self.transform)

        # 추가
        train_size = int(0.8 * len(cifar10_train))
        val_size = len(cifar10_train) - train_size

        self.train_dataset, self.val_dataset = random_split(cifar10_train, [train_size, val_size])

        # Subset to match the sample size requirements
        #self.train_dataset = Subset(cifar10_train, torch.arange(30000))
        #self.val_dataset = Subset(cifar10_val, torch.arange(6000))

        # Load CIFAR-10-C dataset for testing and combine with CIFAR-100 and SVHN
        self.test_dataset  = self.load_balanced_test_data()
        self.ood_dataset = self.load_ood_data()

    def load_balanced_test_data(self):
        # List of corruption types
        corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 
                       'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 
                       'pixelate', 'jpeg_compression']

        cifar_c_image = []
        cifar_c_labels = []
        cifar_c_severity_levels = []

        # Load clean CIFAR-10 test images (500 images for severity level 0)
        cifar10_test = CIFAR10(root=self.data_dir, train=False, transform=self.transform)
        indices = np.random.choice(len(cifar10_test), 1500, replace=False) #500
        for idx in indices:
            img, label = cifar10_test[idx]
            cifar_c_image.append(img)
            cifar_c_labels.append(label)
            cifar_c_severity_levels.append(0)

        # Load 500 images for each severity level 1 to 5 (500 images total, 100 from each corruption type)
        for severity in range(1, 6):
            images_selected = 0
            for corruption in corruptions:
                if images_selected >= 2000:
                    break
                cifar10c = CIFAR10CDataset(data_dir=self.cifar_c_dir, corruption_type=corruption, severity=severity, transform=self.transform)
                indices = np.random.choice(len(cifar10c), min(100, 2000 - images_selected), replace=False)
                for idx in indices:
                    img, label, _, actual_severity = cifar10c[idx]
                    #print(actual_severity) #
                    #print(corruption)
                    cifar_c_image.append(img)
                    cifar_c_labels.append(label)
                    #cifar_c_severity_levels.append(corruption)
                    cifar_c_severity_levels.append(actual_severity)
                    images_selected += 1
                    if images_selected >= 2000:
                        break

        return CustomDataset(cifar_c_image, cifar_c_labels, cifar_c_severity_levels)

    def load_ood_data(self):
        ood_image = []
        ood_labels = []
        ood_type = []

        # Load CIFAR-100 test images (3000 images)
        cifar100_test = CIFAR100(root=self.data_dir, train=False, transform=self.transform)
        indices = np.random.choice(len(cifar100_test), 6000, replace=False)
        for idx in indices:
            img, label = cifar100_test[idx]
            ood_image.append(img)
            ood_labels.append(label)
            ood_type.append("cifar100_ood")

        # Load SVHN test images (3000 images)
        svhn_test = SVHN(root=self.data_dir, split='test', transform=self.transform)
        indices = np.random.choice(len(svhn_test), 6000, replace=False)
        for idx in indices:
            img, label = svhn_test[idx]
            ood_image.append(img)
            ood_labels.append(label)
            ood_type.append("svhn_ood")

        return CustomDataset(ood_image, ood_labels, ood_type)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def ood_dataloader(self):
        return DataLoader(self.ood_dataset, batch_size=self.batch_size, shuffle=False)



import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import CIFAR100

class CIFAR100CDataset(Dataset):
    def __init__(self, data_dir, corruption_type, severity, transform=None):
        self.data_dir = data_dir
        self.corruption_type = corruption_type
        self.severity = severity
        self.transform = transform
        
        file_path = os.path.join(self.data_dir, f'{self.corruption_type}.npy')
        self.data = np.load(file_path)
        self.labels = np.load(os.path.join(self.data_dir, 'labels.npy'))

        # Only take the specified severity level (each severity level contains 10,000 images)
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        self.data = self.data[start_idx:end_idx]
        self.labels = self.labels[start_idx:end_idx]

        # Ensure data is in (C, H, W) format if currently in (H, W, C)
        if self.data.shape[-1] == 3:
            self.data = self.data.transpose(0, 3, 1, 2)  # Convert to (N, C, H, W)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, self.corruption_type, self.severity


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='/mnt/e/study_dataset/cifar100/cifar', cifar_c_dir='/mnt/e/study_dataset/cifar100/cifar-c', batch_size=8):
        super().__init__()
        self.data_dir = data_dir
        self.cifar_c_dir = cifar_c_dir
        self.batch_size = batch_size

        # Define transforms for training, validation, and testing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        # Download CIFAR-100 datasets
        CIFAR100(root=self.data_dir, train=True, download=True)
        CIFAR100(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Load CIFAR-100 dataset
        cifar100_train = CIFAR100(root=self.data_dir, train=True, transform=self.transform)
        #cifar100_val = CIFAR100(root=self.data_dir, train=False, transform=self.transform)
        CIFAR10(root=self.data_dir, train=False, download=True)
        SVHN(root=self.data_dir, split='test', download=True)

        train_size = int(0.8 * len(cifar100_train))
        val_size = len(cifar100_train) - train_size
        self.train_dataset, self.val_dataset = random_split(cifar100_train, [train_size, val_size])

        # Load CIFAR-100-C dataset for testing

        self.test_dataset = self.load_balanced_test_data()
        self.ood_dataset = self.load_ood_data()
        

    def load_balanced_test_data(self):
        corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 
                       'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 
                       'pixelate', 'jpeg_compression']
        severity_levels = range(1, 6)

        combined_images = []
        combined_labels = []
        combined_severity_levels = []

        # Load clean CIFAR-100 test images (1000 images for severity level 0)
        cifar100_test = CIFAR100(root=self.data_dir, train=False, transform=self.transform)
        indices = np.random.choice(len(cifar100_test), 1000, replace=False)
        for idx in indices:
            img, label = cifar100_test[idx]
            combined_images.append(img)
            combined_labels.append(label)
            combined_severity_levels.append(0)

        # Load 1000 images for each severity level 1 to 5
        for severity in severity_levels:
            all_corrupted_images = []
            all_corrupted_labels = []

            # Combine all corruptions for the given severity
            for corruption in corruptions:
                cifar100c = CIFAR100CDataset(data_dir=self.cifar_c_dir, corruption_type=corruption, severity=severity, transform=self.transform)
                all_corrupted_images.extend(cifar100c.data)
                all_corrupted_labels.extend(cifar100c.labels)

            # Randomly select 1000 samples from the combined corruptions
            selected_indices = np.random.choice(len(all_corrupted_images), 1000, replace=False)
            for idx in selected_indices:
                img = torch.tensor(all_corrupted_images[idx])
                label = all_corrupted_labels[idx]
                combined_images.append(img)
                combined_labels.append(label)
                combined_severity_levels.append(severity)
            

        # Convert combined lists to tensors and create a TensorDataset
        #combined_images = torch.stack(combined_images)
        #combined_labels = torch.tensor(combined_labels)
        #combined_severity_levels = torch.tensor(combined_severity_levels)

        return CustomDataset(combined_images, combined_labels, combined_severity_levels)
    
    def load_ood_data(self):
        ood_image = []
        ood_labels = []
        ood_type = []

        # Load CIFAR-100 test images (3000 images)
        cifar10_test = CIFAR10(root=self.data_dir, train=False, transform=self.transform)
        indices = np.random.choice(len(cifar10_test), 3000, replace=False)
        for idx in indices:
            img, label = cifar10_test[idx]
            ood_image.append(img)
            ood_labels.append(label)
            ood_type.append("cifar10_ood")
        
        # Load SVHN test images (3000 images)
        svhn_test = SVHN(root=self.data_dir, split='test', transform=self.transform)
        indices = np.random.choice(len(svhn_test), 3000, replace=False)
        for idx in indices:
            img, label = svhn_test[idx]
            ood_image.append(img)
            ood_labels.append(label)
            ood_type.append("svhn_ood")

        return CustomDataset(ood_image, ood_labels, ood_type)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=8, shuffle=False)
    
    def ood_dataloader(self):
        return DataLoader(self.ood_dataset, batch_size=8, shuffle=False)  

  
class Example(Dataset):
    def __init__(self, size, is_test=False):
        self.size = size
        self.is_test = is_test
        self.images = torch.randn(size, 3, 32, 32)  # (N, C, H, W) 형식의 임의 이미지 생성
        self.labels = torch.randint(0, 10, (size,))  # 0~9까지의 임의 라벨 생성

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.is_test:
            # test 데이터일 경우 "example" 문자열을 추가로 반환
            return image, label, "example"
        else:
            return image, label

def create_dataloaders():
    # 임의의 데이터셋 생성
    train_dataset = Example(size=128)
    val_dataset = Example(size=32)
    test_dataset = Example(size=36, is_test=True)

    # DataLoader 구성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=36, shuffle=False)

    return train_loader, val_loader, test_loader


import torchvision

    
def collate_fn(batch):
    """Colate function for dataloader as dictionary."""
    images, targets = zip(*batch)
    images = torch.stack(images)
    targets = torch.tensor(targets)
    return {"input": images, "target": targets}




from PIL import Image


class MNISTCDataset(Dataset):
    def __init__(self, data_dir, corruption_type, transform=None):
        self.data_dir = data_dir
        self.corruption_type = corruption_type
        self.transform = transform

        # Load the corrupted images and labels for the specified corruption type
        corruption_path = os.path.join(self.data_dir, self.corruption_type)
        self.data = np.load(os.path.join(corruption_path, 'test_images.npy'))
        self.labels = np.load(os.path.join(corruption_path, 'test_labels.npy'))

        # Select the images corresponding to the specified severity level
        #start_idx = (severity - 1) * 10000
        #end_idx = severity * 10000
        self.data = self.data #[start_idx:end_idx]
        self.labels = self.labels #[start_idx:end_idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        image = np.squeeze(image, axis=2)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label, self.corruption_type   #, self.severity
    

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='/mnt/e/study_dataset/mnist', mnist_c_dir='/mnt/e/study_dataset/mnist/mnist_c', batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.mnist_c_dir = mnist_c_dir
        self.batch_size = batch_size

        # Define transforms for training, validation, and testing
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 이미지를 Tensor로 변환하고 [0, 1] 범위로 정규화
            transforms.Normalize((0.5,), (0.5,))  # 평균 0.5, 표준편차 0.5로 정규화
        ])


    def prepare_data(self):
        # Download MNIST, CIFAR-10, and SVHN datasets
        MNIST(root=self.data_dir, train=True, download=True)
        MNIST(root=self.data_dir, train=False, download=True)
        FashionMNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Load and split MNIST dataset into train and validation sets
        mnist_train = MNIST(root=self.data_dir, train=True, transform=self.transform)
        train_size = int(0.8 * len(mnist_train))
        val_size = len(mnist_train) - train_size
        self.train_dataset, self.val_dataset = random_split(mnist_train, [train_size, val_size])

        # Load MNIST-C dataset for testing
        self.test_dataset = self.load_balanced_test_data()
        
        # Load OOD data using CIFAR-10 and SVHN
        self.ood_dataset = self.load_ood_data()

    def load_balanced_test_data(self):
        # List of corruption types for MNIST-C
        corruptions = ['identity', 'shot_noise', 'impulse_noise', 'glass_blur', 'motion_blur', 'shear', 'scale', 
                       'rotate', 'brightness', 'translate', 'stripe', 'fog', 'spatter', 'dotted_line', 'zigzag', 
                       'canny_edges']

        combined_images = []
        combined_labels = []
        combined_corruption_types = []
        combined_severity_levels = []

        # Load clean MNIST test images (1000 images for severity level 0)
        mnist_test = MNIST(root=self.data_dir, train=False, transform=self.transform)
        indices = np.random.choice(len(mnist_test), 1500, replace=False)
        for idx in indices:
            img, label = mnist_test[idx]
            combined_images.append(img)
            combined_labels.append(label)
            combined_corruption_types.append('clean')
            combined_severity_levels.append(0)

        # Load corrupted images for each severity level from MNIST-C
        for corruption in corruptions:
            mnist_c_dataset = MNISTCDataset(data_dir=self.mnist_c_dir, corruption_type=corruption, transform=self.transform)
            
            # Randomly select 1000 samples for each severity level
            selected_indices = np.random.choice(len(mnist_c_dataset), 100, replace=False)
            for idx in selected_indices:
                img, label, corruption_type = mnist_c_dataset[idx]
                combined_images.append(img)
                combined_labels.append(label)
                combined_corruption_types.append(corruption_type)

        return CustomDataset(combined_images, combined_labels, combined_corruption_types)

    def load_ood_data(self):
        ood_images = []
        ood_labels = []
        ood_types = []

        # Load CIFAR-10 test images (3000 images)
        cifar10_test = FashionMNIST(root=self.data_dir, train=False, transform=self.transform)
        indices = np.random.choice(len(cifar10_test), 3000, replace=False)
        for idx in indices:
            img, label = cifar10_test[idx]
            ood_images.append(img)
            ood_labels.append(label)
            ood_types.append('cifar10_ood')

        return CustomDataset(ood_images, ood_labels, ood_types)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def ood_dataloader(self):
        return DataLoader(self.ood_dataset, batch_size=self.batch_size, shuffle=False)


