# main.py

from dataset import CIFAR10DataModule

data_module = CIFAR10DataModule(batch_size=64)
data_module.prepare_data()
data_module.setup()

def check_data_loaders(data_module):
    # Check training data
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    ood_loader = data_module.ood_dataloader()


    print("Checking training data:")
    train_batch = next(iter(train_loader))
    print(f"Training data size: {len(train_loader.dataset)}")
    print(f"First batch size: {train_batch[0].size()}")
    print(f"Sample image shape: {train_batch[0][0].shape}, Label: {train_batch[1][0]}")

    print("\nChecking validation data:")
    val_batch = next(iter(val_loader))
    print(f"Validation data size: {len(val_loader.dataset)}")
    print(f"First batch size: {val_batch[0].size()}")
    print(f"Sample image shape: {val_batch[0][0].shape}, Label: {val_batch[1][0]}")

    print("\nChecking combined test data:")
    test_batch = next(iter(test_loader))
    print(f"Test data size: {len(test_loader.dataset)}")
    print(f"First batch size: {len(test_batch[0])}")
    print(f"Sample image shape: {test_batch[0][0].shape}, Label: {test_batch[1][0]}")
    print(f"Severity Level: {test_batch[2][0]}")
    
    print("\nChecking combined ood data:")
    ood_batch = next(iter(ood_loader))
    print(f"Test data size: {len(ood_loader.dataset)}")
    print(f"First batch size: {len(ood_batch[0])}")
    print(f"Sample image shape: {ood_batch[0][0].shape}, Label: {ood_batch[1][0]}")
    print(f"Severity Level: {ood_batch[2][0]}")


# Run the check
check_data_loaders(data_module)

