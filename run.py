import torch
from dataset import CBISDDSMDataset
from model import ResNet, resnet34
from torch import tensor, nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), min(batch * dataloader.batch_size, size)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            predicted = pred.argmax(1)
            expected = y.argmax(1)
            correct += (predicted == expected).sum().item()

    test_loss /= num_batches
    correct /= size
    # print(f"Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def model_train():
    torch.set_num_threads(16)
    train_dataset = CBISDDSMDataset(
        train=True, transform=None,
        target_transform=None)
    test_dataset = CBISDDSMDataset(
        train=False, transform=None,
        target_transform=None)

    learning_rate = 5e-3
    batch_size = 64
    epochs = 15

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    model = resnet34(1, 2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    return model


def save_model(model: nn.Module):
    path = "./saved_model.pt"
    torch.save(model.state_dict(), path)


def load_model() -> nn.Module:
    path = "./saved_model.pt"
    model = resnet34(1, 2)
    model.load_state_dict(torch.load(path))
    return model


if __name__ == '__main__':
    model = model_train()
    save_model(model)
