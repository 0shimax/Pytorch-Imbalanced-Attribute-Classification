import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from model.net import Net


def ace_loss(y_hat, target):
    inference = F.log_softmax(y_hat, dim=1)
    return F.nll_loss(inference, target)


def calculate_std_h(var_phx, n_batch):
    return torch.pow(var_phx + (var_phx**2)/(n_batch - 1), 0.5)


def wfl_loss(y_hat, y, wc, gamma=0.5):
    """
    wc = e^{-a_c}
    where a_c the prior class distribution of the cth attribute
    """
    term1 = (1 - F.softmax(y_hat))**gamma * torch.log(F.softmax(y_hat)) * y
    term2 = F.softmax(y_hat)**gamma * torch.log(1 - F.softmax(y_hat)) * (1 - y)
    agg = wc*(term1 + term2)
    return -agg.sum()


def ax_loss(yax_hat, y, std_hx):
    return (1 + std_hx) * ace_loss(yax_hat, y)


def calculate_loss(yp, y_a1, y_a2, target, wc,
                   std_h1, std_h2, i_epoch, burn_in=2):
    loss = None
    if i_epoch < burn_in:
        loss = ace_loss(yp, target)
    else:
        loss = wfl_loss(yp, target, wc)
        loss += ax_loss(y_a1, target, std_h1)
        loss += ax_loss(y_a2, target, std_h2)
    return loss


def train(args, model, device, train_loader, optimizer, epoch, wc):
    model.train()

    var_ph1 = torch.zeros(1)
    var_ph2 = torch.zeros(1)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        n_batch = data.shape[0]
        optimizer.zero_grad()

        std_h1 = calculate_std_h(var_ph1, n_batch)
        std_h2 = calculate_std_h(var_ph1, n_batch)
        yp, y_a1, y_a2 = model(data)

        fst_index = np.arange(n_batch)
        var_ph1 = y_a1[[fst_index, target]].var()
        var_ph2 = y_a2[[fst_index, target]].var()

        loss = calculate_loss(yp, y_a1, y_a2, target, wc, std_h1, std_h1, epoch)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            yp, y_a1, y_a2 = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(F.log_softmax(yp, dim=1), target,
                                    size_average=False).item()
            # get the index of the max log-probability
            pred = yp.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum)

    wc = np.zeros(10)
    for _, target in train_loader:
        for t in target:
            wc[t] += 1
    wc /= len(train_loader) * args.batch_size

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, wc)
        test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
