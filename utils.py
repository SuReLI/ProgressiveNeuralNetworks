import torch 
import torch.optim as optim
import torchvision
from prognet import *

class ReshapeTransform(object):
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)
    
def model_eval(model,dataloader,criterion,device,task_id = -1):
    model.eval()
    with torch.no_grad(): 
        eval_loss = 0
        for batch_idx, data in enumerate(dataloader): 
            img,y = data
            img = img.to(device)
            y = y.to(device)
            if isinstance(model,ProgNet):
                y_pred = model(img,task_id = task_id)
            else: 
                y_pred = model(img)
            loss = criterion(y_pred,y,reduction = 'mean')
            eval_loss += loss.item()
    
    model.train()
    return eval_loss/len(dataloader)



def train(model,dataloader,criterion,optimizer,device):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        img, y = data
        img = img.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        # forward
        y_pred = model(img)
        loss = criterion(y_pred, y,reduction = 'mean')
        # backward
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss/ len(dataloader)
    
    
def dataloader_MNIST(folder = './data/',batch_size = 64,train = True,digits = list(range(10)),flatten_data = True): 
    if flatten_data:
        transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)),
                               ReshapeTransform((-1,))
                             ])
    else: 
        transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])
    dataset = torchvision.datasets.MNIST(folder, train = train, download = True, 
                                         transform = transform)
    digits_mask = [(target in digits) for target in dataset.targets]
    dataset.targets = dataset.targets[digits_mask]
    dataset.data = dataset.data[digits_mask]
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size = batch_size, 
                                             shuffle = True)
    return dataloader
                  