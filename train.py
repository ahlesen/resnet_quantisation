from model import *
from data import *

import torch.optim as optim

from IPython.display import clear_output
import matplotlib.pyplot as plt

#фиксируем seed 
def init_random_seed(value=0):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)


def train_model(resnet20, num_epochs, criterion, optimizer, scheduler, loader_train, loader_test, device, with_plt=False):
    # Train the resnet20
    total_step = len(loader_train)
    total_step_val = len(loader_test) 
    train_loss_l = []
    val_loss_l = []
    acc_train = []
    acc_val = []

    best_val_acc = 0
    for epoch in range(num_epochs):
        resnet20.train()

        train_loss = 0
        correct_train = 0
        total_train = 0
        for i, (images, labels) in enumerate(loader_train):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = resnet20(images)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.data.cpu().numpy()
            
        train_loss_l.append(train_loss/total_step)
        acc_train.append(100 * correct_train / total_train)
        # validate
        val_loss = 0
        correct_val = 0
        total_val = 0
        resnet20.eval()
        with torch.no_grad():
            for images, labels in loader_test: #loader_val
                images = images.to(device)
                labels = labels.to(device)
                outputs = resnet20(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.data.cpu().numpy()
            val_loss_l.append(val_loss/total_step_val)
            acc_val.append(100 * correct_val / total_val)
            
        if ((epoch %10) == 0) & (best_val_acc < acc_val[-1]):
            best_val_acc = acc_val[-1]
            # Save best model checkpoint
            if 'ReduceLROnPlateau' in str(scheduler):
                sheduler_st_dict = scheduler.state_dict()
                torch.save(resnet20.state_dict(), 'lr_'+str(sheduler_st_dict['_last_lr'][0])+'_pat_'+str(sheduler_st_dict['patience'])+'_epoch_'+str(epoch)+'_'+str(acc_val[-1])+'_resnet20.ckpt')
            else:
                sheduler_st_dict = scheduler.state_dict()
                torch.save(resnet20.state_dict(), 'lr_'+str(sheduler_st_dict['_last_lr'][0])+'_SGD_NEW_'+'_epoch_'+str(epoch)+'_'+str(acc_val[-1])+'_resnet20.ckpt') 
        if with_plt:
            clear_output(True)
        print(f'train_loss: {train_loss/total_step:.3f}')
        print(f'  val_loss: {val_loss/total_step_val:.3f}')
        print(f'Accuracy of the resnet20 on the train images: {100 * correct_train / total_train:.1f}')
        print(f'Accuracy of the resnet20 on the val images: {100 * correct_val / total_val:.1f}')

        if with_plt:
            len_val = len(val_loss_l)
            plt.figure(figsize=(14, 12))
            plt.subplot(211)
            plt.plot(np.arange(len_val) + 1 , train_loss_l, label = 'train loss')
            plt.plot(np.arange(len_val) + 1 , val_loss_l, label = 'validation loss')
            
            plt.ylabel('loss')
            plt.xlabel('epoch number')
            plt.legend()
            plt.grid()
            
            plt.subplot(212)
            plt.plot(np.arange(len_val) + 1, acc_train, label='train')
            plt.plot(np.arange(len_val) + 1, acc_val, label='validation')
            plt.ylabel('accuracy')
            plt.xlabel('epoch number')
            plt.legend()
            plt.grid()
            plt.show()

        if scheduler is not None:
            if 'ReduceLROnPlateau' in str(scheduler):
                scheduler.step(val_loss)
            else:
                scheduler.step()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RANDOM_SEED = 1712
    
    init_random_seed(value=RANDOM_SEED)
    
    loader_train, loader_test = get_data()

    resnet20 = ResNet(BasicBlock, [3, 3, 3])

    # Hyper-parameters
    num_epochs = 200
    learning_rate = 1e-2

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(resnet20.parameters(), lr=learning_rate,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_model(resnet20, num_epochs, criterion, optimizer, scheduler, loader_train, loader_test, device)

