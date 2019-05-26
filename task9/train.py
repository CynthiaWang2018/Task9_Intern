import torch
import torch.nn.functional as F
from model import FNet
from torch import optim
from load_data import load_data, load_test_data
import numpy as np
import torch.nn as nn

#device=torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
#[1000,10]  [1000,10]  [1000,10,2
np_train_x, np_train_y, np_train_xy = load_data(x_path='data/task9_train_x.csv', y_path='data/task9_train_y.csv')
np_test_x, np_test_y, np_test_xy = load_test_data(x_path='data/task9_evaluate_finetune_x.csv', y_path='data/task9_evaluate_finetune_y.csv')

fnet = FNet(1, 50, 1)

def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

print('fnet.parameters', fnet.parameters)
optimizer = optim.Adam(fnet.parameters(), lr=0.001)  # 0.001最好
loss_func = nn.MSELoss()

f = open("output/train_model", 'w')
for epoch in range(2):
    fnet.train()
    epoch_loss = 0
    epoch_test_loss = 0
    np.random.shuffle(np_train_xy)  # 1000打乱

    # Train
    for i in range(np_train_xy.shape[0]):
        train_xy = np_train_xy[i]  # [10, 2]
        train_x = train_xy[:, 0] # [10,]
        train_y = train_xy[:, 1] # [10,]

        tensor_x = torch.from_numpy(train_x[:,np.newaxis]).float() # [10, 1]
        tensor_y = torch.from_numpy(train_y[:, np.newaxis]).float()
        np.random.shuffle(train_xy)  # 打乱
        tensor_xy = torch.from_numpy(train_xy).float()

        # forward
        prediction = fnet(tensor_x, tensor_xy) # [10, 1] [10, 2]  # 前向传播
        loss = loss_func(prediction, tensor_y) # 计算loss

        # backward
        optimizer.zero_grad() # 梯度归零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    if epoch % 200 == 0:
        adjust_learning_rate(optimizer, decay_rate=.5)

    for i in range(np_test_xy.shape[0]):
        fnet.eval()
        test_xy = np_test_xy[i]  # [5, 2]
        test1_xy = test_xy[-1:, :] # [1, 2]
        test_xy = test_xy[:-1, :] # [4, 2]

        test_x = test_xy[:, 0] # [4,]
        test1_x = test1_xy[:, 0] # [1, ]
        test1_y = test1_xy[:, 1] # [1, ]

        tensor_test1_x = torch.from_numpy(test1_x[:, np.newaxis]).float() # [1, 1]
        tensor_test1_y = torch.from_numpy(test1_y[:, np.newaxis]).float()

        tensor_test_xy = torch.from_numpy(test_xy).float()  # [4, 2]

        prediction_test = fnet(tensor_test1_x, tensor_test_xy)  # [1, 1] [4, 2]
        loss_test = loss_func(prediction_test, tensor_test1_y)
        epoch_test_loss += loss_test
    print(epoch_test_loss)
    f.write(str(epoch_test_loss)+"\n")

torch.save(fnet.state_dict(), 'fnetmodel.pkl')
print('Finished Training')
f.close()

