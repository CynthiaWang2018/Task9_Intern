import torch
from model import FNet
from load_data import load_result_data, load_test_data
import  numpy as np
import pandas as pd

#device=torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
#[1000,10]  [1000,10]  [1000,10,2]

np_test_x, np_test_y, np_test_xy = load_test_data(x_path="data/task9_evaluate_finetune_x.csv", y_path="data/task9_evaluate_finetune_y.csv")
np_result_x = load_result_data(x_path="data/task9_evaluate_x.csv")

fnet = FNet(1, 50, 1)
map_location = lambda storage, loc: storage
fnet.load_state_dict(torch.load("fnetmodel.pkl", map_location=map_location))

fnet.eval()

# 结果
result=[]
for i in range(np_test_xy.shape[0]): # [100, 5, 2]
    fnet.eval()
    test_xy = np_test_xy[i] # [5, 2]
    test_x = test_xy[:, 0] # [5, ]
    test1_x = np_result_x[i] # [100, ]

    tensor_test1_x = torch.from_numpy(test1_x[:, np.newaxis]).float() # [100, 1]
    tensor_test_xy = torch.from_numpy(test_xy).float() # [5, 2]

    prediction_test = fnet(tensor_test1_x, tensor_test_xy) # [100, 1]

    result.append(prediction_test.data.squeeze().numpy())

np_result = np.array(result)
data1 = pd.DataFrame(np_result)
data1.to_csv('output/submission.csv', index=None, header=0)