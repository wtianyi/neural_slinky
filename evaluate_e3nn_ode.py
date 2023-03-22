import torch
import numpy as np

from neural_slinky import coords_transform
from neural_slinky import e3nn_models
from neural_slinky import neural_ode_utils
from neural_slinky import utils


checkpoint = "./e3nn_checkpoint.pt"

def readinData():
    folder = './SlinkyGroundTruth'
    true_y = np.loadtxt(folder+'/helixCoordinate_2D.txt',delimiter=',')
    true_v = np.loadtxt(folder+'/helixVelocity_2D.txt',delimiter=',')
    true_y = torch.from_numpy(true_y).float()
    true_v = torch.from_numpy(true_v).float()
    true_y = torch.reshape(true_y,(true_y.size()[0],80,3))
    true_v = torch.reshape(true_v,(true_v.size()[0],80,3))
    # print(true_v.shape)
    return torch.cat((true_y, true_v),-1)

