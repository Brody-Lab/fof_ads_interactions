import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

to_t = lambda array: torch.tensor(array, device = device, dtype = dtype)
from_t = lambda tensor: tensor.to("cpu").detach().numpy()

import sys
sys.path.insert(1, '../../../../figure_code/')

plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.Spectral(np.linspace(0,1,8)))
plt.rcParams["font.family"] = 'Sans-serif'