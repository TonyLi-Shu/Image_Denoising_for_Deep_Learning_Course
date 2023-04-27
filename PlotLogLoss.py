import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn
from tqdm import tqdm


model_path = "./model"
train_loss = np.load(os.path.join(model_path,"train_loss.npy"))
val_loss   = np.load(os.path.join(model_path,"valid_loss.npy"))

plt.figure()
plt.plot(np.log(train_loss),label = "train")
plt.plot(np.log(val_loss), label="valid")
plt.legend()
plt.title("Log Loss")
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig(os.path.join(model_path, "LogLoss.png"), dpi=200,bbox_inches='tight')
plt.show()