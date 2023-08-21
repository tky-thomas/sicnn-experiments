from matplotlib import pyplot as plt
import pickle
import numpy as np
import torch
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

# Plot the results
with open('experiment_results/sesn_voc_segmentation.pkl', 'rb') as fp:
    fp.seek(0)
    results = CPU_Unpickler(fp).load()

plt.figure()

x = np.arange(0, len(results['train_acc']))

train_results = results['train_acc']
train_results_list = []
for thing in train_results:
    train_results_list.append(thing.item())

val_results = results['val_acc']
val_results_list = []
for thing in val_results:
    val_results_list.append(thing.item())

plt.plot(x, train_results_list, label="Training Loss")
plt.plot(x, val_results_list, label="Validation Loss")
plt.legend()

plt.title("SESN VOC Segmentation Results")
plt.xlabel("Epochs (~1500 samples per epoch)")
plt.ylabel("Loss")

print(max(results['val_acc']))

plt.show()
print(results)

