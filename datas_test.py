import retvid
import torchdeepretina as tdr
import numpy as np
import torch
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0")

model_file = "~/torch-deep-retina/models/15-10-07_naturalscene.pt"
model = tdr.analysis.read_model_file(model_file)
model.eval()
model.to(DEVICE)

stimgen = retvid.datas.StimGen(n_samples=10, filt_depth=40)
for i in range(stimgen.n_classes*4):
    print(i)
    stim = stimgen[i]
    print("stim:", stim.shape)
    resp_dict = tdr.utils.inspect(model, stim, batch_size=500, to_numpy=True)
    resp = resp_dict['outputs']
    fig = plt.figure()
    plt.plot(resp)
    plt.title(i)
    plt.savefig("fig{}.png".format(i))


