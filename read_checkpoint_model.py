import torch 
model = torch.load("data-bin/wmt14.en-fr.fconv-py/model.pt", map_location=torch.device('cpu'))
print(model.keys())
print(model['model'].keys())
