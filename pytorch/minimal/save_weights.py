import torch
import torch.nn as nn

"""
Save the whole model
"""

model = nn.Linear(10,10)

torch.save(model, 'testfile.mdl')

loaded_model = torch.load('testfile.mdl')

print(type(loaded_model))


"""
Save and restore weights
"""

torch.save(model.state_dict(), 'testfile.wgt')
model = nn.Linear(10, 10)
model.load_state_dict(torch.load('testfile.wgt'))
assert model is not None
