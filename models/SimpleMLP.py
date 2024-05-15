import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import BaseModel

class SimpleMLP(BaseModel):
    def __init__(self, config, input_dim, action_dim, max_moves, master=True, name='None'):
        super(SimpleMLP, self).__init__(config, input_dim, action_dim, max_moves, master, name)
        self.version = 'MLP'
        self.fc1 = nn.Linear(288, 500)
        self.fc2 = nn.Linear(500, 1000)
        self.fc3 = nn.Linear(1000, 500)
        self.fc4 = nn.Linear(500, 288)
        
        self.fc5 = nn.Linear(288, 200)

        self.fc6 = nn.Linear(200, action_dim)

        self.initialize_optimizers()

        self.to('cpu')
        

    def to(self, device):
        super(SimpleMLP, self).to(device)
        self.device = device
        
    def forward(self, inputs, mat):
        # Mat -> (B, 1, 12, 12)
        # Inputs -> (B, 1, 12, 12) 

        inputs = torch.concat((inputs.float(), mat.float()), dim=1)
        inputs = inputs.view(inputs.shape[0], -1)
        out = F.relu(self.fc1(inputs))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        logits = self.fc6(out)
        policy = F.softmax(logits, dim=1)

        return logits.cpu(), policy.cpu()
        #mat = mat.view(mat.shape[0], mat.shape[1],-1)
        #inputs = inputs.view(inputs.shape[0], inputs.shape[1],-1)
        #attention_input = torch.concat((inputs.float(), mat.float()), dim=2).squeeze(1)
