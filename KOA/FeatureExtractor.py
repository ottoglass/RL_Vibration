from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from chronos import ChronosPipeline

class reshaper(nn.Module):
    def __init__(self,output_size=20,stride=2):
        super().__init__()
        self.output_size=output_size
        self.stride=stride
    def forward(self, x):
        x=F.unfold(x,(self.output_size,2),stride=self.stride)
        x=torch.movedim(x,2,1)
        return x
class remove_tuple(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x[0]

class RNN_Model(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, x, num_features: int = 256):
        super(RNN_Model, self).__init__()
        self.rnn = nn.Sequential(
            reshaper(),
            nn.Linear(40,20),
            nn.ReLU(),
            nn.LSTM(20, 10, batch_first=True,num_layers=10,dropout=0.1),
            remove_tuple(),
            nn.TransformerEncoderLayer(nhead=5,d_model=10 ,batch_first=True),
            nn.BatchNorm1d(num_features=141),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        self.rnn_large=nn.Sequential(            
            reshaper(output_size=80,stride=2),
            nn.Linear(160,20),
            nn.ReLU(),
            nn.LSTM(20, 10, batch_first=True,num_layers=5,dropout=0.1),
            remove_tuple(),
            nn.BatchNorm1d(num_features=111),
            nn.Flatten(),
        )
        # self.rnn_large=nn.Sequential(            
        #     reshaper(output_size=160,stride=10),
        #     nn.Linear(320,20),
        #     nn.ReLU(),
        #     nn.LSTM(20, 10, batch_first=True,num_layers=5,dropout=0.1),
        #     remove_tuple(),
        #     nn.BatchNorm1d(num_features=111),
        #     nn.Flatten(),
        #     )
        # self.rnn_large=nn.Sequential(            
        #     reshaper(output_size=5,stride=2),
        #     nn.Linear(10,20),
        #     nn.ReLU(),
        #     nn.LSTM(20, 10, batch_first=True,num_layers=5,dropout=0.1),
        #     remove_tuple(),
        #     nn.BatchNorm1d(num_features=111),
        #     nn.Flatten(),
        #     )
        self.bypass=nn.Sequential(
            nn.Flatten(),
            nn.Linear(x.shape[-1]*x.shape[-2],num_features),
            nn.ReLU()
        )
        with torch.no_grad():
            n_flatten = self.rnn(x).shape[1]
            n_flatten+= self.bypass(x).shape[1]
            n_flatten+=self.rnn_large(x).shape[1]


        self.linear = nn.Sequential(
            nn.Linear(n_flatten,num_features*2),
            nn.ReLU(),
            nn.Linear(num_features*2, num_features),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            )
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([self.rnn(observations),self.rnn_large(observations),self.bypass(observations)],dim=1))


class T5_Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,weights=None, features_dim: int = 256,model_name='small'):
            super().__init__(observation_space, features_dim)
            model_list={'small':(512,"amazon/chronos-t5-small"),'base':(768,"amazon/chronos-t5-base")}
            x=torch.as_tensor(observation_space.sample()[None]).float()
            self.pipeline = ChronosPipeline.from_pretrained(
                model_list[model_name][1],
                device_map="cuda:0",
                torch_dtype=torch.float32,
            )
            y=x[:,0,:,1]
            self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear((y.shape[-1]+1)*model_list[model_name][0],features_dim*2),
            nn.ReLU(),
            nn.Linear(features_dim*2, features_dim),
            nn.Tanh(),
            nn.BatchNorm1d(num_features=128),
            )
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x=observations[:,0,:,1].to(device='cpu')
        x,_=self.pipeline.embed(x)            
        # return self.linear(x.to('cuda:0'))  x
        return self.linear(x)
            



class RNN_Extractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box,weights=None, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        x=torch.as_tensor(observation_space.sample()[None]).float()
        self.rnn = nn.Sequential(
            reshaper(),
            nn.Linear(40,20),
            nn.ReLU(),
            nn.LSTM(20, 10, batch_first=True,num_layers=10,dropout=0.1),
            remove_tuple(),
            nn.TransformerEncoderLayer(nhead=5,d_model=10 ,batch_first=True),
            nn.BatchNorm1d(num_features=141),
            nn.Flatten(),
        )
        self.bypass=nn.Sequential(
            nn.Flatten(),
            nn.Linear(x.shape[-1]*x.shape[-2],features_dim),
            nn.ReLU()
        )
        with torch.no_grad():
            n_flatten = self.rnn(x).shape[1]
            n_flatten+= self.bypass(x).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten,features_dim*2),
            nn.ReLU(),
            nn.Linear(features_dim*2, features_dim),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([self.rnn(observations),self.bypass(observations)],dim=1))

class RNN_ExtractorV2(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box,weights=None, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or 
        if weights==None:
            x=torch.as_tensor(observation_space.sample()[None]).float()
            self.model=RNN_Model(x,num_features=features_dim)
        else:
            x=torch.as_tensor(observation_space.sample()[None]).float()
            self.model=RNN_Model(x,num_features=features_dim)
            self.model.load_state_dict(torch.load(weights))


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model.forward(observations)
