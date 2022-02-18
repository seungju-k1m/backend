# backend

### This repo provides a standard API for constructing neural network architecture (NNArch) using `Pytorch`, covering Multi-Input-Multi-Output (MIMO) NNArch without any change of code. Corresponding to the NNArch, it can generate its own optimizer. Also, it supports fundamental ingredients of Reinforement Learning such as Replay Memory. In my repos, it is widely used for efficient and robust development.

&nbsp;

# Installation
In your repo, 
    
    git submodule add https://github.com/seungju-mmc/backend.git
    git submodule update --init

&nbsp;

# API

The backend API constructs NNArch using  ` model ` classes.

`model ` is configured by  `json ` file (below):

```json
{
    "model":{
        "module00":{
            "netCat":"CNN2D",
            "iSize":4,
            "nLayer":4,
            "fSize":[8, 4, 3, -1],
            "nUnit":[32, 64, 64],
            "padding":[0, 0, 0],
            "stride":[4, 2, 1],
            "act":["relu", "relu", "relu"],
            "BN":[false, false, false, false],
            "linear":true,
            "input":[0],
            "prior":0
        },
        "module01":{
            "netCat":"ViewV2",
            "prevNodeNames":["module00"],
            "input":[1],
            "prior":1
        },
        "module02":{
            "netCat":"LSTMNET",
            "hiddenSize":512,
            "nLayer":1,
            "iSize":3136,
            "device":"cpu",
            "FlattenMode":true,
            "return_hidden":false,
            "prior":2,
            "prevNodeNames":["module01"]
        },
        "module03":{
            "netCat":"MLP",
            "iSize":512,
            "nLayer":2,
            "fSize":[512, 6],
            "act":["relu", "linear"],
            "BN":[false, false, false],
            "prior":3,
            "prevNodeNames":["module02"]
        },
        "module03_1":{
            "netCat":"MLP",
            "iSize":512,
            "nLayer":2,
            "fSize":[512,  1],
            "act":["relu",  "linear"],
            "BN":[false, false, false],
            "prior":3,
            "prevNodeNames":["module02"]
        },
        "module04":{
            "netCat":"Add",
            "prior":4,
            "prevNodeNames":["module03", "module03_1"]
        },
        "module04_1":{
            "netCat":"Mean",
            "prior":4,
            "prevNodeNames":["module03"]
        },
        "module05":{
            "netCat":"Substract",
            "prior":5,
            "prevNodeNames":["module04", "module04_1"],
            "output":true
        }
    }
}
```

```python

from backend.utils import jsonParser
from backend.NNArch import model

import torch

path = './demo.json'

data = jsonParser(path)

m = model(data.model)

input_ex = torch.rand((1, 4, 84, 84))
shape = torch.tensor([1, 1, -1])

action_value = m.forward([input_ex, shape])


# You can also operate cell control in RNN

m.detachCellState()

cell_states = m.getCellState()

m.setCellState(cell_states)
```

This example is the NNArch proposed by [`R2D2`](https://openreview.net/forum?id=r1lyTjAqYX&utm_campaign=RL%20Weekly&utm_medium=email&utm_source=Revue%20newsletter)


Without any change of code, you can rebuild NNArch. In my case, it is useful to log experiments.