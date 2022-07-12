"""Neural Network.
"""
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn


def get_activation_fn(fn_name: str, **kwargs: Any) -> Optional[nn.Module]:
    """Get activation fn.

    Args:
        fn_name: activation function name.
    """
    try:
        activation_fn = getattr(nn, fn_name)
        return activation_fn(**kwargs)
    except AttributeError:
        return None


class MLP(nn.Module):
    """Multi Layer Perceptron."""

    def __init__(
        self,
        kernel_size: List[int],
        activation_fn: List[str],
        input_size: int,
        num_layer: int,
        batch_norm: Optional[List[bool]] = None,
        bias: Optional[List[bool]] = None,
    ) -> None:
        """Initialize it.

        Args:
            kernelt_size: size of kernel. For example, [128, 64, 32]
            activation_fn: activation_function.
                For example, ['linear', 'ReLU', 'ReLU']
            input_size: input tensor size.
            num_layer: The depth of MLP.
            batch_norm: If it set to None, return List of False
                which length is num_layer.
            bias: If it is set to None, return List of False
                which length is num_layer.
        """
        super(MLP, self).__init__()
        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.activation_fn = activation_fn
        self.input_size = input_size
        self.batch_norm = batch_norm or [False for _ in range(num_layer)]
        self.bias = bias or [True for _ in range(num_layer)]
        assert (
            len(self.kernel_size) != self.num_layer
        ), f"{self.kernel_size} is not valid setting."
        assert (
            len(self.activation_fn) != self.num_layer
        ), f"{self.activation_fn} is not valid setting."
        assert (
            len(self.batch_norm) != self.num_layer
        ), f"{self.batch_norm} is not valid."
        assert (
            len(self.bias) != self.num_layer
        ), f"{self.bias} is not valid."

    def build_model(self):
        """Build Model."""
        input_tensor_size = self.input_size
        for layer, (
            kernel_size,
            activation_fn_name,
            batch_norm,
            bias,
        ) in enumerate(
            zip(
                self.kernel_size,
                self.activation_fn,
                self.batch_norm,
                self.bias,
            )
        ):
            self.add_module(
                "MLP_" + str(layer + 1),
                nn.Linear(input_tensor_size, kernel_size, bias=bias),
            )

            if batch_norm:
                self.add_module(
                    "BN_" + str(layer + 1), nn.BatchNorm1d(kernel_size)
                )
            activation_fn = get_activation_fn(activation_fn_name)
            if activation_fn is not None:
                self.add_module(
                    "Activation_fn_" + str(layer + 1), activation_fn
                )
            input_tensor_size = kernel_size

    def forward(self, feature: List[torch.Tensor]) -> torch.Tensor:
        """Forward it."""
        assert isinstance(feature, list), "Class of feature must be List."
        feature = feature[0]
        for layer in self.children():
            feature = layer.forward(feature)
        return feature


class CNN2D(nn.Module):
    """Convolutional Neurl Networ 2D."""

    def __init__(
        self,
        kernel_size: List[int],
        unit_size: List[int],
        activation_fn: List[str],
        input_size: int,
        num_layer: int,
        padding: Optional[List[int]] = None,
        stride: Optional[List[int]] = None,
        batch_norm: Optional[List[bool]] = None,
        bias: Optional[List[bool]] = None,
        wid: Optional[int] = None,
        hei: Optional[int] = None,
    ) -> None:
        """Initialize it."""
        super(CNN2D, self).__init__()

        self.input_size = input_size

        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm or [False for _ in range(num_layer)]
        self.bias = bias or [True for _ in range(num_layer)]

        self.unit_size = unit_size
        self.padding = padding or [0 for _ in range(self.num_layer)]
        self.stride = stride or [1 for _ in range(self.num_layer)]
        self.activation_fn = activation_fn
        self._conv = nn.Conv2D
        self._batch_norm = nn.BatchNorm2d
        self._wid = wid
        self._hei = hei

        # Assertion.
        assert (
            len(self.kernel_size) != self.num_layer
        ), f"{self.kernel_size} is not valid setting."
        assert (
            len(self.activation_fn) != self.num_layer
        ), f"{self.activation_fn} is not valid setting."
        assert (
            len(self.padding) != self.num_layer
        ), f"{self.padding} is not valid setting."
        assert (
            len(self.stride) != self.num_layer
        ), f"{self.stride} is not valid setting."
        assert (
            len(self.batch_norm) != self.num_layer
        ), f"{self.batch_norm} is not valid."
        assert (
            len(self.bias) != self.num_layer
        ), f"{self.bias} is not valid."

    def build_model(self) -> None:
        """Build Model."""
        input_tensor_size = self.input_size
        for layer, (
            unit_size,
            kernel_size,
            stride,
            padding,
            bias,
            batch_norm,
            activation_fn_name,
        ) in enumerate(
            zip(
                self.unit_size,
                self.kernel_size,
                self.stride,
                self.padding,
                self.bias,
                self.batch_norm,
                self.activation_fn,
            )
        ):

            self.add_module(
                "conv_" + str(layer + 1),
                self._conv(
                    input_tensor_size,
                    unit_size,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
            )
            input_tensor_size = unit_size
            if batch_norm:
                self.add_module(
                    "BN_" + str(layer + 1), self._batch_norm(unit_size)
                )
            activation_fn = get_activation_fn(activation_fn_name)
            if activation_fn is not None:
                self.add_module(
                    "Activation_fn_" + str(layer + 1), activation_fn
                )

    def get_input_tensor_size(self) -> int:
        """Get input tensor size for flattening."""
        assert isinstance(
            self._wid, int
        ), "You must specify wid and hei initilizing it."
        dummy_feature = torch.zeros(
            (1, self.input_size, self._wid, self._hei)
        )
        embedding = self.forward([dummy_feature])
        flatten_embedding = embedding.view((1, -1))
        input_tensor_size = flatten_embedding.shape[-1]
        return input_tensor_size

    def forward(self, feature: List[torch.Tensor]) -> torch.Tensor:
        """Forwarding it."""
        assert isinstance(feature, list), "Class of feature must be List."
        feature = feature[0]
        for layer in self.children():
            feature = layer.forward(feature)
        return feature


class CNN1D(CNN2D):
    """Convoluational Neural Network for 1D."""

    def __init__(self, **kwargs):
        """Initalize it."""
        super(CNN1D, self).__init__(**kwargs)
        self._conv = nn.Conv1D
        self._batch_norm = nn.BatchNorm1d

    def get_input_tensor_size(self) -> int:
        """Get input tensor size for flattening."""
        assert isinstance(
            self._wid, int
        ), "You must specify wid and hei initilizing it."
        dummy_feature = torch.zeros((1, self.input_size, self._wid))
        embedding = self.forward(dummy_feature)
        flatten_embedding = embedding.view((1, -1))
        input_tensor_size = flatten_embedding.shape[-1]
        return input_tensor_size

    def forward(self, feature: List[torch.Tensor]) -> torch.Tensor:
        """Forwarding it."""
        assert isinstance(feature, List), "Class of feature must be List."
        feature = feature[0]
        for layer in self.children():
            feature = layer.forward(feature)
        return feature


class RNN(nn.Module):
    """Recureent Neural Network."""

    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        num_layer: int,
        flatten_mode: bool,
        trainable_init_cell_state: bool = False,
        device: str = "cpu",
    ) -> None:
        """Initialize it."""
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = torch.device(device)
        self.flatten_mode = flatten_mode
        self.num_layer = self.num_layer
        self.rnn = nn.LSTM(input_size, self.hidden_size, self.num_layer)
        self.trainable_init_cell_state = trainable_init_cell_state

        if trainable_init_cell_state:
            # set init hidden state.
            init_hidden_state = torch.randn((num_layer, 1, hidden_size))
            init_hidden_state.data.uniform_(1 / hidden_size).to(
                self.device
            )
            init_hidden_state = nn.parameter.Parameter(init_hidden_state)

            # set init cell state
            init_cell_state = torch.randn((num_layer, 1, hidden_size))
            init_cell_state.data.uniform_(1 / hidden_size).to(self.device)
            init_cell_state = nn.parameter.Parameter(init_hidden_state)
            self.cell_state = (init_hidden_state, init_cell_state)
        else:
            init_cell_state, init_hidden_state = (
                torch.zeros((num_layer, 1, hidden_size)),
                torch.zeros((num_layer, 1, hidden_size)),
            )

        self.init_hidden_state, self.init_cell_state = (
            init_hidden_state,
            init_cell_state,
        )

    def get_cell_state(self, batch_size: int=1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get hidden and cell states.
        
        Args:
            batch_size: output has such shape:
                [num_layer, batch_size, hidden_state]
        """
        # clone한 torch 역시 backward에 기여된다.
        return (self.cell_state[0].clone(), self.cell_state[1].clone())

    def set_cell_state(
        self, cell_state: Tuple[torch.Tensor, tuple.Tensor]
    ):
        """Set hidden and cell states."""
        self.cell_state = cell_state

    def detach_cell_state(self):
        """Detach cell states."""
        self.cell_state = (
            self.cell_state[0].clone().detach(),
            self.cell_state[1].clone().detach(),
        )

    def reset_cell_state(self):
        """Reset zero cell states."""
        self.cell_state = (self.init_hidden_state, self.init_cell_state)

    def forward(self, inputs: List[Any]):
        assert isinstance(inputs, list), "It is not valid."

        if len(inputs) > 1:
            feature, hidden_states = inputs
            self.set_cell_state(hidden_states)
        else:
            feature = inputs[0]
        feature: torch.Tensor

        batch_size = feature.shape[0]

        batched_hidden_state = torch.stack(
            [self.cell_state[0] for _ in range(batch_size)], 1
        )
        batched_cell_state = torch.stack(
            [self.cell_state[1] for _ in range(batch_size)], 1
        )

        embedding, (hidden_state, cell_state) = self.rnn(
            feature, (batched_hidden_state, batched_cell_state)
        )
        if self.flatten_mode:
            embedding = torch.squeeze(embedding, 0)
            self.set_cell_state

        if nDim == 1:
            output, (hn, cn) = self.rnn(state, self.cell_state)
            if self.FlattenMode:
                output = torch.squeeze(output, dim=0)
            self.cell_state = (hn, cn)
        else:
            output, (hn, cn) = self.rnn(state, self.cell_state)
            if self.FlattenMode:
                output = output.view(-1, self.hidden_size)
                output = output.view(-1, self.hidden_size)
            self.cell_state = (hn, cn)

        if self.return_hidden:
            output = output[-1:, :, :]

        # output consists of output, hidden, cell state
        return output


class GRU(nn.Module):
    """
    LSTMNET class는 LSTM을 지원한다.
    LSTM는 다음을 통해 설정할 수 있다.
    configuration:
        args:
            iSize:[int], input의 형태
            num_layer:[int], layer의 갯수, 현재 1밖에 지원안함!
            hiddenSize:[int], cell state의 크기
            Number_Agent:[int], 현재 환경에서 돌아가는 agent의 수
            FlattenMode:[bool], lstm의 output은 <seq, batch, hidden>를
                                                <seq*batch, hidden>로 변환
    """

    def __init__(self, net_info):
        super(GRU, self).__init__()
        self.net_info = net_info
        self.hidden_size = net_info["hiddenSize"]
        self.num_layer = net_info["num_layer"]
        iSize = net_info["iSize"]
        device = net_info["device"]
        self.device = torch.device(device)
        use_init_parameter = (
            self.net_info["use_init_parameter"]
            if "use_init_parameter" in self.net_info.keys()
            else False
        )

        if use_init_parameter:
            a = torch.randn((1, 1, self.hidden_size)).to(self.device)
            a.data.uniform_(-0.08, 0.08)
            self.init_CellState = a
            self.cell_state = self.init_CellState.data
        else:
            self.init_CellState = torch.zeros((1, 1, self.hidden_size)).to(
                self.device
            )

            self.cell_state = self.init_CellState

        self.rnn = nn.GRU(iSize, self.hidden_size, self.num_layer)
        self.FlattenMode = net_info["FlattenMode"]
        try:
            self.return_hidden = net_info["return_hidden"]
        except:
            self.return_hidden = False

    def getCellState(self):
        """
        CellState을 반환한다.
        output:
            dtype:List, (hstate, cstate)
            state:torch.tensor, shape:[1, Agent의 숫자, hiddenSize]
        """
        # clone한 torch 역시 backward에 기여된다.
        return self.cell_state

    def setCellState(self, cellState):
        """
        CellState를 설정한다.
        args:
            cellState:List, (hstate, cstate)
            state:torch.tensor, shape:[1, Agent의 숫자, hiddenSize]
        """
        self.cell_state = cellState

    def detachCellState(self):
        "GRU의 BTTT를 지원하기 위해서는 detaching이 필요하다."
        self.cell_state = self.cell_state.clone().detach()

    def zeroCellState(self, num=1):
        """
        cellState를 zero로 변환하는 과정이다.
        환경이 초기화 되면, lstm역시 초기화 되어야한다.
        """
        self.cell_state = self.init_CellState.data

    def forward(self, state):
        state = state[0]
        if len(state) == 2:
            self.setCellState(state[1])
        nDim = state.shape[0]
        if nDim == 1:
            output, hn = self.rnn(state, self.cell_state)
            if self.FlattenMode:
                output = torch.squeeze(output, dim=0)
            self.cell_state = hn
        else:
            output, hn = self.rnn(state, self.cell_state)
            if self.FlattenMode:
                output = output.view(-1, self.hidden_size)
                output = output.view(-1, self.hidden_size)
            self.cell_state = hn

        if self.return_hidden:
            if output.shape[0] == 1:
                pass
            else:
                output = output[-1:, :, :]

        # output consists of output, hidden, cell state
        return output


class Attention(nn.Module):
    """
    Args:
        iSize[int]: dimension of state
        kernel_size[int]: dimension of weight
        use_bias[Bool]: use bias
        activation[str]: activation
        device[str]: which device?
    """

    def __init__(self, net_info: dict):
        super(Attention, self).__init__()
        self.input_size = net_info["iSize"]
        self.kernel_size = net_info["kernel_size"]
        key = list(net_info.keys())
        self.use_bias = (
            net_info["use_bias"] if "use_bias" in key else False
        )
        self.activation_fnivation = (
            net_info["activation"] if "activation" in key else "tanh"
        )
        self.activation_fnivation = get_activation_fn(
            self.activation_fnivation
        )
        self.device = (
            torch.device(net_info["device"])
            if "device" in key
            else torch.device("cpu")
        )
        self.build_model()

    def build_model(self):
        attention_weight = torch.randn(
            self.input_size, self.kernel_size
        ).to(self.device)
        attention_weight.data.uniform_(-0.08, 0.08)
        self.attention_weight = nn.Parameter(attention_weight)

        attention_weight_sum = torch.randn(self.kernel_size).to(
            self.device
        )
        attention_weight_sum.data.uniform_(-0.08, 0.08)
        self.attention_weight_sum = nn.Parameter(attention_weight_sum)

    def forward(self, x):
        if type(x) == List:
            state = x[0]
        # BATCH, SEQ, DIM
        embedding = torch.matmul(state, self.attention_weight)
        embedding = self.activation_fnivation.forward(embedding)

        embedding_score = torch.matmul(
            embedding, self.attention_weight_sum
        )
        # BATCH, SEQ
        embedding_score = torch.exp(embedding_score)
        sum_attention = torch.sum(embedding_score, dim=-1).view((-1, 1))
        attention_score = embedding_score / (sum_attention + 1e-5)
        attention_score = torch.unsqueeze(attention_score, dim=-1)
        neighbor_state = embedding * attention_score
        neighbor_state = torch.sum(neighbor_state, dim=1)
        # BATCH, HIDDEN
        return neighbor_state


class Cat(nn.Module):
    """
    concat을 지원한다.
    """

    def __init__(self, data):
        super(Cat, self).__init__()

    def forward(self, x):
        return torch.cat(x, dim=-1)


class Stack(nn.Module):
    def __init__(self, data):
        super(Stack, self).__init__()
        self.dim = data["dim"]

    def forward(self, x):
        return torch.stack(x, dim=self.dim)


class Unsequeeze(nn.Module):
    """
    unsequeeze를 지원
    """

    def __init__(self, data):
        super(Unsequeeze, self).__init__()
        key = list(data.keys())
        self.dim = data["dim"]
        self.max_dim = data["max_dim"] if "max_dim" in key else 1000

    def forward(self, x):
        if type(x) == List:
            x = x[0]
        if len(x.shape) < self.max_dim:
            return torch.unsqueeze(x, dim=self.dim)
        else:
            return x


class AvgPooling(nn.Module):
    def __init__(self, data):
        super(AvgPooling, self).__init__()
        stride = data["stride"]
        kernel_size = data["kernel_size"]
        padding = data["padding"]
        self.layer = nn.AvgPool1d(
            kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        if type(x) == List:
            x = x[0]
        return self.layer(x)


class View(nn.Module):
    """
    view를 지원. 이때 view는 shape를 변환하는 것을 의미한다.
    """

    def __init__(self, data):
        super(View, self).__init__()
        self.shape = data["shape"]

    def forward(self, x):
        if type(x) == List:
            x = x[0]
        return x.view(self.shape)


class ViewV2(nn.Module):
    def __init__(self, data):
        super(ViewV2, self).__init__()

    def forward(self, x):
        if type(x) == List:
            x_ = x[1]
            y = List(x[0].cpu().detach().numpy())
        return x_.view(y)


class Select(nn.Module):
    def __init__(self, net_info):
        super(Select, self).__init__()
        self.net_info = net_info
        self.num = self.net_info["num"]

    def forward(self, state):
        if type(state) == List:
            state = state[0]
            if type(state) == List:
                state = state[self.num]
        return state


class Permute(nn.Module):
    def __init__(self, net_info):
        super(Permute, self).__init__()
        self.net_info = net_info
        self.permute = self.net_info["permute"]

    def forward(self, state):
        if type(state) == List:
            state = state[0]
        state = state.permute(self.permute).contiguous()
        return state


class Subtrack(nn.Module):
    def __init__(self, net_info):
        super(Subtrack, self).__init__()
        pass

    def forward(self, state):
        return state[0] - state[1]


class Add(nn.Module):
    def __init__(self, net_info):
        super(Add, self).__init__()
        pass

    def forward(self, state):
        return state[0] + state[1]


class Mean(nn.Module):
    def __init__(self, net_info):
        super(Mean, self).__init__()
        pass

    def forward(self, state):
        return state[0].mean(dim=-1, keepdim=True)
