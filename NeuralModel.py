import torch.nn as nn
import constant as co


class NeuralNetwork(nn.Module):
    def __init__(self, blue_print):
        super(NeuralNetwork, self).__init__()
        self.blue_print = blue_print
        self.hidden_layers = nn.ModuleList()
        for i in range(self.blue_print.get(co.HIDDEN_LAYER_COUNT) + 1):
            if i == 0:
                self.hidden_layers.append(
                    nn.Linear(
                        self.blue_print.get(co.INPUT_NODE_COUNT),
                        self.blue_print.get(co.HIDDEN_LAYER_NODE_COUNT),
                    )
                )
            else:
                self.hidden_layers.append(
                    nn.Linear(
                        self.blue_print.get(co.HIDDEN_LAYER_NODE_COUNT),
                        self.blue_print.get(co.HIDDEN_LAYER_NODE_COUNT),
                    )
                )

        self.out_layer = nn.Linear(
            self.blue_print.get(co.HIDDEN_LAYER_NODE_COUNT),
            self.blue_print.get(co.OUTPUT_NODE_COUNT),
        )
        self.activation = self.blue_print.get(co.ACTIVATION_FUNCTION)()
        self.hidden_active = (
            self.blue_print.get(co.ACTIVATION_FUNCTION_HIDDEN_LAYER)()
            if self.blue_print.get(co.ACTIVATION_FUNCTION_HIDDEN_LAYER) is not None
            else self.blue_print.get(co.ACTIVATION_FUNCTION)()
        )
        self.out_active = (
            self.blue_print.get(co.ACTIVATION_FUNCTION_OUTPUT_LAYER)()
            if self.blue_print.get(co.ACTIVATION_FUNCTION_OUTPUT_LAYER) is not None
            else self.blue_print.get(co.ACTIVATION_FUNCTION)()
        )

    def forward(self, x):

        for i in range(len(self.hidden_layers)):
            if i == 0:
                x = self.activation(self.hidden_layers[i](x))
            else:
                x = self.hidden_active(self.hidden_layers[i](x))
        x = self.out_active(self.out_layer(x))
        return x
