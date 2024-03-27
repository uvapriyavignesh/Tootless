import constant as co


def neural_network_blue_print(
    input_node_count,
    activation_fnc,
    output_node_count,
    hidden_layer_count=1,
    hidden_layer_node_count=10,
    hidden_layer_activation=None,
    output_layer_activation=None,
):
    if [input_node_count, activation_fnc, output_node_count].__contains__(None):
        raise Exception(
            f"[input_node_count,activation_fnc,output_node_count] : {str([input_node_count,activation_fnc,output_node_count])}  should not be None"
        )
    blue_print = {
        co.INPUT_NODE_COUNT: input_node_count,
        co.ACTIVATION_FUNCTION: activation_fnc,
        co.OUTPUT_NODE_COUNT: output_node_count,
        co.HIDDEN_LAYER_COUNT: hidden_layer_count,
        co.HIDDEN_LAYER_NODE_COUNT: hidden_layer_node_count,
    }
    if hidden_layer_activation is not None:
        blue_print[co.ACTIVATION_FUNCTION_HIDDEN_LAYER] = hidden_layer_activation
    if output_layer_activation is not None:
        blue_print[co.ACTIVATION_FUNCTION_OUTPUT_LAYER] = output_layer_activation
    return blue_print
