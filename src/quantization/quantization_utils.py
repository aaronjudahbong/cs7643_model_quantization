# Torch imports.
import torch
import torch.nn as nn
import torch.ao.quantization as tq

SEED = 42
def set_seed(seed = SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def build_qconfig(quantization_type, config):
    mode = config["mode"]
    if mode == "default":
        if quantization_type == "ptq":
            return tq.get_default_qconfig_mapping("fbgemm")
        elif quantization_type == "qat":
            return tq.get_default_qat_qconfig_mapping("fbgemm")
        else:
            raise ValueError(f"Quantization type {quantization_type} not supported - must be 'qat' or 'ptq'")
    
    # Define weight and activation dtype.
    weight_dtype = torch.qint8 if config["weights"]["dtype"] == "qint8" else torch.quint8
    act_dtype = torch.qint8 if config["activations"]["dtype"] == "qint8" else torch.quint8

    # Define quantization ranges based on model.
    # FBGEMM backend uses a reduced range for activations
    if mode == "int8":
        weight_quant_min, weight_quant_max = -128, 127
        act_quant_min, act_quant_max = 0, 127 # reduce_range = True
    elif mode == "int6":
        weight_quant_min, weight_quant_max = -32, 31
        act_quant_min, act_quant_max = 0, 31 # reduce_range = True
    elif mode == "int4":
        weight_quant_min, weight_quant_max = -8, 7
        act_quant_min, act_quant_max = 0, 7 # reduce_range = True
    else:
        raise ValueError(f"Mode {mode} not supported - must be 'int8', 'int6', 'int4', or 'default'")

    # Build QConfig for weights and activations.
    # Define weight scheme based on granularity (keep it symmetric).
    # If per_channel need to use PerChannelMinMaxObserver, else use MinMaxObserver.
    weights_granularity = config["weights"]["granularity"]
    if weights_granularity == "per_channel":
        weight_scheme = torch.per_channel_symmetric
        weight_observer = tq.PerChannelMinMaxObserver.with_args(
            dtype=weight_dtype,
            qscheme=weight_scheme,
            quant_min=weight_quant_min,
            quant_max=weight_quant_max,
        )
    elif weights_granularity == "per_tensor":
        weight_scheme = torch.per_tensor_symmetric
        weight_observer = tq.MinMaxObserver.with_args(
            dtype=weight_dtype,
            qscheme=weight_scheme,
            quant_min=weight_quant_min,
            quant_max=weight_quant_max,
        )
    else:
        raise ValueError(f"Weights granularity {weights_granularity} not supported - must be 'per_channel' or 'per_tensor'")

    # Keep activation scheme fixed (affine).
    act_scheme = torch.per_tensor_affine
    activations_observer = config['activations']['observer']
    if activations_observer == 'minmax':
        act_observer = tq.MinMaxObserver.with_args(
        dtype=act_dtype,
        qscheme=act_scheme,
        quant_min=act_quant_min,
        quant_max=act_quant_max,
    )
    elif activations_observer == 'histogram':
        act_observer = tq.HistogramObserver.with_args(
            dtype=act_dtype,
            qscheme=act_scheme,
            quant_min=act_quant_min,
            quant_max=act_quant_max,
        )
    elif activations_observer == 'movingavg':
        act_observer = tq.MovingAverageMinMaxObserver.with_args(
            dtype=act_dtype,
            qscheme=act_scheme,
            quant_min=act_quant_min,
            quant_max=act_quant_max,
        )
    else:
        raise ValueError(f"Activations observer {activations_observer} not supported - must be 'minmax', 'histogram' or 'movingavg'")

    global_config = tq.QConfig(activation=act_observer, weight=weight_observer)
    # Apply qconfig to entire model/for all layers
    qconfig_mapping = tq.QConfigMapping().set_global(global_config)

    fixed_qconfig = tq.QConfig(
        activation = tq.FixedQParamsObserver.with_args(
            dtype= torch.quint8,
            scale = 1.0 / 256.0,
            zero_point = 0,
        ),
        weight = weight_observer,
    )
    
    # Hardsigmoid has a fixed min/max mapping (need to use FixedQParamsObserver).
    qconfig_mapping.set_object_type(nn.Hardsigmoid, fixed_qconfig)

    if config["skip_aspp"]: 
        # Skip quantizing the ASPP module (only quantize the backbone).
        qconfig_mapping.set_module_name("classifier.0", None) # ASPP module (convs and project).

    if config.get("skip_classifier", False):
        qconfig_mapping.set_module_name("classifier", None) # Full classifier (ASPP + head).

    return qconfig_mapping