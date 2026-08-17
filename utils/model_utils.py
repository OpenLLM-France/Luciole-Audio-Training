import logging
import torch 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def check_frozen_layers_peft_model(model):
    for i, layer in enumerate(model.base_model.model.model.layers):
        for name, param in layer.named_parameters():
            logger.info(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")

def freeze_transformer_layers(model, num_layer):
    for i, layer in enumerate(model.model.layers):
        if i < num_layer:
            for param in layer.parameters():
                param.requires_grad = False
            
def freeze_component(component: torch.nn.Module, name: str) -> None:
    if component is None:
        return
    logger.info(f"Freezing {name} for training")
    for param in component.parameters():
        param.requires_grad = False