import segmentation_models_pytorch as smp

def get_model(config):

    params = config['model']
    
    model = smp.UnetPlusPlus(
        encoder_name=params['encoder_name'],        
        encoder_weights=params['encoder_weights'], 
        in_channels=params['in_channels'],   # 1
        classes=params['classes'],           # 1
        activation=None 
    )
    
    return model