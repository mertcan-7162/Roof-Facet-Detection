import segmentation_models_pytorch as smp

def get_model(config):

    model_params = config['model']
    
    model = smp.UnetPlusPlus(
        encoder_name=model_params['encoder_name'],        
        encoder_weights=model_params['encoder_weights'],     
        in_channels=model_params['in_channels'],
        classes=model_params['classes'], 
        activation=None 
    )
    
    return model