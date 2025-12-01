import segmentation_models_pytorch as smp

def get_shape_model(config):
    """
    Config dosyasından parametreleri okuyup 1 kanallı U-Net++ oluşturur.
    """
    params = config['model']
    
    model = smp.UnetPlusPlus(
        encoder_name=params['encoder_name'],        
        encoder_weights=params['encoder_weights'], 
        in_channels=params['in_channels'],   # 1
        classes=params['classes'],           # 1
        activation=None # Loss fonksiyonunda 'from_logits=True' kullanacağız
    )
    
    return model