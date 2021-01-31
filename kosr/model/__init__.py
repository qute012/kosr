from kosr.utils.convert import vocab

def build_model(conf):
    model_type = conf['setting']['model_type']
    device = conf['setting']['device']
    if model_type=='transformer':
        from kosr.model.transformer.model import Transformer as model
    elif model_type=='transducer':
        from kosr.model.transformer.model import Transducer as model
        
    return model(len(vocab), **conf['model']).to(device)