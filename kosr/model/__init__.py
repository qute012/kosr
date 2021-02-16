from kosr.utils.convert import vocab

def build_model(conf):
    model_type = conf['setting']['model_type']
    device = conf['setting']['device']
    if model_type=='transformer':
        from kosr.model.transformer.model import Transformer as model
    elif model_type=='transformer_joint_ctc':
        from kosr.model.transformer.model import TransformerJointCTC as model
    elif model_type=='transducer':
        from kosr.model.transducer.model import Transducer as model
        
    return model(**conf['model']).to(device)