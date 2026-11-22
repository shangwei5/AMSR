
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']      # one input: L

    if model == 'plain':
        from models.model_plain import ModelPlain as M

    elif model == 'pixelwise_sr':
        from models.model_pixelwise_sr import ModelPSR as M

    elif model == 'pixelwise_sr_auto':
        from models.model_pixelwise_sr_auto import ModelPSR as M

    elif model == 'pixelwise_sr_auto_v1':
        from models.model_pixelwise_sr_auto_v1 import ModelPSR as M

    elif model == 'pixelwise_sr_mask':
        from models.model_pixelwise_sr_mask import ModelPSR as M

    elif model == 'pixelwise_sr_mask_progress':
        from models.model_pixelwise_sr_mask_progress import ModelPSR as M

    elif model == 'dyconv_sr':
        from models.model_dyconv_sr import ModelPSR as M

    elif model == 'hy_dyconv_sr':
        from models.model_hy_dyconv_sr import ModelPSR as M

    elif model == 'pixelwise_sr_inr':
        from models.model_pixelwise_sr_inr import ModelPSR as M

    elif model == 'pixelwise_sr_visual':
        from models.model_pixelwise_sr_visual import ModelPSR as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
