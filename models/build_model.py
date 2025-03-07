# Copyright 2022 CircuitNet. All rights reserved.

import models

def build_model(arg):
    opt = vars(arg)
    model = models.__dict__[opt['model_type']](**opt)
    model.init_weights(**opt)
    if opt['test_mode']:
        model.eval()
    return model
