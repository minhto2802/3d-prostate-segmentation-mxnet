

def get_optimizer_dict(_optimizer='adam', wd=1e-5, lr=1e-3, beta1=0.9, momentum=0.9,
                       lr_scheduler=None):
    """Get optimizer parameters dictionary corresponding to the optimizer"""
    optimizers = {
        'adadelta': {'rho': .9, 'epsilon': 1e-5, 'wd': wd},
        'adagrad': {'eps': 1e-7, 'learning_rate': lr, 'wd': wd},
        'adam': {'learning_rate': lr, 'beta1': beta1, 'beta2': .99, 'epsilon': 1e-8},
        'lookaheadadam': {'learning_rate': lr, 'beta1': beta1, 'beta2': .99, 'epsilon': 1e-8},
        'adamax': {'learning_rate': lr, 'beta1': beta1, 'beta2': .999},
        'dcasgd': {'learning_rate': lr, 'wd': wd, 'momentum': .0, 'lamda': .04},
        'ftml': {'learning_rate': lr, 'beta1': beta1, 'beta2': .99, 'epsilon': 1e-8, 'wd': wd},
        'ftrl': {'lamda1': 0.01, 'learning_rate': lr, 'beta': 1},
        'lbsgd': {},
        'NAG': {'learning_rate': lr, 'wd': wd, 'momentum': .0},
        'ndabs': {},
        'nadam': {'learning_rate': lr, 'beta1': beta1, 'beta2': .99, 'epsilon': 1e-8, 'schedule_decay': 0.004},
        'RMSProp': {'learning_rate': lr, 'gamma1': beta1, 'gamma2': .9, 'epsilon': 1e-3, 'centered': False},
        'SGD': {'learning_rate': lr, 'wd': wd, 'momentum': momentum},
        'lookaheadsgd': {'learning_rate': lr, 'wd': wd, 'momentum': momentum},
        'sgld': {},
        'signum': {'learning_rate': lr, 'wd': wd, 'momentum': momentum},
    }

    optimizer_dict = optimizers[_optimizer]
    optimizer_dict['lr_scheduler'] = lr_scheduler
    return optimizer_dict
