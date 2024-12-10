class GIBParser():
    def __init__(self):
        super().__init__()
        
        self.num_layers = 3
        self.hidden = 128
        self.epochs = 100
        self.lr = 0.001
        self.lr_decay_factor = 0.5
        self.lr_decay_step_size = 50
        self.inner_loop = 50
        # self.beta = 0.5
        # self.pp_weight = 0.9
        self.beta = 0.1
        self.pp_weight = 0.3


gib_args = GIBParser()
