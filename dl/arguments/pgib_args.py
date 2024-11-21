class PGIBParser():
    def __init__(self):
        super().__init__()
        
        self.latent_dim = [128, 128, 128]
        self.epochs = 300
        self.num_prototypes_per_class = 7
        self.readout = 'max'
        self.lr = 0.001
        self.weight_decay = 0
        self.pp_weight = 0.3
        self.folds = 10
        self.warm_epochs = 10
        self.alpha1 = 0.0001
        self.alpha2 = 0.01
        self.con_weight = 5
        self.early_stopping = 10000
        self.proj_epochs = 50
        self.count = 1
        self.share = True
        self.merge_p = 0.3
        self.proto_percnetile = 0.1


pgib_args = PGIBParser()
