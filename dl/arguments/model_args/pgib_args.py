class PGIBParser():
    def __init__(self, alpha1 = None, alpha2 = None, pp_weight = None, con_weight = None):
        super().__init__()
        
        self.count = 1
        self.merge_p = 0.3
        self.proj_epochs = 50
        self.proto_percnetile = 0.1
        self.early_stopping = 10000
        self.warm_epochs = 10
        self.num_prototypes_per_class = 7
        self.share = True
        
        self.pp_weight = pp_weight
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.con_weight = con_weight


pgib_args = PGIBParser()
