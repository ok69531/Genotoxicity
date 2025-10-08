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
        
        self.rollout = 10                         
        self.high2low = False
        self.c_puct = 5
        self.min_atoms = 5
        self.max_atoms = 10
        self.expand_atoms = 10

        def process_args(self) -> None:
            import os
            self.explain_model_path = os.path.join(self.checkpoint,
                                                self.dataset_name,
                                                f"{self.model_name}_best.pth")
        
        self.reward_method: str = 'mc_l_shapley'                         
        self.local_raduis: int = 4                                     
        self.subgraph_building_method: str = 'zero_filling'
        self.sample_num: int = 100                                    


pgib_args = PGIBParser()
