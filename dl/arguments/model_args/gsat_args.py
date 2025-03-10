class GSATParser():
    def __init__(self, pred_loss_coef = None, info_loss_coef = None, fix_r = None, decay_r = None, decay_interval = None, final_r = None, init_r = None):
        super().__init__()
        
        self.dropout_p = 0.3
        self.use_edge_attr = True
        self.multi_label = False
        self.learn_edge_att = False
        self.extractor_dropout_p = 0.5
        
        self.pred_loss_coef = pred_loss_coef
        self.info_loss_coef = info_loss_coef
        self.fix_r = False
        self.decay_r = 0.1
        self.decay_interval = 10
        self.final_r = 0.5
        self.init_r = 0.9


gsat_args = GSATParser()
