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
        self.fix_r = fix_r
        self.decay_r = decay_r
        self.decay_interval = decay_interval
        self.final_r = final_r
        self.init_r = init_r


gsat_args = GSATParser()
