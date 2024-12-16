import argparse

from .data_args import (
    tg471_args,
    tg473_args,
    tg474_args,
    tg475_args, 
    tg476_args,
    tg478_args,
    tg487_args
)

from .model_args import (
    gib_args,
    vgib_args,
    gsat_args,
    pgib_args
)


def add_arguments_from_class(parser, instance):
    """
    클래스 인스턴스의 속성을 argparse 인자로 추가하는 함수
    """
    for key, value in vars(instance).items():
        arg_type = type(value)  # 속성 타입을 가져옴
        parser.add_argument(f"--{key}", type=arg_type, default=value, help=f"Set {key}")


def load_arguments(parser, tg_num: int, model: str):
    # parser = argparse.ArgumentParser()

    if tg_num == 471:
        add_arguments_from_class(parser, tg471_args)
        if model == 'gib':
            gib_args.beta = 0.5; gib_args.pp_weight = 0.3; gib_args.inner_loop = 100
            add_arguments_from_class(parser, gib_args)
        elif model == 'vgib':
            vgib_args.mi_weight = 0.; vgib_args.con_weight = 0.
            add_arguments_from_class(parser, vgib_args)
        elif model == 'gsat':
            gsat_args.pred_loss_coef = 0.; gsat_args.info_loss_coef = 0.
            gsat_args.fix_r = False; gsat_args.final_r = 0.; gsat_args.init_r = 0.
            gsat_args.decay_interval = 0.; gsat_args.decay_r = 0.
            add_arguments_from_class(parser, gsat_args)
        elif model == 'pgib':
            pgib_args.alpha1 = 0.; pgib_args.alpha2 = 0.
            pgib_args.pp_weight = 0.; pgib_args.con_weight = 0.
            add_arguments_from_class(parser, pgib_args)
    
    elif tg_num == 473:
        add_arguments_from_class(parser, tg473_args)
        if model == 'gib':
            gib_args.inner_loop = 70; gib_args.beta = 0.1; gib_args.pp_weight = 0.1
            add_arguments_from_class(parser, gib_args)
        elif model == 'vgib':
            vgib_args.mi_weight = 0.; vgib_args.con_weight = 0.
            add_arguments_from_class(parser, vgib_args)
        elif model == 'gsat':
            gsat_args.pred_loss_coef = 0.; gsat_args.info_loss_coef = 0.
            gsat_args.fix_r = False; gsat_args.final_r = 0.; gsat_args.init_r = 0.
            gsat_args.decay_interval = 0.; gsat_args.decay_r = 0.
            add_arguments_from_class(parser, gsat_args)
        elif model == 'pgib':
            pgib_args.alpha1 = 0.; pgib_args.alpha2 = 0.
            pgib_args.pp_weight = 0.; pgib_args.con_weight = 0.
            add_arguments_from_class(parser, pgib_args)
    
    elif tg_num == 474:
        add_arguments_from_class(parser, tg474_args)
        if model == 'gib':
            gib_args.inner_loop = 100; gib_args.beta = 0.1; gib_args.pp_weight = 0.9
            add_arguments_from_class(parser, gib_args)
        elif model == 'vgib':
            vgib_args.mi_weight = 0.; vgib_args.con_weight = 0.
            add_arguments_from_class(parser, vgib_args)
        elif model == 'gsat':
            gsat_args.pred_loss_coef = 0.; gsat_args.info_loss_coef = 0.
            gsat_args.fix_r = False; gsat_args.final_r = 0.; gsat_args.init_r = 0.
            gsat_args.decay_interval = 0.; gsat_args.decay_r = 0.
            add_arguments_from_class(parser, gsat_args)
        elif model == 'pgib':
            pgib_args.alpha1 = 0.; pgib_args.alpha2 = 0.
            pgib_args.pp_weight = 0.; pgib_args.con_weight = 0.
            add_arguments_from_class(parser, pgib_args)
    
    elif tg_num == 475:
        add_arguments_from_class(parser, tg475_args)
        if model == 'gib':
            gib_args.inner_loop = 50; gib_args.beta = 0.1; gib_args.pp_weight = 0.5
            add_arguments_from_class(parser, gib_args)
        elif model == 'vgib':
            vgib_args.mi_weight = 0.; vgib_args.con_weight = 0.
            add_arguments_from_class(parser, vgib_args)
        elif model == 'gsat':
            gsat_args.pred_loss_coef = 0.; gsat_args.info_loss_coef = 0.
            gsat_args.fix_r = False; gsat_args.final_r = 0.; gsat_args.init_r = 0.
            gsat_args.decay_interval = 0.; gsat_args.decay_r = 0.
            add_arguments_from_class(parser, gsat_args)
        elif model == 'pgib':
            pgib_args.alpha1 = 0.; pgib_args.alpha2 = 0.
            pgib_args.pp_weight = 0.; pgib_args.con_weight = 0.
            add_arguments_from_class(parser, pgib_args)
    
    elif tg_num == 476:
        add_arguments_from_class(parser, tg476_args)
        if model == 'gib':
            gib_args.inner_loop = 50; gib_args.beta = 0.3; gib_args.pp_weight = 0.9
            add_arguments_from_class(parser, gib_args)
        elif model == 'vgib':
            vgib_args.mi_weight = 0.; vgib_args.con_weight = 0.
            add_arguments_from_class(parser, vgib_args)
        elif model == 'gsat':
            gsat_args.pred_loss_coef = 0.; gsat_args.info_loss_coef = 0.
            gsat_args.fix_r = False; gsat_args.final_r = 0.; gsat_args.init_r = 0.
            gsat_args.decay_interval = 0.; gsat_args.decay_r = 0.
            add_arguments_from_class(parser, gsat_args)
        elif model == 'pgib':
            pgib_args.alpha1 = 0.; pgib_args.alpha2 = 0.
            pgib_args.pp_weight = 0.; pgib_args.con_weight = 0.
            add_arguments_from_class(parser, pgib_args)
    
    elif tg_num == 478:
        add_arguments_from_class(parser, tg478_args)
        if model == 'gib':
            gib_args.inner_loop = 50; gib_args.beta = 0.1; gib_args.pp_weight = 0.5
            add_arguments_from_class(parser, gib_args)
        elif model == 'vgib':
            vgib_args.mi_weight = 0.; vgib_args.con_weight = 0.
            add_arguments_from_class(parser, vgib_args)
        elif model == 'gsat':
            gsat_args.pred_loss_coef = 0.; gsat_args.info_loss_coef = 0.
            gsat_args.fix_r = False; gsat_args.final_r = 0.; gsat_args.init_r = 0.
            gsat_args.decay_interval = 0.; gsat_args.decay_r = 0.
            add_arguments_from_class(parser, gsat_args)
        elif model == 'pgib':
            pgib_args.alpha1 = 0.; pgib_args.alpha2 = 0.
            pgib_args.pp_weight = 0.; pgib_args.con_weight = 0.
            add_arguments_from_class(parser, pgib_args)
    
    elif tg_num == 487:
        add_arguments_from_class(parser, tg487_args)
        if model == 'gib':
            gib_args.inner_loop = 30; gib_args.beta = 0.1; gib_args.pp_weight = 0.9
            add_arguments_from_class(parser, gib_args)
        elif model == 'vgib':
            vgib_args.mi_weight = 0.; vgib_args.con_weight = 0.
            add_arguments_from_class(parser, vgib_args)
        elif model == 'gsat':
            gsat_args.pred_loss_coef = 0.; gsat_args.info_loss_coef = 0.
            gsat_args.fix_r = False; gsat_args.final_r = 0.; gsat_args.init_r = 0.
            gsat_args.decay_interval = 0.; gsat_args.decay_r = 0.
            add_arguments_from_class(parser, gsat_args)
        elif model == 'pgib':
            pgib_args.alpha1 = 0.; pgib_args.alpha2 = 0.
            pgib_args.pp_weight = 0.; pgib_args.con_weight = 0.
            add_arguments_from_class(parser, pgib_args)
    
    else: raise ValueError(f'TG {args.tg_num} not supported.')

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    
    return args