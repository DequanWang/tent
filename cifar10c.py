import logging

import torch

from robustbench.data import load_cifar10c
from robustbench.utils import clean_accuracy as accuracy

from tent import tent
from conf import cfg, load_cfg_fom_args


def evaluate(cfg_file):
    load_cfg_fom_args(cfg_file=cfg_file,
                      description="CIFAR-10-C evaluation.")
    logger = logging.getLogger(__name__)
    for severity in cfg.CORRUPTION.SEVERITY:
        for corruption_type in cfg.CORRUPTION.TYPE:
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            model = tent(cfg.CORRUPTION.MODEL)
            acc = accuracy(model, x_test, y_test, cfg.OPTIM.BATCH_SIZE)
            logger.info('accuracy [{}{}]: {:.2%}'.format(
                corruption_type, severity, acc))


if __name__ == '__main__':
    evaluate('cifar10c.yaml')
