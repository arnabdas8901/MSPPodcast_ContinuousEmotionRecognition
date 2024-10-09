import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
sys.path.append("/home/adas/Projects/Wav_LM/")
import yaml
import torch
import click
import shutil
import warnings
import os.path as osp
from munch import Munch
from model import TLTRModel
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from data_loader import build_webdataloader

warnings.simplefilter('ignore')


import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True

@click.command()
@click.option('-p', '--config_path', default='/home/adas/Projects/Wav_LM/wavlm_prediction/wavlm_config.yml', type=str)

def main(config_path):
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 10)
    device = config.get('device', 'cpu')
    epochs = config.get('epochs', 100)
    save_freq = config.get('save_freq', 20)
    fp16_run = config.get('fp16_run', False)
    train_shad_urls = config.get('train_shad_urls', None)
    test_shad_urls = config.get('test_shad_urls', None)
    model_mode = config.get("model_mode", "tl_down_tr_512_1_8")

    train_dataloader = build_webdataloader(train_shad_urls,
                                        batch_size=batch_size,
                                        num_workers=8,
                                        device=device,
                                        )
    val_dataloader = build_webdataloader(test_shad_urls,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=8,
                                      device=device,
                                      )

    model = TLTRModel(mode=model_mode)
    lr = config['optimizer_params'].get('lr', 1e-3)
    weight_decay = config['optimizer_params'].get('weight_decay', 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99), eps=1e-09)
    trainer = Trainer(args=Munch(config['loss_params']),
                      model=model,
                      optimizer=optimizer,
                      device=device,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      logger=logger,
                      fp16_run=fp16_run)

    for _ in range(1, epochs+1):
        epoch = trainer.epochs
        train_results = trainer._train_epoch()
        eval_results = trainer._eval_epoch()
        results = train_results.copy()
        results.update(eval_results)
        logger.info('--- epoch %d ---' % epoch)
        txt = ''
        for key, value in results.items():
            if isinstance(value, float):
                txt = txt + key + ':'+ format(value, ".4f") + '  '
                writer.add_scalar(key, value, epoch)
            else:
                for v in value:
                    writer.add_figure('eval_spec', v, epoch)
        logger.info(txt)
        if (epoch % save_freq) == 0:
            trainer.save_checkpoint(osp.join(log_dir, 'epoch_%05d.pth' % epoch))

    return 0

if __name__=="__main__":
    main()

