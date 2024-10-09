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
from baseline import EmotionModel
from dataset import build_dataloader
from trainer import TrainerBaseline
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, Wav2Vec2Config

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
@click.option('-p', '--config_path', default='/home/adas/Projects/Wav_LM/Configs/config_baseline.yml', type=str)

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
    train_path = config.get('train_data', None)
    val_path = config.get('val_data', None)
    regressor_dim_out = config.get("regressor_dim_out", 3)
    checkpoint_path = config.get("checkpoint_path", "")
    feature_len = config.get("feature_len", 1024)

    train_list, val_list = get_data_path_list(train_path, val_path)
    train_dataloader = build_dataloader(train_list,
                                        batch_size=batch_size,
                                        num_workers=12,
                                        device=device,
                                        )
    val_dataloader = build_dataloader(val_list,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=12,
                                      device=device,
                                      )

    config_w2v2 = Wav2Vec2Config.from_pretrained('audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim', _from_auto=False,
                _from_pipeline=None)
    config_w2v2._name_or_path = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    model = EmotionModel(config_w2v2)
    #model.load_state_dict(torch.load("/netscratch/adas/experments/EmoRecog/Baseline/w2v2_modelWeights.pt"), strict=False)
    if len(config.get("pretrained_path", "")) > 0:
        model.load_state_dict(torch.load(config.get("pretrained_path")))
    model.to(device)
    lr = config['optimizer_params'].get('lr', 1e-3)
    weight_decay = config['optimizer_params'].get('weight_decay', 1e-4)
    #optimizer = torch.optim.Adam(model.regression_head.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99), eps=1e-09)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    """spk_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":device})
    spk_encoder.eval()
    spk_encoder.to(device)"""
    trainer = TrainerBaseline(args=Munch(config['loss_params']),
                      model=model,
                      optimizer=optimizer,
                      device=device,
                      train_dataloader=train_dataloader,
                      val_dataloader=val_dataloader,
                      logger=logger,
                      fp16_run=fp16_run)

    for _ in range(1, epochs+1):
        epoch = trainer.epochs
        warnings.filterwarnings("ignore")
        train_results = trainer._train_epoch()
        eval_results = trainer._eval_epoch()
        results = train_results.copy()
        results.update(eval_results)
        logger.info('--- epoch %d ---' % epoch)
        txt = ''
        for key, value in results.items():
            if isinstance(value, float):
                txt = txt + key + ':'+ format(value, ".4f") + '  '
                #logger.info('%-15s: %.4f' % (key, value))
                writer.add_scalar(key, value, epoch)
            else:
                for v in value:
                    writer.add_figure('eval_spec', v, epoch)
        logger.info(txt)
        if (epoch % save_freq) == 0:
            trainer.save_checkpoint(osp.join(log_dir, 'epoch_%05d.pth' % epoch))

    return 0



def get_data_path_list(train_path, val_path):
    with open(train_path, 'r') as f:
        train_list = f.readlines()
    with open(val_path, 'r') as f:
        val_list = f.readlines()

    return train_list, val_list

if __name__=="__main__":
    main()

