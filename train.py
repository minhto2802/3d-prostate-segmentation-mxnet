import os

try:
    SEED = int(os.getenv('SEED'))
except:
    SEED = 0

os.urandom(SEED)

os.environ['MXNET_ENFORCE_DETERMINISM'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

from gluoncv.utils.random import seed

seed(SEED)

import mxnet as mx

[mx.random.seed(SEED, ctx=mx.gpu(i)) for i in range(mx.context.num_gpus())]
mx.random.seed(SEED, ctx=mx.cpu())

from mxnet import ndarray as nd
from argparse import ArgumentParser
from datetime import datetime
import time
from mxboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore")
import pickle
from model import Segmentation


def parse_args():
    """Get commandline parameters"""
    parser = ArgumentParser('RadPath Arguments')
    parser.add_argument('-expn', '--experiment_name', type=str, default='pix2pix_uda_v1')
    parser.add_argument('-rid', '--run_id', type=str, default='999')
    parser.add_argument('-gid', '--gpu_id', type=str, default='0')
    parser.add_argument('-ngpu', '--num_gpus', type=int, default=0)
    parser.add_argument('-ep', '--epochs', type=int, default=99999)
    parser.add_argument('--total_iter', type=int, default=1000)
    parser.add_argument('--checkpoint_iter', type=int, default=1999)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_downs', type=int, default=4)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument('--l_type', type=str, default='dice')
    parser.add_argument("--generator", type=str, default="dmnet")
    parser.add_argument('--no_augmentation', action='store_true')
    parser.add_argument('--not_augment_values', action='store_true')
    parser.add_argument("--norm_0mean", action='store_true')
    parser.add_argument("--initializer", type=str, default='none')
    parser.add_argument("--dtype", type=str, default='float32')
    parser.add_argument("--num_fpg", type=int, default=8)
    parser.add_argument("--growth_rate", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--validation_start", type=int, default=0)
    # LR scheduler parameters
    parser.add_argument('--lr_scheduler', type=str, default='factor')
    parser.add_argument('--warmup_mode', type=str, default='linear')
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--max_lr', type=float, default=1e-2)
    parser.add_argument('--lr_step', type=float, default=9999, help='For factor learning rate scheduler')
    parser.add_argument('--lr_steps', type=str, default='9999', help='For multifactor learning rate scheduler')
    parser.add_argument('--lr_factor', type=float, default=1)
    parser.add_argument('--finish_lr', type=float, default=1e-7)
    parser.add_argument('--cycle_length', type=int, default=1000)
    parser.add_argument('--stop_decay_iter', type=int, default=10000)
    parser.add_argument('--final_drop_iter', type=int, default=11000)
    parser.add_argument('--cooldown_length', type=int, default=5000)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--warmup_begin_lr', type=float, default=1e-5)
    parser.add_argument('--inc_fraction', type=float, default=0.9)
    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--cycle_length_decay', type=float, default=.95)
    parser.add_argument('--cycle_magnitude_decay', type=float, default=.98)
    parser.add_argument('--show_generator_summary', action='store_true')
    parser.add_argument('--discriminator_update_interval', type=int, default=1)
    parser.add_argument('--norm_type', type=str, default='batch', help='batch | group | instance')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument("--base_channel_drnn", type=int, default=8,
                        help='number of channels in the first layer of DRNN')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model = Segmentation(args)
    sw = SummaryWriter(logdir='%s' % model.result_folder_logs, flush_secs=5)
    # print(sw.get_logdir())
    first_iter = True  # A trick to use running losses
    best_score = -9999
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
    # logging.basicConfig(level=print)

    tic = time.time()
    btic = time.time()
    count = 0
    # model.create_net()
    model.train_iter._current_it = 0
    # Calculate the number of images increasing every level
    for epoch in range(args.epochs):
        model.current_epoch = epoch

        for i, batch in enumerate(model.train_iter):
            model.current_it = model.trainerG.optimizer.num_update
            model.train_iter._current_it = model.current_it
            A_rp, wp = batch

            model.set_inputs(A_rp=A_rp, wp=wp)
            model.optimize_G()
            # Compute running loss
            model.update_running_loss(
                first_iter=first_iter)  # running_loss attributes will be created in the first iter
            first_iter = False

            # Print log infomation every ten batches
            if (model.current_it + 1) % args.log_interval == 0:
                print('speed: {} samples/s'.format(args.batch_size / ((time.time() - btic) / args.log_interval)))
                print('Segmentation loss = %.5f at iter %d epoch %d'
                      '[current_lr=%.8f, it=%d]'
                      % (nd.mean(nd.concatenate(list(model.loss_G))).asscalar(), model.current_it, epoch,
                         model.trainerG.learning_rate, model.trainerG.optimizer.num_update))
            btic = time.time()
            sw.add_scalar('learning_rate', model.trainerG.learning_rate,
                          global_step=model.trainerG.optimizer.num_update)
            # Hybridize networks to speed-up computation
            if (i + epoch) == 0:
                model.hybridize_networks()

            new_best = False
            if ((model.current_it + 1) % model.val_interval == 0) & (model.current_it >= args.validation_start):
                val_data = model.validate()
                print('          [Validation] loss_to_ground_truth: %.4f' % model.running_loss_seg_val)
                # Update mxboard
                metric_list = model.update_mxboard(sw=sw, epoch=model.current_it, val_data=val_data, best_score=best_score)
                score = metric_list['dice']
                print('Current score (Dice): %.4f' % score)
                # Save models after each epoch
                if score > best_score:
                    new_best = True
                    best_score = score
                print('time: %4f' % (time.time() - tic))
                tic = time.time()

            model.update_running_loss(num_batch=model.current_it + 1)

            if (model.current_it == args.checkpoint_iter) or (new_best and (best_score > .89)):
                model.save_checkpoints()
                model.result_folder_checkpoint_iter = '%s/iter_%04d' % (
                    model.result_folder_checkpoint, model.current_it)
                model.result_folder_inference = model.result_folder_checkpoint_iter.replace('checkpoints', 'inference')
                if not os.path.exists(model.result_folder_inference):
                    os.makedirs(model.result_folder_inference)
                with open('%s/test_data' % model.result_folder_inference, 'wb') as fp:
                    pickle.dump(val_data, fp)
            if model.current_it >= args.total_iter:
                exit()
