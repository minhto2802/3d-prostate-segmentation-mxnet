import csv
import os
import pickle
import re
from glob import glob
import numpy as np
import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon.utils import split_and_load as sal
from sklearn.preprocessing import StandardScaler

from utils import metrics
from utils.dataloader import DataLoader
from utils.batchify_fn import BatchifyFn
from utils.datasets import PROMISE12
from utils.learning_rate_scheduler import *
from utils.losses import DiceLoss
from utils.optimizers import get_optimizer_dict
from utils.sampler import BatchSampler, RandomSamplerStratify2Sets
from utils.transformations import joint_transform, just_crop

inits = {
    'none': mx.init.Uniform(),
    'normal': mx.init.Normal(.05),
    'xavier': mx.init.Xavier(magnitude=2.2),
    'he': mx.init.MSRAPrelu(),
}


def sort_data_list(_data_list):
    """

    :param _data_list:
    :return:
    """
    idx = [re.search('Case\d+', d).group() for d in _data_list]
    return [_data_list[i] for i in np.argsort(idx)]


class Init:
    """Initialize training parameters and directories"""

    def __init__(self, args):
        self.__dict__.update(args.__dict__)
        # self.dataset_root = 'datasets/%s/' % self.dataset_name
        if isinstance(self.run_id, int):
            self.result_folder = 'results/%s/run_%03d/' % (self.experiment_name, self.run_id)
        else:
            self.result_folder = 'results/%s/%s/' % (self.experiment_name, self.run_id)
        self.result_folder_checkpoint = '%s/checkpoints' % self.result_folder
        self.result_folder_logs = '%s/logs' % self.result_folder
        folders = [field for field in list(self.__dict__.keys()) if 'folder' in field]
        for folder in folders:
            if not os.path.exists(self.__getattribute__(folder)):
                os.makedirs(self.__getattribute__(folder))

        self.ctx = [mx.gpu(int(i)) for i in self.gpu_id.split(',')]
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        self.batch_size = args.batch_size
        self.save_setting()

        # Load data
        data_list = sort_data_list(glob('inputs/resampled/training/*.npy'))
        val_list = [data_list[k] for k in [0, 1, 14, 15, 26, 27, 39, 40]]
        train_list = [_data_list for _data_list in data_list if _data_list not in val_list]

        # Condition to split dataset
        train_dataset = PROMISE12(train_list,
                                  input_size=self.input_size,
                                  transform=joint_transform,
                                  is_val=self.no_augmentation,
                                  not_augment_values=self.not_augment_values,
                                  batch_size=self.batch_size,
                                  )
        sampler = RandomSamplerStratify2Sets(len(train_list), 0, self.batch_size, 0)
        batch_sampler = BatchSampler(sampler, self.batch_size, last_batch='discard')

        self.train_iter = DataLoader(
            train_dataset,
            num_workers=self.num_workers,
            thread_pool=False,
            batchify_fn=BatchifyFn(batch_size=self.batch_size).batchify_fn,
            prefetch=None,
            batch_sampler=batch_sampler
        )

        val_dataset = PROMISE12(val_list,
                                input_size=self.input_size,
                                transform=joint_transform,
                                is_val=True
                                )
        self.val_iter = DataLoader(val_dataset,
                                   batch_size=1,
                                   num_workers=self.num_workers,
                                   last_batch='keep',
                                   shuffle=False,
                                   thread_pool=False,
                                   prefetch=None,
                                   )
        self.best_metrics = {
            'dice': 0,
        }

    def save_setting(self):
        """save input setting into a csv file"""
        with open('%s/parameters.csv' % self.result_folder, 'w') as f:
            w = csv.writer(f)
            for key, val in self.__dict__.items():
                w.writerow([key, val])

    def load_data(self):
        """Load all Numpy"""
        print('Loading input file...')
        with open("inputs/resampled_combined/training", 'rb') as fp:
            x = pickle.load(fp)
        print('Done!')
        return x


class Segmentation(Init):
    def __init__(self, args):
        super(Segmentation, self).__init__(args=args)
        self.set_lr_scheduler()
        self.set_networks()
        self.def_loss()

    def set_networks(self):
        # Pixel2pixel networks
        n_in = 1
        if self.generator == 'drnn':
            from networks.drnn_3d import DenseMultipathNet, Init as init_net_params
            opts = init_net_params(num_fpg=self.num_fpg, growth_rate=self.growth_rate,
                                   init_channels=self.base_channel_drnn)
            self.netG = DenseMultipathNet(opts)

        self.netG.initialize(inits[self.initializer], ctx=self.ctx, force_reinit=True)

        self.trainerG = gluon.Trainer(self.netG.collect_params(),
                                      optimizer=self.optimizer,
                                      optimizer_params=get_optimizer_dict(
                                          self.optimizer,
                                          lr=self.base_lr,
                                          lr_scheduler=self._lr_scheduler,
                                          wd=self.wd,
                                          beta1=self.beta1,
                                      ))

        # amp.init_trainer(self.trainerG)  # automatic mixed precision
        largest_batch_size = int(np.ceil(self.batch_size / len(self.gpu_id.split(','))))
        if self.show_generator_summary:
            [self.netG.summary(
                nd.random.normal(0, 1, shape=(largest_batch_size, n_in, self.input_size, self.input_size), ctx=ctx)) for
                ctx
                in self.ctx]

    def def_loss(self):
        """"""
        loss_fn = {
            'dice': DiceLoss,
        }

        self.seg_train = loss_fn[self.l_type]()
        self.seg_val = loss_fn[self.l_type]()

    def set_inputs(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, sal(value, ctx_list=self.ctx, even_split=False))

    def set_lr_scheduler(self):
        """Setup learning rate scheduler"""
        self.lr_steps = [int(lr) for lr in self.lr_steps.split(',')]
        schedules = {
            'one_cycle': OneCycleSchedule(
                start_lr=self.min_lr, max_lr=self.max_lr, cycle_length=self.cycle_length,
                cooldown_length=self.cooldown_length, finish_lr=self.finish_lr, inc_fraction=self.inc_fraction,
            ),
            'triangular': TriangularSchedule(
                min_lr=self.min_lr, max_lr=self.max_lr, cycle_length=self.cycle_length, inc_fraction=self.inc_fraction,
            ),
            'factor': mx.lr_scheduler.FactorScheduler(
                step=self.lr_step, factor=self.lr_factor, warmup_mode=self.warmup_mode,
                warmup_steps=self.warmup_steps, warmup_begin_lr=self.warmup_begin_lr, base_lr=self.base_lr,
            ),
            'multifactor': mx.lr_scheduler.MultiFactorScheduler(
                step=self.lr_steps, factor=self.lr_factor, base_lr=self.base_lr, warmup_mode=self.warmup_mode,
                warmup_begin_lr=self.warmup_begin_lr, warmup_steps=self.warmup_steps,
            ),
            'poly': mx.lr_scheduler.PolyScheduler(
                max_update=self.cycle_length, base_lr=self.base_lr, pwr=2, final_lr=self.min_lr,
            ),
            'cycle': CyclicalSchedule(
                TriangularSchedule, min_lr=self.min_lr, max_lr=self.max_lr, cycle_length=self.cycle_length,
                inc_fraction=self.inc_fraction,
                cycle_length_decay=self.cycle_length_decay,
                cycle_magnitude_decay=self.cycle_magnitude_decay,
                final_drop_iter=self.final_drop_iter,
            ),
            'cosine': LinearWarmUp(
                OneCycleSchedule(start_lr=self.min_lr, max_lr=self.max_lr, cycle_length=self.cycle_length,
                                 cooldown_length=self.cooldown_length, finish_lr=self.finish_lr),
                start_lr=self.warmup_begin_lr,
                length=self.warmup_steps,
            )
        }
        self._lr_scheduler = schedules[self.lr_scheduler]

    def optimize_G(self):
        """Optimize generator"""
        with autograd.record():
            self.fake_out = [self.netG(A_rp) for A_rp in self.A_rp]
            self.loss_seg_train = [self.seg_train(fake_out, wp) for fake_out, wp in zip(self.fake_out, self.wp)]
            self.loss_G = self.loss_seg_train
            [loss_G.backward() for loss_G in self.loss_G]

        self.trainerG.step(1, ignore_stale_grad=False)

    def update_running_loss(self, first_iter=False, num_batch=None):
        """Compute running loss"""
        if num_batch is None:
            if first_iter:
                loss_fields = [field for field in self.__dict__.keys() if ('loss' in field) or ('err' in field)]
                self.running_loss_fields = ['running_' + field for field in loss_fields]
                [self.__setattr__(field, 0.) for field in self.running_loss_fields]
            for loss_field in self.running_loss_fields:
                _loss = nd.concatenate(list(self.__getattribute__(loss_field.replace('running_', ''))))
                self.__setattr__(loss_field, (self.__getattribute__(loss_field) + _loss.mean().asscalar()))
        else:
            for loss_field in self.running_loss_fields:
                self.__setattr__(loss_field, (self.__getattribute__(loss_field) / num_batch))

    def update_mxboard(self, sw, epoch, best_score=0, val_data=None):
        """ SHOW STATS AND IMAGES ON TENSORBOARD. THIS SHOULD BE RUN AFTER RUNNNING UPDATE_RUNNING_LOSS """
        for loss_field in self.running_loss_fields:
            _loss = self.__getattribute__(loss_field)
            _loss = _loss.mean().asscalar() if isinstance(_loss, nd.NDArray) else _loss.mean()
            if 'loss_seg' in loss_field:  # True density
                sw.add_scalar('loss/seg_loss', _loss, global_step=epoch)
            else:  # GAN loss
                loss_type = loss_field.split('_')[0] + '_' + \
                            loss_field.split('_')[1] + '_' + \
                            loss_field.split('_')[2]
                sw.add_scalar('loss/' + loss_type, _loss, global_step=epoch)
        if hasattr(self, 'running_loss_seg_val'):
            sw.add_scalar('loss/seg_loss_val', self.running_loss_seg_val, global_step=epoch)

        metric_list = metrics.update_mxboard_metric_v1(sw, data=val_data, global_step=epoch,
                                                       metric_names=[
                                                           'dice'
                                                       ],
                                                       prefix='validation_', best_score=best_score)
        return metric_list

    def validate(self):
        """Perform validation"""
        l = []
        input_list, pred_list, wp_list = [], [], []

        for i, (A_rp, wp) in enumerate(self.val_iter):
            # Inputs to GPUs (or CPUs)
            self.set_inputs(A_rp_val=A_rp, wp_val=wp)
            pred = [self.netG(A_rp_val) for A_rp_val in self.A_rp_val]
            # Split segmentation and regression outputs if multitask learning is used
            pred = nd.concatenate(pred)

            # merge data across all used GPUs
            self.A_rp_val, self.wp_val = [
                nd.concatenate(list(x)) for x in [
                    self.A_rp_val,
                    self.wp_val]
            ]

            input_list.append(self.A_rp_val.asnumpy())
            pred_list.append(pred.asnumpy())
            wp_list.append(self.wp_val.asnumpy())

            l.append(self.seg_val(pred, self.wp_val).asnumpy())

        self.running_loss_seg_val = np.concatenate([*l]).mean()

        return input_list, pred_list, wp_list

    def save_checkpoints(self):
        """Saved parameters"""
        self.result_folder_checkpoint_current_iter = '%s/iter_%04d' % (
            self.result_folder_checkpoint, self.current_it)
        os.makedirs(self.result_folder_checkpoint_current_iter) if not os.path.exists(
            self.result_folder_checkpoint_current_iter) else None

        self.netG_filename = '%s/netG.params' % (self.result_folder_checkpoint_current_iter,)
        self.netG.save_parameters(self.netG_filename)

    def load_checkpoints(self, pretrained_dir=None):
        if pretrained_dir:
            self.netG.load_parameters(pretrained_dir, ctx=self.ctx,
                                      ignore_extra=True)
        else:
            self.netG_filename = '%s/netG.params' % (self.result_folder_checkpoint_iter,)

            """Load parameters from checkpoints"""
            self.netG.load_parameters(self.netG_filename, ctx=self.ctx,
                                      ignore_extra=True)

    def hybridize_networks(self):
        self.netG.hybridize(static_alloc=True, static_shape=True)
