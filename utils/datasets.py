from mxnet.gluon.data.dataset import Dataset
import numpy as np


class PROMISE12(Dataset):
    def __init__(self, data_list, transform=None, is_val=False, input_size=256,
                 not_augment_values=False, batch_size=None):
        self._transform = transform
        self._data_list = data_list
        self._is_val = is_val
        self._input_size = input_size
        self._not_augment_values = not_augment_values
        self._batch_size = batch_size
        self._count_instance = 0

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        if self._transform is not None:

            aug_data = self._transform(
                np.load(self._data_list[idx]),
                is_val=self._is_val,
                input_size=self._input_size,
                not_augment_values=self._not_augment_values,
            )
            return tuple([d.astype('float32') for d in aug_data])
        else:
            return tuple([np.load(d[idx]) for d in self._data_list])


if __name__ == "__main__":
    import pylab as plt
    from skimage.util import montage
    import pickle
    from utils.dataloader import DataLoader
    from utils.batchify_fn import BatchifyFn
    from utils.sampler import BatchSampler, RandomSamplerStratify2Sets
    from utils.transformations import joint_transform
    from glob import glob

    batch_size = 1
    input_size = 256


    def sort_data_list(_data_list):
        """

        :param _data_list:
        :return:
        """
        import re
        idx = [re.search('Case\d+', d).group() for d in _data_list]
        return [_data_list[i] for i in np.argsort(idx)]


    data_list = sort_data_list(glob('../inputs/resampled/training/*.npy'))
    test_list = sort_data_list(glob('../inputs/resampled/test/*.npy'))
    val_list = [data_list[k] for k in [0, 1, 14, 15, 26, 27, 39, 40]]
    train_list = [_data_list for _data_list in data_list if _data_list not in val_list]

    train_dataset = PROMISE12(train_list,
                              input_size=input_size,
                              transform=joint_transform,
                              is_val=False,
                              not_augment_values=False,
                              batch_size=8,
                              )

    val_dataset = PROMISE12(val_list,
                            input_size=input_size,
                            transform=joint_transform,
                            is_val=True
                            )

    test_dataset = PROMISE12(test_list,
                             input_size=input_size,
                             transform=joint_transform,
                             is_val=True
                             )

    sampler = RandomSamplerStratify2Sets(train_list.__len__(), 0, batch_size, 0)
    batch_sampler = BatchSampler(sampler, batch_size, last_batch='discard')

    train_iter = DataLoader(
        train_dataset,
        num_workers=0,
        thread_pool=False,
        batchify_fn=BatchifyFn(batch_size=batch_size).batchify_fn,
        prefetch=None,
        batch_sampler=batch_sampler
    )
    val_iter = DataLoader(val_dataset,
                          batch_size=1,
                          num_workers=0,
                          last_batch='keep',
                          shuffle=False,
                          thread_pool=False,
                          prefetch=None,
                          )
    test_iter = DataLoader(test_dataset,
                          batch_size=1,
                          num_workers=0,
                          last_batch='keep',
                          shuffle=False,
                          thread_pool=False,
                          prefetch=None,
                          )

    for (i, batch) in enumerate(train_iter):
        x, m = batch[0].squeeze().asnumpy(), batch[1].squeeze().asnumpy()
        thr = x[m == 1].max()
        x[x > thr] = thr
        plt.figure(1, (12, 12))
        plt.imshow(montage(x), cmap='gray')
        plt.contour(montage(m))
        plt.title('Test %02d' % i)
        plt.show()

    for (i, batch) in enumerate(test_iter):
        x = batch[0].squeeze().asnumpy()
        plt.figure(1, (12, 12))
        plt.imshow(montage(x), cmap='gray')
        plt.title('Test %02d' % i)
        plt.show()
