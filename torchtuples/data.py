'''
Small changes to the pytorch dataloader. 
Now works with with batches (slicing rather than loop over indexes), 
and can potentially stop workers from shutting down at each batch.
'''
# import random
import warnings
import copy
import torch
import torch.utils.data
import torchtuples


def identity_collate_fn(x):
    return x


class DataLoaderBatch(torch.utils.data.dataloader.DataLoader):
    __doc__ = ("""A hacky version to speed up pytorch's DataLoader.""" + 
               torch.utils.data.dataloader.DataLoader.__doc__)
    # This is a hack that will hopefully be removed from future implementations.
    # The idea is to let the DataSet read a batch at a time, instead of the torch approach of
    # reading one element in the batch at a time (with a loop). For numpy datasets this is much
    # faster.
    # The hack works by setting `self._auto_collation` to False as thils will cause `_MapDatasetFetcher` 
    # to not iterate over the indices in the batch. However, we need to rewrite `self._index_sampler` so
    # we still use a `self.batch_sampler` instead of `self.sampler` when `self._auto_collation` is False.
    def __iter__(self):
        self._done_init = True  # A flag set to change the behavior of _auto_collation
        # self.collate_fn = lambda x: x  # Identity function, because we don't want it do anything.
        self.collate_fn = identity_collate_fn  # Identity function, because we don't want it do anything.
        return super().__iter__()

    @property
    def _auto_collation(self):
        if hasattr(self, '_done_init'):
            return False  # Return false when called by __iter__
        else:
            return super()._auto_collation
    
    @property
    def _index_sampler(self):
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # (see _utils/fetch.py) to read data at each time. This would be
        # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
        # We can't change `.sampler` and `.batch_sampler` attributes for BC
        # reasons.
        if super()._auto_collation: # This will return True even if we have `self._done_init`.
            return self.batch_sampler
        else:
            return self.sampler


if (torch.__version__ >= '1.1.0') and (torch.__version__ < '1.2.0'):
    from torchtuples import _legacy_v1_1_0
    DataLoaderBatch = _legacy_v1_1_0.DataLoaderSlice


class DataLoaderSlice(DataLoaderBatch):
    def __init__(self, *args, **kwargs):
        warnings.warn("Use `DataLoaderBatch` instead. `DataLoaderSlice` will be removed", DeprecationWarning)
        super().__init__(*args, **kwargs)


class RandomSamplerContinuous(torch.utils.data.dataloader.RandomSampler):
    """Samples elements randomly, without replacement, and continues for ever.

    Arguments:
        data_source (Dataset): dataset to sample from
    """
    def __iter__(self):
        while True:
            for i in iter(torch.randperm(len(self.data_source)).long()):
                yield i


class DatasetTuple(torch.utils.data.Dataset):
    """Dataset where input and target can be tuples.

    Arguments:
        input {tuple or list} -- What is passed to the network (list of x tensors.)
        target {tuple or list} -- Label information passed to the loss function
            (list of y tensors)
    """
    def __init__(self, *data):
        self.data = torchtuples.tuplefy(*data)
        if not self.data.apply(lambda x: type(x) is torch.Tensor).flatten().all():
            warnings.warn("All data is not torch.Tensor. Consider fixing this.")

    def __getitem__(self, index):
        if (not hasattr(index, '__iter__')) and (type(index) is not slice):
            index = [index]
        return self.data.iloc[index]

    def __len__(self):
        lens = self.data.lens().flatten()
        if not lens.all_equal():
            raise RuntimeError("Need all tensors to have same lenght.")
        return lens[0]


class DatasetTupleSingle(DatasetTuple):
    """Like DatasetTuple, but does not read in batches and can therefore be used with
    the regular torch.utils.data.DataLoader.

    Dataset where input and target can be tuples.

    Arguments:
        input {tuple or list} -- What is passed to the network (list of x tensors.)
        target {tuple or list} -- Label information passed to the loss function
            (list of y tensors)
    """
    def __getitem__(self, index):
        return self.data.iloc[index]


class DatasetInputOnly:
    """Class for chaning a Dataset contraining inputs and targets
    to only return the inputs.
    Usefurll for predict methods.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index][0]


def dataloader_input_only(dataloader):
    """Create a new dataloader the only returns the inputs and not the targets.
    Useful when calling predict:

        dataloader = ...
        model.fit(dataloader)
        dl_input = dataloader_input_only(dataloader)
        model.predict(dl_inpt)

    See e.g. MNIST examples code.
    """
    if torch.__version__ <= '1.2.0':
        return _dataloader_input_only_v_less_than_1_2_0(dataloader)

    if type(dataloader.sampler) is not torch.utils.data.sampler.SequentialSampler:
        warnings.warn("Dataloader might not be deterministic!")
    dl = dataloader
    new_dataset = DatasetInputOnly(dl.dataset)
    dl_new = type(dl)(new_dataset, batch_sampler=dl.batch_sampler, num_workers=dl.num_workers,
                      collate_fn=dl.collate_fn, pin_memory=dl.pin_memory, drop_last=dl.drop_last,
                      timeout=dl.timeout, worker_init_fn=dl.worker_init_fn,
                      multiprocessing_context=dl.multiprocessing_context)
    return dl_new

def _dataloader_input_only_v_less_than_1_2_0(dataloader):
    if type(dataloader.sampler) is not torch.utils.data.sampler.SequentialSampler:
        warnings.warn("Dataloader might not be deterministic!")
    dl_new = copy.copy(dataloader)
    dl_new.dataset = DatasetInputOnly(dl_new.dataset)
    return dl_new
