"""
modified from UNITER codebase

A meta data loader for sampling from different datasets / training tasks
A prefetch loader to speedup data loading
"""
import random

import torch
from torch.utils.data import DataLoader
from src.utils.distributed import any_broadcast


class MetaLoader(object):
    """ wraps multiple data loader """
    def __init__(self, loaders, distributed=False):
        assert isinstance(loaders, dict)
        self.name2loader = {}
        self.name2iter = {}
        self.sampling_pools = []
        n_batches_in_epoch = 0
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r = l
            elif isinstance(l, DataLoader):
                r = 1
            else:
                raise ValueError()
            n_batches_in_epoch += len(l.dataset) * r / l.batch_size
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.sampling_pools.extend([n]*r)
        self.n_batches_in_epoch = n_batches_in_epoch
        self.distributed = distributed

    def __iter__(self):
        """ this iterator will run indefinitely """
        task = self.sampling_pools[0]
        while True:
            task = random.choice(self.sampling_pools)
            if self.distributed:
                # make sure all process is training same task
                task = any_broadcast(task, 0)
            iter_ = self.name2iter[task]
            try:
                batch = next(iter_)
            except StopIteration:
                iter_ = iter(self.name2loader[task])
                batch = next(iter_)
                self.name2iter[task] = iter_

            yield task, batch


class PrefetchLoader(object):
    """
    overlap compute and cuda data transfer
    (copied and then modified from nvidia apex)
    """
    def __init__(self, loader, img_normalize=None):
        self.loader = loader
        #self.stream = torch.cuda.Stream()
        self.img_normalize = img_normalize

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            is_tuple = isinstance(batch, tuple)
            if is_tuple:
                task, batch = batch
            batch["visual_inputs"] = batch["visual_inputs"].float()
            if self.img_normalize is not None:
                batch["visual_inputs"] = self.img_normalize(
                    batch["visual_inputs"])
            if is_tuple:
                yield task, batch
            else:
                yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        # if record_stream() doesn't work, another option is to make sure
        # device inputs are created on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input,
        #                                        device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target,
        #                                         device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use
        # by the main stream at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        #with torch.cuda.stream(self.stream):
        #    self.batch = move_to_cuda(self.batch)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this
            # side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

    def next(self, it):
        #torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        #if batch is not None:
        #    record_cuda_stream(batch)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method
