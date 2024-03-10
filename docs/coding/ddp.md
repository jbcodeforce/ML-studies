# DDP and 

## Distributed Data Parallel with PyTorch

???- info "Updates"
    Created 01/2024

The goal of Distributed Data Parallel is to train model in distributed computers but keep model integrity. PyTorch offers a [DDP library (`torch.distributed`)](https://pytorch.org/tutorials/beginner/dist_overview.html) to facilitate this complex processing on multiple GPU hosts or using multiple machines.

On one host, the model is trained on CPU/GPU, from the complete data set. It processes the forward pass to compute weights, computes the lost, performs the backward propagation for the gradients, then optimizes the gradients. 

Using more hosts, we can split the dataset and send those data to different hosts, which have the same initial model and optimizer function.

![](./diagrams/ddp-1.drawio.png)

Sending different data set to train the different models, leads to different gradients, so different models.

DDP adds a synchronization step before optimizing the gradients so each model has the same weights:

![](./diagrams/ddp-2.drawio.png)

Each gradients from all the replicas are aggregated between model using the bucketed [Ring AllReduce algorithm](https://towardsdatascience.com/visual-intuition-on-ring-allreduce-for-distributed-deep-learning-d1f34b4911da). DDP overlaps gradient computation with communication to synch them between models. The synchronization step does not need to wait for all gradient within one model to be computed, it can start communication along the ring while the backward pass is still running, this ensures the GPUs are always working.

On a computer with multiple GPUs, each GPU will run on process, which communicates between each other. A process group helps to discover those processes and manage the communication. One host is the master to coordinate the processes across GPUs and machines. 

![](./diagrams/ddp-3.drawio.png)

The backend is [Nvidia communication library](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#overview), `nccl`. On a single machine and one GPU it is not needed to use DDP, but to test the code and principle the `gloo` backend can be used.

```python
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
```

A model in the trainer is now a DDP using the gpu_id

```python
 self.model = DDP(model, device_ids=[gpu_id])
```

Each process has its own instance of the trainer class. Only one process will perform the checkpoint save:

```python
 if self.gpu_id == 0 and epoch % self.save
```

The `torch.multiprocessing` is responsible to take a function and spawns it to all processes in the distributed group. The rank, for each trainer, is assigned by multiprocessing.

```python
mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
```

### Multi-GPUs with [torchrun](https://pytorch.org/docs/stable/elastic/run.html)

`torchrun` provides a superset of the functionality as torch.distributed.launch to manage worker failures by restarting all workers from last checkpoint. The number of nodes may change overtime.

### Code sample

[multiple_gpu_ddp.py](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/ddp/multiple_gpu_ddp.py) demonstrates the basic DDP code to train a model using multiple GPUs machine, and the `from torch.utils.data.distributed import DistributedSampler` the `torch.nn.parallel.DistributedDataParallel` and `torch.distributed` modules. 

```python
def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

```

To run this example with 50 epochs saved every 10 epochs: `python multi_gpu_ddp.py 50 10 `

### Source of information

* [DDP Tutorial Series code repository.](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series) and [YouTube videos by Suraj Subramanian.](https://www.youtube.com/playlist?list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj)
* [See minGPT gir repository](https://github.com/subramen/minGPT-ddp)