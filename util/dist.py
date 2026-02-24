# util/dist.py — single-process stub
import torch

def init_distributed_mode(args):
    args.distributed = False
    args.rank = 0
    args.world_size = 1
    args.gpu = 0

def is_dist_avail_and_initialized():
    return False

def get_rank():
    return 0

def get_world_size():
    # 싱글 프로세스
    return 1

def is_main_process():
    return True

def barrier():
    pass

def save_on_master(state, path):
    torch.save(state, path)

def all_gather(obj):
    return [obj]

def reduce_dict(input_dict, average=False):
    return input_dict

def all_reduce(tensor, op="sum"):
    """
    싱글 프로세스용 no-op all_reduce.
    학습 코드에서 dist.all_reduce(t); t /= dist.get_world_size() 형태를 그대로 쓰기 위함.
    """
    return tensor