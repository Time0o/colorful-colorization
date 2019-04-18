import torch.cuda as cuda


def cuda_time(fname, f, *args, **kwargs):
    start = cuda.Event(enable_timing=True)
    end = cuda.Event(enable_timing=True)

    cuda.synchronize()

    start.record()
    ret = f(*args, **kwargs)
    end.record()

    cuda.synchronize()

    t = start.elapsed_time(end) / 1000

    fmt = "'{}' ran in {:.2e} seconds"
    print(fmt.format(fname, t, flush=True))

    return ret
