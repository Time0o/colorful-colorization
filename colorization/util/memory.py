import psutil
import pycuda.autoinit
import pycuda.driver


def _convert(size, unit):
    norm = {
        'B': 1,
        'KiB': 1024,
        'MiB': 1024**2,
        'GiB': 1024**3
    }

    if unit not in norm.keys():
        raise ValueError("invalid unit")

    return size / norm[unit]


def tensor_memory(t, unit='GiB'):
    return _convert(t.element_size() * t.nelement(), unit)


def RAM_in_use(unit='percent'):
    if unit == 'percent':
        return psutil.virtual_memory().percent
    else:
        return _convert(psutil.virtual_memory().used, unit)


def GPU_mem_in_use(unit='percent'):
    free, total = pycuda.driver.mem_get_info()

    if unit == 'percent':
        return 100 * (total - free) / total
    else:
        return _convert(total - free, unit)


def print_mem_summary(msg):
    fmt = ": {:.2f}% of RAM and {:.2f}% of GPU in use"
    msg += fmt.format(RAM_in_use(), GPU_mem_in_use())

    print(msg, flush=True)
