#!/usr/bin/env python

import random, time, os
from subprocess import Popen, PIPE


def idle_gpu(n=1, min_memory=4096, time_step=60 * 1, time_out=3600 * 16):
    elaspsed_time = 0
    p = Popen(['/bin/bash', '-c', "nvidia-smi | grep GeForce | wc -l"], stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    n_GPUs = int(out)
    random.seed(time.time() % os.getpid())
    rand_priority = [random.random() for x in range(n_GPUs)]
    rand_priority[0] += 1
    while elaspsed_time < time_out:
        cmd = "nvidia-smi | grep Default | awk '{print NR-1,$9,$11,$13,$3}' | sed 's/MiB//g;s/%//g;s/C//g'"
        p = Popen(['/bin/bash', '-c', cmd], stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        query_result, err = out.decode('utf8'), err.decode('utf8')
        rc = p.returncode
        query_result = query_result.strip().split('\n')

        gpu_list = list()
        for i, gpu_info in enumerate(query_result):
            gpu, memory_usage, memory_total, gpu_usage, temp = gpu_info.split(' ')
            memory_usage, memory_total, gpu_usage, temp = int(memory_usage), int(memory_total), int(gpu_usage), int(temp)
            memory = memory_total - memory_usage
            gpu_list.append((gpu, int(round(memory_usage / 1000)), int(round(gpu_usage/10)), int(round(temp / 10)), rand_priority[i], memory)) # reverse use
        ans = sorted(gpu_list, key=lambda x:(x[1], x[2], x[3], x[4]))
        if ans[0][-1] < min_memory:
            print("Waiting for available GPU... (%s)" % (time.asctime()))
            # time.sleep(60 * 10)
            time.sleep(time_step)
            elaspsed_time += time_step
            if elaspsed_time > time_out:
                raise MemoryError("Error: No available GPU with memory > %d MiB" % (min_memory))
        else:
            break

    #return ','.join(ans[0][0])
    return ','.join([ans[i][0] for i in range(n)])

def set_GPU(n_gpu=1, min_memory=5120, quiet=False):
    gpu_ids = idle_gpu(n=n_gpu, min_memory=min_memory)
    if not quiet:
        print("- Using GPU: {}".format(gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids


if __name__ == "__main__":
    ans = idle_gpu(n=1, min_memory=4000)
    print(ans)
