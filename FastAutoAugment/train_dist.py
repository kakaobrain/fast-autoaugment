import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

import time
import os
import threading
import six
from six.moves import queue

from FastAutoAugment import safe_shell_exec


def _exec_command(command):
    host_output = six.StringIO()
    try:
        exit_code = safe_shell_exec.execute(command,
                                            stdout=host_output,
                                            stderr=host_output)
        if exit_code != 0:
            print('Launching task function was not successful:\n{host_output}'.format(host_output=host_output.getvalue()))
            os._exit(exit_code)
    finally:
        host_output.close()
    return exit_code


def execute_function_multithreaded(fn,
                                   args_list,
                                   block_until_all_done=True,
                                   max_concurrent_executions=1000):
    """
    Executes fn in multiple threads each with one set of the args in the
    args_list.
    :param fn: function to be executed
    :type fn:
    :param args_list:
    :type args_list: list(list)
    :param block_until_all_done: if is True, function will block until all the
    threads are done and will return the results of each thread's execution.
    :type block_until_all_done: bool
    :param max_concurrent_executions:
    :type max_concurrent_executions: int
    :return:
    If block_until_all_done is False, returns None. If block_until_all_done is
    True, function returns the dict of results.
        {
            index: execution result of fn with args_list[index]
        }
    :rtype: dict
    """
    result_queue = queue.Queue()
    worker_queue = queue.Queue()

    for i, arg in enumerate(args_list):
        arg.append(i)
        worker_queue.put(arg)

    def fn_execute():
        while True:
            try:
                arg = worker_queue.get(block=False)
            except queue.Empty:
                return
            exec_index = arg[-1]
            res = fn(*arg[:-1])
            result_queue.put((exec_index, res))

    threads = []
    number_of_threads = min(max_concurrent_executions, len(args_list))

    for _ in range(number_of_threads):
        thread = threading.Thread(target=fn_execute)
        if not block_until_all_done:
            thread.daemon = True
        thread.start()
        threads.append(thread)

    # Returns the results only if block_until_all_done is set.
    results = None
    if block_until_all_done:
        # Because join() cannot be interrupted by signal, a single join()
        # needs to be separated into join()s with timeout in a while loop.
        have_alive_child = True
        while have_alive_child:
            have_alive_child = False
            for t in threads:
                t.join(0.1)
                if t.is_alive():
                    have_alive_child = True

        results = {}
        while not result_queue.empty():
            item = result_queue.get()
            results[item[0]] = item[1]

        if len(results) != len(args_list):
            raise RuntimeError(
                'Some threads for func {func} did not complete '
                'successfully.'.format(func=fn.__name__))
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str)
    parser.add_argument('--num-gpus', type=int, default=4)
    parser.add_argument('--master', type=str, default='task1')
    parser.add_argument('--port', type=int, default=1958)
    parser.add_argument('-c', '--conf', type=str)
    parser.add_argument('--args', type=str, default='')

    args = parser.parse_args()

    try:
        hosts = ['task%d' % (x + 1) for x in range(int(args.host))]
    except:
        hosts = args.host.split(',')

    cwd = os.getcwd()
    command_list = []
    for node_rank, host in enumerate(hosts):
        ssh_cmd = f'ssh -t -t -o StrictHostKeyChecking=no {host} -p 22 ' \
                  f'\'bash -O huponexit -c "cd {cwd} && ' \
                  f'python -m torch.distributed.launch --nproc_per_node={args.num_gpus} --nnodes={len(hosts)} ' \
                  f'--master_addr={args.master} --master_port={args.port} --node_rank={node_rank} ' \
                  f'FastAutoAugment/train.py -c {args.conf} {args.args}"' \
                  '\''
        print(ssh_cmd)

        command_list.append([ssh_cmd])

    execute_function_multithreaded(_exec_command,
                                   command_list[1:],
                                   block_until_all_done=False)

    print(command_list[0])

    while True:
        time.sleep(1)

    # thread = threading.Thread(target=safe_shell_exec.execute, args=(command_list[0][0],))
    # thread.start()
    # thread.join()

    # while True:
    #     time.sleep(1)
