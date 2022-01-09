import argparse
from time import time

from src.train import train_task_a, train_task_b1, train_task_b2


def parse_arguments():
    def parse_task(task: str):
        if task.lower() == 'a':
            return 'a'
        if task.lower() == 'b1':
            return 'b1'
        if task.lower() == 'b2':
            return 'b2'
        raise ValueError('Only available preparations: A, B1 and B2')

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--task',
        type=parse_task,
        help='task name',
        required=True
    )
    parser.add_argument(
        '--output',
        type=str,
        help='output name',
        required=True
    )
    parser.add_argument(
        '--pretrained-bert',
        type=str,
        help='pretrained bert path',
        required=True
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='dataset path',
        required=True
    )

    args = parser.parse_args()

    return args


def main(task):
    if task == 'a':
        train_task_a(args.pretrained_bert, args.dataset, args.output)
    elif task == 'b1':
        train_task_b1(args.pretrained_bert, args.dataset, model_name=args.output)
    elif task == 'b2':
        train_task_b2(args.pretrained_bert, args.dataset, model_name=args.output)


if __name__ == '__main__':
    init_time = time()
    args = parse_arguments()
    main(args.task)
    print('Total elapsed time:', time() - init_time)
