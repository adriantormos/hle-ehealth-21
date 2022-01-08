import argparse
from time import time
from pathlib import Path

from src.pipeline import pipeline_task_a


def parse_arguments():
    def parse_task(task: str):
        if task.lower() == 'a':
            return 'a'
        if task.lower() == 'b':
            return 'b'
        if task.lower() == 'full':
            return 'full'
        raise ValueError('Only available tasks: A, B and full')

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--task',
        type=parse_task,
        help='task name',
        required=True
    )
    parser.add_argument(
        '--tokenizer-pretrained-model',
        type=str,
        help='tokenizer pretrained model path',
        required=False
    )

    args = parser.parse_args()

    # arguments checker
    if args.task == 'a':
        if not args.tokenizer_pretrained_model:
            raise ValueError('When calling task a a pretrained tokenizer model is required')
    elif args.task == 'b':
        pass
    elif args.task == 'full':
        pass

    return args


def main(task):
    if task == 'a':
        output = pipeline_task_a(args.tokenizer_pretrained_model)
        output.dump(Path('output/output_task_a.txt'))
    elif task == 'b':
        pass
    elif task == 'full':
        pass


if __name__ == '__main__':
    init_time = time()
    args = parse_arguments()
    main(args.task)
    print('Total elapsed time:', time() - init_time)
