import argparse
from time import time
from pathlib import Path

from src.pipeline import pipeline_task_a, pipeline_task_b1, pipeline_task_b2, pipeline


def parse_arguments():
    def parse_task(task: str):
        if task.lower() == 'a':
            return 'a'
        if task.lower() == 'b1':
            return 'b1'
        if task.lower() == 'b2':
            return 'b2'
        if task.lower() == 'full_b1':
            return 'full_b1'
        if task.lower() == 'full_b2':
            return 'full_b2'

        raise ValueError('Only available tasks: A, B1 (one classifier), B2 (two classifiers), full_b1 (one classifier) and full_b2 (two classifiers)')

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
        help='output path',
        required=True
    )
    parser.add_argument(
        '--pretrained-bert',
        type=str,
        help='pretrained bert path',
        required=True
    )
    parser.add_argument(
        '--pretrained-classifier',
        type=str,
        help='pretrained classifier path',
        required=True
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='dataset path',
        required=True
    )

    args = parser.parse_args()

    # arguments checker
    if args.task in ['b1', 'b2']:
        if not args.pretrained_classifier:
            raise ValueError('When calling task b1 or b2 a pretrained classifier is required')

    return args


def main(args):
    if args.task == 'a':
        output = pipeline_task_a(args.pretrained_bert, args.dataset)
        output.dump(Path(args.output))
    elif args.task == 'b1':
        output = pipeline_task_b1(args.pretrained_classifier, args.pretrained_bert, args.dataset)
        output.dump(Path(args.output))
    elif args.task == 'b2':
        output = pipeline_task_b2(args.pretrained_classifier + '_bin', args.pretained_classifier + '_rel', args.pretrained_bert, args.dataset)
        output.dump(Path(args.output))
    elif args.task == 'full_b1':
        output = pipeline(args.pretrained_bert, [args.pretrained_classifier], args.pretrained_bert, args.dataset)
        output.dump(Path(args.output))
    elif args.task == 'full_b2':
        output = pipeline(args.pretrained_bert, [args.pretrained_classifier + '_bin', args.pretained_classifier + '_rel'], args.pretrained_bert, args.dataset)
        output.dump(Path(args.output))


if __name__ == '__main__':
    init_time = time()
    args = parse_arguments()
    main(args)
    print('Total elapsed time:', time() - init_time)
