import argparse
from tagger1 import main as tagger1
from tagger2 import main as tagger2
from tagger3 import main_pretrained as tagger3_pretrained
from tagger3 import main_not_pretrained as tagger3_not_pretrained
from tagger4 import main as tagger4
from top_k import main as top_k


def build_parser():
    parser = argparse.ArgumentParser(
        prog="tagger_launcher.py",
        description="A driver code for tagger1, tagger2, tagger3",
    )

    parser.add_argument(
        "--part",
        "-p",
        type=int,
        default="1",
        nargs=1,
        choices=range(1, 6),
        help="Which part of the exercise to run, i.e 1 for tagger1",
        required=True,
    )

    parser.add_argument(
        "--task",
        "-t",
        type=str,
        nargs=1,
        default="ner",
        choices=["ner", "pos"],
        help="Which task to run, i.e ner or pos",
        required=False,
    )

    parser.add_argument(
        "--pretrained",
        "--pre",
        action="store_true",
        help="Whether to use pretrained embeddings or not, for tagger3 only",
        required=False,
        default=False,
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.pretrained and not args.task != 3:
        print("Pretrained argument is only for tagger3")
        exit(1)

    part = args.part[0]
    task = args.task[0]

    print(
        f"Running part {part} {'for task '+task if part!=2 else ''}{', using pretrained embeddings' if args.pretrained else ''}"
    )
    if part == 1:
        if task == "ner":
            tagger1("ner")
        else:
            tagger1("pos")

    elif part == 2:
        top_k()

    elif part == 3:
        if task == "ner":
            tagger2("ner")
        else:
            tagger2("pos")

    elif part == 4:
        if task == "ner":
            if pretrained:
                tagger3_pretrained("ner")
            else:
                tagger3_not_pretrained("ner")
        else:
            if args.pretrained:
                tagger3_pretrained("pos")
            else:
                tagger3_not_pretrained("pos")

    elif part == 5:
        if task == "ner":
            tagger2("ner")
        else:
            tagger2("pos")


if __name__ == "__main__":
    main()
