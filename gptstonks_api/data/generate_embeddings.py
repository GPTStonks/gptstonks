import argparse

import pandas as pd
import torch
from openbb_chat.classifiers.stransformer import STransformerZeroshotClassifier
from rich.progress import track


def parse_args(args=None):
    parser = argparse.ArgumentParser("Create data tensor representing OpenBB documents.")
    parser.add_argument(
        "-kc",
        "--keys-csv",
        type=str,
        required=True,
        help="Input .csv with pairs of descriptions and Python definitions. Columns should be 'Descriptions' and 'Definitions'.",
    )
    parser.add_argument(
        "-o",
        "--output-file-pt",
        type=str,
        required=True,
        help="Output file to store PyTorch embeddings of the descriptions.",
    )
    parser.add_argument(
        "-cm",
        "--classifier-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF MNLI model to use. Default: sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "-s", "--separator", type=str, default="@", help="Separator in .csv. Default: @."
    )
    return parser.parse_args(args=args)


def main(args):
    df = pd.read_csv(args.keys_csv, sep=args.separator)
    df = df.dropna()
    descriptions = df["Descriptions"].tolist()
    definitions = df["Definitions"].tolist()

    keys = []
    for idx, descr in track(
        enumerate(descriptions), total=len(descriptions), description="Processing..."
    ):
        topics = definitions[idx][: definitions[idx].index("(")].split(".")[1:]
        if descr.find("[") != -1:
            descr = descr[: descr.find("[")].strip()
        if descr.strip()[-1] != ".":
            search_str = f"{descr.strip()}. Topics: {', '.join(topics)}."
        else:
            search_str = f"{descr.strip()} Topics: {', '.join(topics)}."
        keys.append(search_str)

    stransformer = STransformerZeroshotClassifier(keys, args.classifier_model)
    torch.save(stransformer.descr_embed, args.output_file_pt)


if __name__ == "__main__":
    main(parse_args())
