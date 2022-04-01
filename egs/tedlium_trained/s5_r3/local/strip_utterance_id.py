#!/usr/bin/python3
"""
This script strips the first column of a text file
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", required=True)
    parser.add_argument("--output_file")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.text_file, encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line_split = line.split(" ")
        new_line = " ".join(line_split[1:])
        new_lines.append(new_line)

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    main()
