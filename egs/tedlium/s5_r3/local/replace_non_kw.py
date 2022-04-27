#!/usr/bin/python3
"""
rsw2148
This script replaces all non-keywords in the text with the <unk> token.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", required=True)
    parser.add_argument("--text_file", required=True)
    parser.add_argument("--output_file")
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--stm", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.text_file, encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:

        if line.startswith(";;"):
            new_lines.append(line)
            continue

        if "ignore_time_segment_in_scoring" in line:
            new_lines.append(line)
            continue

        words = line.split(" ")
        if args.text:
            line_prefix = words[0]
            words = words[1:]
        elif args.stm:
            line_prefix = " ".join(words[:6])
            words = words[6:]
            assert "unknown" not in line_prefix
        else:
            raise ValueError("Need to set one of --text or --stm")

        # Replace non-keywords in the line
        new_words = [line_prefix]
        for word in words:
            if word != args.keyword:
                if word.endswith("\n"):
                    new_words.append("<unk>\n")
                else:
                    new_words.append("<unk>")
            else:
                new_words.append(word)

        new_line = " ".join(new_words)
        new_lines.append(new_line)


    with open(args.output_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)




if __name__ == "__main__":
    main()
