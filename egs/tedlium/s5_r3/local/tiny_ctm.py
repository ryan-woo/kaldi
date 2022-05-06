# Make a tiny ctm file for the reference
# I found that the time stamp in the existing reference file for
# the set of utterances selected for the demo is always less than 260.

import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ctm", required=True)
    parser.add_argument("--output-ctm", required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.input_ctm) as f:
        lines = f.readlines()

    outlines = []
    for line in lines:
        if line.startswith("Aimee"):
            split = line.split()
            start_time = split[2]
            if float(start_time) < 260:
                outlines.append(line)

    with open(args.output_ctm, "w") as f:
        f.writelines(outlines)
    

if __name__ == "__main__":
    main()