import sys
import getopt
import os
from StencilChainSimulator.code.parser import Parser

def main(argv):
    options = set()
    inputfile = ""
    try:
        # argument with value needs colon, -h, -i <value>, --input=<value>
        opts, args = getopt.getopt(argv, "hi:gs", ["help", "input=", "graph", "sim"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "-help"):
            usage()
            sys.exit()
        elif opt in ("-i", "--input"):
            options.add("input")
            inputfile = arg
        elif opt in ("-g", "--graph"):
            options.add("graph")
        elif opt in ("-s", "--sim"):
            options.add("sim")
    execute(options, inputfile)


def execute(options, inputfile):
    # check if input file is available
    if "input" in options:

        ''' do the parsing process here'''

        # inform user about task
        print("Parsing input from " + inputfile)

        # do the parsing
        parser = Parser(inputfile)

        # check if caller requests graph
        if "graph" in options:
            print("Generate graph.")
        # check if caller requests sim
        if "sim" in options:
            print("Run simulation.")
    else:
        print("Nothing to parse. Exit.")


def usage():
    print("usage: main.py --help --input <inputfile> --graph --sim")


if __name__ == "__main__":
    main(sys.argv[1:])
