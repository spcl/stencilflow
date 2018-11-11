import sys
import getopt
from StencilChainSimulator.code.parser import Parser


def main(argv):
    options = set()
    input_file = ""
    dimension = ""
    try:
        # argument with value needs colon, -h, -i <value>, --input=<value>
        opts, args = getopt.getopt(argv, "hi:gsd:", ["help", "input=", "graph", "sim", "dimension"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "-help"):
            usage()
            sys.exit()
        elif opt in ("-i", "--input"):
            options.add("input")
            input_file = arg
        elif opt in ("-g", "--graph"):
            options.add("graph")
        elif opt in ("-s", "--sim"):
            options.add("sim")
        elif opt in ("-d", "--dimension"):
            options.add("dimension")
            dimension = arg
    execute(options, input_file, dimension)


def execute(options, input_file, dimension):
    # check if input file is available
    if "input" in options:

        ''' do the parsing process here'''

        # inform user about task
        print("Parsing input from " + input_file)

        # do the parsing
        parser = Parser(input_file)

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
