import sys
import getopt


def main(argv):
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
            inputfile = arg
            print("Input file is", inputfile)
        elif opt in ("-g", "--graph"):
            print("Generate graph.")
        elif opt in ("-s", "--sim"):
            print("Run simulation.")
    return


def usage():
    print("usage: main.py -h -i <inputfile> -graph -sim")


if __name__ == "__main__":
    main(sys.argv[1:])
