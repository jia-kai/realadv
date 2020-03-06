import sys
import importlib

def main():
    """use a main function to avoid the module has name __main__ and gets
    imported again with the fully qualified name"""
    if len(sys.argv) == 1:
        print(f'usage: {sys.argv[0]} <submodule> <submodule args ...>')
    name = sys.argv[1]
    del sys.argv[1]
    return importlib.import_module(f'realadv.{name}').main()

if __name__ == '__main__':
    main()
