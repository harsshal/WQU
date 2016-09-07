__author__ = 'harsshal'

def set_file_input():
    import sys
    if len(sys.argv) >= 2:
        import fileinput
        sys.stdin = fileinput.input(sys.argv[1])

def array_input(msg,type):
    str = input(msg)
    return list(map(type,str.split()))
