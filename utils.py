import time


def timing(f):
    def g(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        end = time.time()
        print("{}s costs".format(end - start))
        return res

    return g
