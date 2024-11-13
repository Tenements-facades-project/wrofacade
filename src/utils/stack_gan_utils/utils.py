# source repo : https://github.com/rightlit/StackGAN-v2-rev
# original project repo : https://github.com/hanzhanggit/StackGAN-v2


import errno
import os


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
