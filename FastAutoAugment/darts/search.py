from ..common.config import Config

def darts_search(conf:Config)->None:
    if not conf['bilevel']:
        print('WARNING: bilevel arg is NOT true. This is useful only for abalation study for bilevel optimization!')