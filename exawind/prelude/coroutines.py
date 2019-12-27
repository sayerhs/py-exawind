# -*- coding: utf-8 -*-

"""\
Coroutine utilities
-------------------

Some code snippets inspired by http://www.dabeaz.com/coroutines/
"""

import re
import functools

def coroutine(func):
    """Prime a coroutine for send commands.

    Args:
        func (coroutine): A function that takes values via yield

    Return:
        function: Wrapped coroutine function
    """
    @functools.wraps(func)
    def _func(*args, **kwargs):
        fn = func(*args, **kwargs)
        next(fn)
        return fn
    return _func

@coroutine
def echo(**kwargs):
    """A simple output sink

    Useful as a consumer of data from other coroutines that just print to console
    """
    while True:
        output = (yield)
        print(output, **kwargs)

@coroutine
def grep(pattern, targets,
         send_close=True,
         matcher="search",
         flags=0):
    """Unix grep-like utility

    Feeds lines matching a target to consumer targets registered with this function

    Args:
        pattern (str): A regular expression as string (compiled internally)
        targets (list): A list of consumer coroutines that want to act on matching lines
        send_close (bool): If True, closes targets when grep exits
        matcher: ``search``, ``match``, ``findall`` methods of regular expression
        flags: Regexp flags used when compiling pattern
    """
    pat = re.compile(pattern, flags=flags)
    sfunc = getattr(pat, matcher)
    try:
        while True:
            line = (yield)
            mat = sfunc(line)
            if mat:
                for tgt in targets:
                    tgt.send(mat)
    except GeneratorExit:
        if send_close:
            for tgt in targets:
                tgt.close()
