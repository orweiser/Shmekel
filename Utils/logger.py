import logging
import os

FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
STREAM_LEVEL = 'debug'
LOG_LEVEL = None
DECORATOR_LEVEL = 'debug'


class ShmekelLogger:
    def __init__(self, name='default_logger', stream_level=STREAM_LEVEL, log_level=LOG_LEVEL,
                 default_decorator_log_level=DECORATOR_LEVEL):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(FORMAT)

        self.stream_handler = None

        if log_level:
            file_handler = logging.FileHandler(os.path.join('logs', '%s.log' % name))
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if stream_level:
            self.stream_handler = logging.StreamHandler()
            self.stream_handler.setLevel(getattr(logging, stream_level.upper()))
            self.stream_handler.setFormatter(formatter)
            logger.addHandler(self.stream_handler)

        self.logger = logger
        self.default_decorator_log_level = default_decorator_log_level

    def _decorator(self, f, log_level=None, elaborate=False, arg_num_chars=100):
        log_level = log_level or self.default_decorator_log_level
        logging_func = getattr(self.logger, log_level)

        loc = '%s [%s.%s]' % (f.__name__, f.__module__, f.__name__)

        def decorator(*args, **kwargs):
            msg = 'Entered: %s with %d args and %d kwargs' % (loc, len(args), len(kwargs))

            if elaborate:
                msg += '\n\targs:'
                for i, arg in args:
                    msg += '\n\t\t%d: %s' % (i, str(arg)[:arg_num_chars])

                msg += '\n\tkwargs:'
                for key, val in kwargs.items():
                    msg += '\n\t\t%s: %s' % (str(key), str(val)[:arg_num_chars])

            logging_func(msg)
            try:
                outs = f(*args, **kwargs)
            except Exception as e:
                self.logger.exception('Failed to execute %s' % loc)
                raise e
            else:
                logging_func('Exited %s successfully' % loc)
                return outs

        return decorator

    def __getattr__(self, item):
        return getattr(self.logger, item)

    def __call__(self, f): return self._decorator(f)

    def debug_dec(self, f): return self._decorator(f, log_level='debug')

    def info_dec(self, f): return self._decorator(f, log_level='info')

    def warning_dec(self, f): return self._decorator(f, log_level='warning')

    def error_dec(self, f): return self._decorator(f, log_level='error')

    def critical_dec(self, f): return self._decorator(f, log_level='critical')

    def elaborate_dec(self, f): return self._decorator(f, log_level='critical', elaborate=True)

    def set_stream_level(self, stream_level):
        self.stream_handler.setLevel(getattr(logging, stream_level.upper()))


logger = ShmekelLogger()

