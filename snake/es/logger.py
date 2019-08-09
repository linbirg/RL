# Python记录日志
import threading
import logging
import time
import os


class Logger(object):
    def __init__(self,
                 name='mainLogger',
                 path="./log/",
                 CLevel=logging.DEBUG,
                 FLevel=logging.DEBUG,
                 show_in_console=True):
        rq = time.strftime('%Y%m%d', time.localtime(time.time()))
        self.path = path
        self.filename = name + '_logger_' + threading.currentThread().getName(
        ) + '_' + rq + '.log'
        self.name = name
        # if Logger.logger is None:
        self.logger = logging.getLogger(self.filename)
        self.logger.setLevel(CLevel)

        try:
            self._mkdir_if_not_exist()  # 目录不存在则创建
            self.fh = logging.FileHandler(self.path + self.filename, 'a+')
            self.fh.setLevel(FLevel)
            self.formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(threadName)s - %(name)s - %(message)s'
            )
            self.fh.setFormatter(self.formatter)
            self.logger.addHandler(self.fh)
        except:
            pass

        if show_in_console:  # 再创建一个handler，用于输出到控制台
            self.ch = logging.StreamHandler()
            self.ch.setFormatter(self.formatter)
            self.ch.setLevel(FLevel)
            self.logger.addHandler(self.ch)

    def _mkdir_if_not_exist(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def close(self):
        self.logger.removeHandler(self.fh)
        self.logger.removeHandler(self.ch)


if __name__ == "__main__":
    logger = Logger()
    logger.debug("test debug")
    logger.info("test info")
    logger.warning("test Warning")
    logger.error("test error")

    logger2 = Logger("test_log", "./logs/")
    logger2.debug("test debug")
    logger2.info("test info")
    logger2.warning("test Warning")
    logger2.error("test error")

    logger3 = Logger("test_log", "./logs/")
    logger3.debug("test debug")
    logger3.info("test info")
    logger3.warning("test Warning")
    logger3.error("test error")