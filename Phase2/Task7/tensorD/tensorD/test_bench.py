# Created by ay27 at 16/11/8
import unittest
import logging
from logging.config import fileConfig
import sys

sys.path.append('..')

if __name__ == "__main__":
    fileConfig('tensorD/conf/logging_config.ini')
    # logging.getLogger().setLevel(logging.ERROR)
    suite = unittest.TestLoader().discover('.', pattern="*_test.py")
    unittest.TextTestRunner(verbosity=2).run(suite)