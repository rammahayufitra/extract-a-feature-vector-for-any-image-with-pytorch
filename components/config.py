import configparser
import ast
import sys


config = configparser.ConfigParser()
config.read("/.opt/public/config.cfg")

FILENAME_OK = config['DATASET']['FILENAME_OK']
FILENAME_ARGUABLY_GOOD = config['DATASET']['FILENAME_ARGUABLY_GOOD']
PATH_DATASET = config['DATASET']['PATH_DATASET']



