import os
import sys

PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from retrieval.retriever import benchmark_data_augmentation_call

def main():
    config_file_path = "config.ini"
    benchmark_data_augmentation_call(config_file_path)
if __name__ == "__main__":
    main()