import sys
import os
sys.path.append('../utilities')
import data_utils
import json
import argparse
from dataset_class import dataset

def parse_arg():
    p = argparse.ArgumentParser()
    p.add_argument('-c', help="The config file including different datasets paths.")
    return p.parse_args()

def run(args):
    f = open(args.c, "r")
    config = json.load(f)
    if not os.path.exists(config["temp_dir"]):
        os.mkdir(config["temp_dir"])
    if not os.path.exists(config["processed_dir"]):
        os.mkdir(config["processed_dir"])
    print('Starting binning input signals...')
    bin_signals(config["signals_dir"], config["signals_names"], config["temp_dir"], config["resolution"])
    print('Finished binning input signals...')
    dataset_obj = dataset(config["cell_type"], config["chr_size_path"],
                config["valid_chroms"], config["valid_bins"], config["resolution"],
                config["hic_file"], config["juicer_path"], config["temp_dir"],
                 config["signals_names"], config["temp_dir"], config["processed_dir"])
    print('Starting making input signals matrices...')
    dataset_obj.generate_signals_and_bin_files()
    print('Finished making input signals matrices...')
    dataset_obj.generate_signals_and_bin_files()
    print("Starting generating Hi-C interaction graph file...")
    dataset_obj.generate_significant_interactions_file('oe', 2000)
    print("Finished generating Hi-C interaction graph file...")


def bin_signals(signals_dir, signals_names, processed_signals_dir, resolution):

    for signal_name in signals_names:
        signal_path = os.path.join(signals_dir, '{}.bedgraph'.format(signal_name))
        processed_signal_path = os.path.join(processed_signals_dir, '{}.bedgraph'.format(signal_name))
        if os.path.exists(processed_signal_path):
            print('{} signal is binned...'.format(signal_name))
        elif os.path.exists(signal_path):
            print('binning {} signal...'.format(signal_name))
            data_utils.bin_bedgraph_file(signal_path, resolution, processed_signal_path, False)
        else:
            print('{} signal does not exist...'.format(signal_name))


if __name__ == '__main__':
    args = parse_arg()
    run(args)
