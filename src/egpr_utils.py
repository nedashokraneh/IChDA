from os import extsep
from math import log10, floor
from intervaltree import Interval, IntervalTree
import subprocess
import gzip
from contextlib import closing

EXT_GZ = "gz"

SUFFIX_GZ = extsep + EXT_GZ

def file_path(filename):
    if not os.path.exists(filename):
        os.makedirs(filename)
    return filename

def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)

def coords_file(windows_filename, resolution):
    with open(windows_filename) as f:
        assert not type(f) == str
        windows = []
        index = 0
        frame_offset = 0
        for line in f:
            line = line.split()
            chrom, start, end = line[0], int(line[1]), int(line[2])
            if not ((end-start) % resolution) == 0:
                if end-start < resolution:
                    continue
                else:
                    end -= (end-start) % resolution
            assert(((end-start) % resolution) == 0)
                # raise Exception("coords must be resolution-aligned!")

            windows.append([chrom, start, end, index, frame_offset])
            index += 1
            frame_offset += (end-start) // resolution

        with open(windows_filename, "w") as f:
            for window in windows:
                f.write("\t".join(map(str, window[:-2])))

        num_frames = frame_offset
        return windows, num_frames

def map_windows(windows):
    windows_map = {}
    for window_index, window in enumerate(windows):
        chrom = window[0]
        start = window[1]
        end = window[2]
        offset = window[4]
        if chrom not in windows_map:
            windows_map[chrom] = [[start, end, offset, window_index]]
        else:
            windows_map[chrom].append([start, end, offset, window_index])
    return windows_map

def create_interval_tree(windows):
    windows_tree = {}
    for window_index, window in enumerate(windows):
        chrom = window[0]
        start = window[1]
        end = window[2]
        offset = window[4]
        if chrom not in windows_tree:
            windows_tree[chrom] = IntervalTree()
        windows_tree[chrom][start:end] = [window_index, offset]
    return windows_tree

def get_frame_index(windows, resolution, chrom, pos):
    if chrom in windows:
        if len(windows[chrom][pos]) == 1:
            interval = list(windows[chrom][pos])[0]
            start = interval.begin
            window_index = interval.data[0]
            window_offset = interval.data[1]
            frame_offset = (pos - start) // resolution
            return window_index, window_offset + frame_offset
        else:
            return -1, -1
    else:
        return -1, -1

def run_command(cmd):
    print(" ".join(cmd))
    subprocess.Popen(cmd).wait()

#def gzip_open(*args, **kwargs):
#    return closing(_gzip_open(*args, **kwargs))

def is_gz_filename(filename):
    return str(filename).endswith(SUFFIX_GZ)

def maybe_gzip_open(filename, mode="rt", *args, **kwargs):
    if filename == "-":
        if mode.startswith("U"):
            raise NotImplementedError("U mode not implemented")
        elif mode.startswith("w") or mode.startswith("a"):
            return sys.stdout
        elif mode.startswith("r"):
            if "+" in mode:
                raise NotImplementedError("+ mode not implemented")
            else:
                return sys.stdin
        else:
            raise ValueError("mode string must begin with one of"
                             " 'r', 'w', or 'a'")

    if is_gz_filename(filename):
        return gzip.open(filename, mode, *args, **kwargs)

    return open(filename, mode, *args, **kwargs)

def ceildiv(dividend, divisor):
    "integer ceiling division"
    # int(bool) means 0 -> 0, 1+ -> 1
    return (dividend // divisor) + int(bool(dividend % divisor))
