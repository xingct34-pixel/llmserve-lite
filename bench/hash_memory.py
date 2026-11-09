"""Measure benchmark-client hashing memory in a fresh Linux process."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys

from run import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['whole-file', 'stream'], required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('ru_maxrss units in this experiment require Linux')
    if args.mode == 'whole-file':
        digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    else:
        digest = sha256_file(args.model)
    print(json.dumps({'mode': args.mode, 'model_bytes': args.model.stat().st_size,
                      'sha256': digest, 'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      'scope': 'benchmark client only, not inference server'}))


if __name__ == '__main__':
    main()
