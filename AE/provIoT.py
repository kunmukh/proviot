import os
import logging
from pathlib import Path
from AutoencoderFed import AutoencoderFed

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s | %(levelname)s\t| %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


def main():
    base = Path(f"../data/")
    benign = base / "example-feature-vector" / "benign-fv-arch.csv"
    anomaly = base / "example-feature-vector" / "anomaly-fv-arch.csv"

    log.info(f"DEBUG: PID: {os.getpid()}")
    log.info(f"DEBUG: DIRECTORY: {str('/'.join(os.listdir()))}")

    autoencoder = AutoencoderFed(benign, anomaly)
    autoencoder.run()


if __name__ == '__main__':
    main()
