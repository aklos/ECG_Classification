import argparse
import numpy as np
from predict import predict_labels


def main(edf_path: str):
    data = edf_to_list(edf_path, 0, 120)
    predict_labels([data], 500, ['ecg'])


def edf_to_list(edf_file_path, offset, limit):
    record = wf.edf2mit(edf_file_path, verbose=True)

    signal = record.p_signal
    data = signal.transpose()

    if offset or limit:
        offset_samples = int(record.fs * offset)
        limit_samples = offset_samples + int(record.fs * limit)
        return data[0][offset_samples:limit_samples]

    return np.array(data[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        help='path to the EDF dataset', required=True)
    args = parser.parse_args()
    main(args.input)
