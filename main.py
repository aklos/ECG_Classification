import argparse
import wfdb as wf
import numpy as np
import pickle
import tensorflow as tf
from predict import predict_labels


def main(edf_path: str, offset: int = 0, limit: int = 0):
    data, sample_rate = edf_to_list(
        edf_path, offset, None if limit == 0 else limit)

    # split signal into 6 second segments
    chunk_size = sample_rate * 6
    list_chunked = np.array([data[i:i + chunk_size]
                             for i in range(0, len(data), chunk_size)])

    device = 'cpu'

    # if tf.test.gpu_device_name() == '/device:GPU:0':
    #     device = 'gpu'

    results = predict_labels(list_chunked, sample_rate, [
                             i for i, x in enumerate(list_chunked)], device=device)

    results = [x for x in results if x[1] == 'A']

    with open(edf_path[:-4] + '_rhythms', 'wb') as file:
        pickle.dump(results, file)

    print(results)


def edf_to_list(edf_file_path, offset, limit):
    record = wf.edf2mit(edf_file_path, verbose=True)

    signal = record.p_signal
    data = signal.transpose()

    if offset or limit:
        offset_samples = int(record.fs * offset)
        limit_samples = offset_samples + int(record.fs * limit)
        return np.array(data[0][offset_samples:limit_samples]), record.fs

    return np.array(data[0]), record.fs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        help='path to the EDF dataset', required=True)
    parser.add_argument('--offset', type=int,
                        help='offset in seconds')
    parser.add_argument('--limit', type=int,
                        help='output limit in seconds')
    args = parser.parse_args()
    main(args.input, args.offset, args.limit)
