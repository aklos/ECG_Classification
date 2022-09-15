import argparse
import wfdb as wf
import numpy as np
from predict import predict_labels


def main(edf_path: str, offset: int = 0, limit: int = 0):
    data = edf_to_list(edf_path, offset, None if limit == 0 else limit)
    # split signal into 10 second segments
    chunk_size = 125 * 10
    list_chunked = np.array([data[i:i + chunk_size]
                             for i in range(0, len(data), chunk_size)])

    for i, x in enumerate(list_chunked):
        results = predict_labels([x], 125, [i])
        print(results)


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
    parser.add_argument('--offset', type=int,
                        help='offset in seconds')
    parser.add_argument('--limit', type=int,
                        help='output limit in seconds')
    args = parser.parse_args()
    main(args.input, args.offset, args.limit)
