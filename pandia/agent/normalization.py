from typing import Union
import numpy as np
import pandia.agent.env_config as env_config


NORMALIZATION_RANGE = [-1, 1]
RESOLUTION_LIST = [144, 240, 360, 480, 720, 960, 1080, 1440, 2160]
LOG_FN = np.cbrt
LOG_FN_R = lambda x: np.power(x, 3)


def nml(name, value: Union[np.ndarray, float], value_range, normalized_range=NORMALIZATION_RANGE, log=None) -> np.ndarray:
    unwrap = False
    if type(value) != np.ndarray:
        value = np.array([value, ], dtype=np.float32)
        unwrap = True
    if log is None:
        log = value_range[1] - value_range[0] > 200
    if name == 'resolution':
        v = [RESOLUTION_LIST.index(min(RESOLUTION_LIST, key=lambda x:abs(x - v))) for v in value]
        value = np.array(v, dtype=np.float32)
        value_range = [0, len(RESOLUTION_LIST)]
        log = False
    value = np.clip(value, value_range[0], value_range[1])
    if log:
        offset = value - value_range[0]
        res = LOG_FN(offset) / LOG_FN(value_range[1] - value_range[0]) * \
            (normalized_range[1] - normalized_range[0]) + normalized_range[0]
    else:
        res = (value - value_range[0]) / (value_range[1] - value_range[0]) * \
            (normalized_range[1] - normalized_range[0]) + normalized_range[0]
    if unwrap:
        return res[0]
    else:
        return res.astype(np.float32)


def dnml(name, value: Union[np.ndarray, float], value_range, normalized_range=NORMALIZATION_RANGE, log=None) -> np.ndarray:
    unwrap = False
    if type(value) != np.ndarray:
        value = np.array([value, ], dtype=np.float32)
        unwrap = True
    if log is None:
        log = value_range[1] - value_range[0] > 200
    if name == 'resolution':
        value_range = [0, len(RESOLUTION_LIST)]
    if log:
        offset = (value - normalized_range[0]) / (normalized_range[1] - normalized_range[0]) * \
            LOG_FN(value_range[1] - value_range[0]) 
        res = LOG_FN_R(offset) + value_range[0]
    else:
        res = (value - normalized_range[0]) / (normalized_range[1] - normalized_range[0]) * \
            (value_range[1] - value_range[0]) + value_range[0]
    if name == 'resolution':
        indexes = np.clip(res, 0, len(RESOLUTION_LIST) - 1).astype(np.int32)
        res = [RESOLUTION_LIST[i] for i in indexes]
        res = np.array(res, dtype=np.int32)
    if unwrap:
        return res[0]
    else:
        return res.astype(np.float32)


def test_resolution():
    value_range = [0, 1]
    for i in RESOLUTION_LIST:
        i = np.array([i])
        v = nml('resolution', i, value_range)
    limit = 2000
    counts = [0] * len(RESOLUTION_LIST)
    for i in range(limit):
        v = (i / limit - .5) * 2
        v = np.array([v])
        res = dnml('resolution', v, value_range)
        j = RESOLUTION_LIST.index(res[0])
        counts[j] += 1
    counts = np.array(counts)
    diff = counts - np.mean(counts)
    assert np.all(np.abs(diff) <= 1)


def test_nml():
    for log in [True, False]:
        for value_range in [[10, 200], [100, 100000]]:
            limit = 20
            for i in range(limit):
                v = int(i / limit * (value_range[1] - value_range[0]) + value_range[0])
                v = np.array([v])
                normalized = nml('test', v, value_range, log=log)
                assert NORMALIZATION_RANGE[0] <= normalized[0] <= NORMALIZATION_RANGE[1]
                vv = dnml('test', normalized, value_range, log=log)
                assert np.abs(vv[0] - v[0]) <= 1


def test():
    from pandia.constants import M, K
    bw_range = env_config.NORMALIZATION_RANGE["bandwidth"]
    print(f'bw range: {[bw_range[0] / M, bw_range[1] / M]} Mbps')
    for bw in [200 * K, 500 * K, 
               M, 2 * M, 5 * M, 
               10 * M, 20 * M, 50 * M, 
               100 * M, 200 * M, 500 * M]:
        print(f'{bw / M:.02f} mbps', nml('bandwidth', bw, 
                                         env_config.NORMALIZATION_RANGE['bandwidth'], log=True))


if __name__ == '__main__':
    test_resolution()
    test_nml()
    test()