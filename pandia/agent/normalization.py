import numpy as np

NORMALIZATION_RANGE = [-1, 1]
RESOLUTION_LIST = [144, 240, 360, 480, 720, 960, 1080]


def nml(name, value: np.ndarray, value_range, normalized_range=NORMALIZATION_RANGE, log=None) -> np.ndarray:
    if log is None:
        log = value_range[1] - value_range[0] > 200
    if name == 'resolution':
        v = [RESOLUTION_LIST.index(min(RESOLUTION_LIST, key=lambda x:abs(x - v))) for v in value]
        value = np.array(v, dtype=np.float32)
        value_range = [0, len(RESOLUTION_LIST)]
    value = np.clip(value, value_range[0], value_range[1])
    if log:
        offset = 1 - value_range[0]
        res = np.log(value + offset) / np.log(value_range[1] + offset) * 2 - 1
    else:
        res = (value - value_range[0]) / (value_range[1] - value_range[0]) * \
            (normalized_range[1] - normalized_range[0]) + normalized_range[0]
    return res.astype(np.float32)


def dnml(name, value: np.ndarray, value_range, normalized_range=NORMALIZATION_RANGE, log=None) -> np.ndarray:
    if log is None:
        log = value_range[1] - value_range[0] > 200
    if name == 'resolution':
        value_range = [0, len(RESOLUTION_LIST)]
    if log:
        offset = 1 - value_range[0]
        res = np.exp((value + 1) / 2 * np.log(value_range[1] + offset)) - offset
    else:
        res = (value - normalized_range[0]) / (normalized_range[1] - normalized_range[0]) * \
            (value_range[1] - value_range[0]) + value_range[0]
    if name == 'resolution':
        indexes = np.clip(res, 0, len(RESOLUTION_LIST) - 1).astype(np.int32)
        res = [RESOLUTION_LIST[i] for i in indexes]
        res = np.array(res, dtype=np.int32)
    return res.astype(np.int32)


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
    for value_range in [[10, 200], [100, 100000]]:
        limit = 20
        for i in range(limit):
            v = int(i / limit * (value_range[1] - value_range[0]) + value_range[0])
            v = np.array([v])
            normalized = nml('test', v, value_range)
            assert NORMALIZATION_RANGE[0] <= normalized[0] <= NORMALIZATION_RANGE[1]
            vv = dnml('test', normalized, value_range)
            assert np.abs(vv[0] - v[0]) <= 1


if __name__ == '__main__':
    test_resolution()
    test_nml()