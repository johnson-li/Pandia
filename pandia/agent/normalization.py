import numpy as np

NORMALIZATION_RANGE = [-1, 1]
RESOLUTION_LIST = [144, 240, 360, 480, 720, 960, 1080]


def nml(name, value: np.ndarray, value_range, normalized_range=NORMALIZATION_RANGE) -> np.ndarray:
    if name == 'resolution':
        v = [RESOLUTION_LIST.index(min(RESOLUTION_LIST, key=lambda x:abs(x - v))) for v in value]
        value = np.array(v, dtype=np.float32)
        value_range = [0, len(RESOLUTION_LIST)]
    value = np.clip(value, value_range[0], value_range[1])
    res = (value - value_range[0]) / (value_range[1] - value_range[0]) * \
        (normalized_range[1] - normalized_range[0]) + normalized_range[0]
    return res.astype(np.float32)


def dnml(name, value: np.ndarray, value_range, normalized_range=NORMALIZATION_RANGE):
    if name == 'resolution':
        value_range = [0, len(RESOLUTION_LIST)]
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
    limit = 200000
    counts = [0] * len(RESOLUTION_LIST)
    for i in range(limit):
        v = (i / limit - .5) * 2
        v = np.array([v])
        res = dnml('resolution', v, value_range)
        j = RESOLUTION_LIST.index(res[0])
        counts[j] += 1
    print('dnml distribution', counts)


if __name__ == '__main__':
    test_resolution()