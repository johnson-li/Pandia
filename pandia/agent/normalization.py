import numpy as np

NORMALIZATION_RANGE = [-1, 1]
RESOLUTION_LIST = [144, 240, 360, 480, 720, 960, 1080]


def nml(name, value: np.ndarray, value_range, normalized_range=NORMALIZATION_RANGE) -> np.ndarray:
    if name == 'resolution':
        v = [RESOLUTION_LIST.index(min(RESOLUTION_LIST, key=lambda x:abs(x - v))) \
            / len(RESOLUTION_LIST) for v in value]
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
        indexes = (res * len(RESOLUTION_LIST)).astype(np.int32)
        indexes = np.clip(indexes, 0, len(RESOLUTION_LIST) - 1)
        res = [RESOLUTION_LIST[i] for i in indexes]
        res = np.array(res, dtype=np.int32)
    return res.astype(np.int32)
