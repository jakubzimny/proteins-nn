from typing import Tuple, List

class DataNormalizer:

    def __init__(self, dataset):
        self._dataset = dataset
        self._mins, self._maxes = self._get_min_max()

    def _get_min_max(self) -> Tuple:
        mins = [100000 for _ in range(0, len(self._dataset))]
        maxes = [-100000 for _ in range(0, len(self._dataset))]
        for row in self._dataset:
            for index, el in enumerate(row):
                if el < mins[index]:
                    mins[index] = el
                if el > maxes[index]:
                    maxes[index] = el
        return (mins, maxes)

    def _scale(self, element: float, index: int) -> float:
        return (element - self._mins[index])/ (self._maxes[index] - self._mins[index])
        
    def _inverse_scale(self, element: float, index: int) -> float:
        return element * (self._maxes[index] - self._mins[index]) + self._mins[index]

    def get_normalized_dateset(self) -> List:
        normalized_set = []
        for row in self._dataset:
            normalized_row = []
            for index, el in enumerate(row):
                normalized_row.append(self._scale(el, index)) 
            normalized_set.append(normalized_row)
        return normalized_set
    
    def inverse_scale_row(self, X: List, Y: List) -> List:
        row = X + Y
        inv_scaled_row = []
        for index, el in enumerate(row):
            inv_scaled_row.append(self._inverse_scale(el, index))
        return inv_scaled_row[:len(X)], inv_scaled_row[len(X):]
