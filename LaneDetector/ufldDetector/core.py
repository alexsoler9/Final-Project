import cv2
import abc
import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class LaneInfo:
    _lanes_points   : np.ndarray # Detected Road Points
    _lanes_status   : np.ndarray # Detected Road Status
    _area_points    : np.ndarray 
    _area_status    : bool
    
    @property
    def lanes_points(self):
        return self._lanes_points
    
    @lanes_points.setter
    def lanes_points(self, arr : np.ndarray) -> None:
        if isinstance(arr, np.ndarray):
            self._lanes_points = arr
        else:
            raise Exception("'lanes_points' must be a np.ndarray[List[Tuple[x, y], ...], ...]")

    @property    
    def lanes_status(self):
        return self._lanes_status
    
    @lanes_status.setter
    def lanes_status(self, value : list) -> None:
        for v in value:
            if type(v) != bool:
                raise Exception("'lanes_status' must be of type bool List[bool, ...]")
        self._lanes_status = value
    
    @property
    def area_status(self):
        return self._area_status

    @area_status.setter
    def area_status(self, value: bool) -> None:
        raise Exception("Use the '_update_lanes_status' API")

    @property
    def area_points(self):
        return self.area_points

    @area_points.setter
    def area_points(self, value: bool):
        raise Exception("Use the '_update_lanes_area' API")

class LaneDetectBase(abc.ABC):
    _defaults = {
        "model_path"    : None,
        "model_config"  : None, 
        "model_type"    : None
    }

    @classmethod
    def set_defaults(cls, config):
        cls._defaults = config

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name " + n
        
    def __init__(self):
        self.__dict__.update(self._defaults)
        self.adjust_lanes = False
        self.lane_info = LaneInfo(  np.ndarray([], dtype=object),
                                    np.ndarray([], dtype=object),
                                    np.ndarray([], dtype=object),
                                    False)