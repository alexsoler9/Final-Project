from enum import Enum

class LaneModelType(Enum):
    UFLD_TUSIMPLE = 0
    UFLDV2_TUSIMPLE = 1

lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255), (0,128,255),(255,128,0),(76,0,153),(128,255,255)]