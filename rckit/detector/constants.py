from enum import IntEnum

class OcularEventMask(IntEnum): 
    NoEvent = 0
    Blink = 1
    Saccade = 2
    Fixation = 3

