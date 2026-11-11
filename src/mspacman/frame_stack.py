'''
frame_stack.py

Stack frames, helps model infer time changes.
'''

import numpy as np
from collections import deque

class FrameStack:
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, frame): # frame: (88, 80)
        for _ in range(self.k):
            self.frames.append(frame) # no history yet, so repeat the first frame
        return np.stack(self.frames) # (4, 88, 80)

    def step(self, frame):
        self.frames.append(frame) # oldest frame drops out automatically
        return np.stack(self.frames)