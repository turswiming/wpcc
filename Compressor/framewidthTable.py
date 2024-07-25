class FrameSizeTable:
    def __init__(self):
        self.table = {
            (1, 2): [4, 8, 16],
            (2, 10): [16],
            (10, 100): [32],
            (100, 1000): [32, 256],
        }

    def get_frame_sizes(self, cp_level):
        for key in self.table.keys():
            if key[0] <= cp_level and cp_level < key[1]:
                return self.table[key]
        return [32, 256]
