class PLA:
    def __init__(self, x, y):
        if type(x) == list and type(y) == list and len(x) == len(y):
            self.x = x
            self.y = y
            self.n = len(x)
            self.d = len(x[0])
        else:
            print("some error")

    def compute_weight(self):
        w = [0] * (len(self.d) + 1)
        for n in range(self.n):
            for d in range(1, self.d):


