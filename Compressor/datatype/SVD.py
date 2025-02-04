class SVD_Data:
    def __init__(self, Us, Ss, Vs,Us_range,Ss_range,Vs_range):
        self.Us = Us
        self.Ss = Ss
        self.Vs = Vs
        self.Us_range = Us_range
        self.Ss_range = Ss_range
        self.Vs_range = Vs_range

    def get(self):
        return self.Us, self.Ss, self.Vs,self.Us_range,self.Ss_range,self.Vs_range

    def getU(self):
        return self.Us

    def getS(self):
        return self.Ss

    def getV(self):
        return self.Vs

    def set(self, Us, Ss, Vs):
        self.Us = Us
        self.Ss = Ss
        self.Vs = Vs

    def setU(self, Us):
        self.Us = Us

    def setS(self, Ss):
        self.Ss = Ss

    def setV(self, Vs):
        self.Vs = Vs

    def __str__(self):
        return f"Us: {self.Us}, Ss: {self.Ss}, Vs: {self.Vs}"