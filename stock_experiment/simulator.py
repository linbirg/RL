
class Simulator(object):
    data = []
    length = 0
    step = 0
    latest = []

    def __init__(self, data):
        self.data = data
        self.step = 0
        self.latest = data[0]
        self.length = len(self.data)

    def next(self):
        if not self.done():
            self.step += 1
            self.latest = self.data[self.step - 1]
        return self.latest

    def set_step(self,step):
        self.step = step
        self.latest = self.data[self.step]
    
    def done(self):
        return self.step >= self.length