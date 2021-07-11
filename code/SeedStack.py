import numpy as np

class SeedStack(object):
    def __init__(self):
        self.seeds = []
        self.oldseeds = [] #记录之前记录过的种子点（不含扩展点）

    def append(self, t):
        t0 = t[0] if t[0] is None else np.array(t[0])
        t1 = t[1] if t[1] is None else np.array(t[1])
        t = (t0, t1, t[2])
        # t 是一个种子点和扩展点的tuple,(seed, extend_point, ismannual) , ismanual=1表示是手动放置的种子（包括初始时自动搜到的）
        # 当t[1]==None,表示该种子点是临时种子点，还未初始化的种子点，一般是自动搜索时产生的
        for tt in self.oldseeds:
            if t[1] is None and tt[1] is None and (tt[0]==t[0]).all(): #(seed, None, ismanual)这样的
                return False
            if t[1] is not None and tt[1] is not None and (tt[0] == t[0]).all() and (tt[1] == t[1]).all():
                return False
        # try:
        #     if self.oldseeds.index(t)>=1999:
        #         return False
        # except ValueError:
        #     # t is not in self.oldseeds
        #     pass


        self.seeds.append(t)
        self.oldseeds.append(t)
        return True

    def pop(self):
        if len(self.seeds)>0:
            return self.seeds.pop()
        else:
            raise RuntimeError('the seeds stack is empty')

    def __len__(self):
        return len(self.seeds)

    def clear(self):
        self.seeds.clear()

    def __getitem__(self, index):
        print('index',index)
        if index >= len(self.seeds) or index < 0:
            raise RuntimeError('index out of range')
        else:
            return self.seeds[index]

    def __iter__(self):
        for i in range (len(self.seeds)):
            yield self.seeds[i]

    def __setitem__(self, index, value):
        if index >= len(self.seeds) or index < 0:
            raise RuntimeError('index out of range')
        else:
            self.seeds[index] = value

    def __str__(self):
        return self.seeds.__str__()

    def reverse(self):
        self.seeds.reverse()

    def empty(self):
        if self.__len__() <=0 :
            return True
        else:
            return False

