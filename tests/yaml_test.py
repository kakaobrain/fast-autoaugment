import yaml

# o = {
#     'a': [
#     {
#         'b':[{
#             'k':5, 'j':8
#         }]
#     }, {
#         'b':[{
#             'k':5, 'j':8
#         }]
#     }, {
#         'b':[{
#             'k':5, 'j':8
#         }]
#     }],
#     'b': [],
#     'c': {'x':1, 'y':2},
#     'd': {},
#     'e': None,
#     'f': 3,
#     'g': [7, 8, 9]
# }

# e = {'cells':[]}

# print(yaml.dump(o))

class T:
    t1 = 't1t'
    t2 = 't2t'

class B:
    def __init__(self) -> None:
        self.a = 10
        self.t = T.t2
        self.tn = None

class OpDesc:
    """Op that is part of each edge
    """
    def __init__(self, name:str, ch_in:int, ch_out:int, stride:int,
                 affine:bool)->None:
        self.name = name
        self.ch_in, self.ch_out = ch_in, ch_out
        self.stride, self.affine = stride, affine
        self.b = B()


o = OpDesc('dd', 2, 3, 4, True)
print(yaml.dump(o))

s=yaml.dump(o)
o1=yaml.load(s)
print(o1)