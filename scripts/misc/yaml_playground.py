import yaml

y = """


d: &d
  f: 21
  g: 31

d1:
  f: 21
  g: 31

c:
  d: *d

"""

d=yaml.load(y, Loader=yaml.Loader)
print(d)
print(d['d']==d['c']['d'])
print(d['d1']==d['c']['d'])