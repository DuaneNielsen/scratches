from intervaltree import Interval, IntervalTree

t = IntervalTree()

t[0:100] = 'one'
t[101:200] = 'two'
t[201:300] = 'three'

print(t[5])
print(t[10])
print(t[202])