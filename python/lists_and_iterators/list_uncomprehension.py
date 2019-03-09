w = [['1wl','1wc'],['2wl','2wc'],['3wl','3wc'],['4wl','4wc']]

for net in zip(*w):
    for element in net:
        print(element)

print([list(net) for net in zip(*w)])

w = [[['1wl','1wc'],['2wl','2wc']],[['3wl','3wc'],['4wl','4wc']]]

# flatten

flt = []
for sublist in w:
    for elem in sublist:
        flt.append(flt)