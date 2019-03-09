unsorted = [(5,'five'),(4,'four'),(3,'three'),(2,'two'),(1,'one')]

sorted_list = sorted(unsorted, key=lambda element: element[0])

print(sorted_list)