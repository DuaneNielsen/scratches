import PySimpleGUI as sg
import time

for i in range(1000):
    sg.OneLineProgressMeter('One Line Meter Example', i + 1, 1000, 'key')
    time.sleep(0.05)