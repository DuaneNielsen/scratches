import PySimpleGUI as sg

layout = [[sg.Text('Robotics Remote Control')],
          [sg.T(' ' * 10), sg.RealtimeButton('Forward')],
          [sg.RealtimeButton('Left'), sg.T(' ' * 15), sg.RealtimeButton('Right')],
          [sg.T(' ' * 10), sg.RealtimeButton('Reverse')],
          [sg.T('')],
          [sg.Quit(button_color=('black', 'orange'))]
          ]

window = sg.Window('Robotics Remote Control', auto_size_text=True).Layout(layout)

#
# Some place later in your code...
# You need to perform a Read or Refresh on your window every now and then or
# else it will appear your program has hung
#
# your program's main loop
while (True):
    # This is the code that reads and updates your window
    event, values = window.Read(timeout=10)
    if event is not None:
        print(event)
    if event == 'Quit' or values is None:
        break

window.Close()  # Don't forget to close your window!   