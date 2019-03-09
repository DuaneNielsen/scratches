import PySimpleGUI as sg

sg.Popup('This is my first message box')

sg.Popup('This box has a custom button color',
         button_color=('black', 'yellow'))

layout = [[sg.Text('My one-shot window.')],
                 [sg.InputText(), sg.FileBrowse()],
                 [sg.Submit(), sg.Cancel()]]

window = sg.Window('Window Title').Layout(layout)

event, values = window.Read()
window.Close()

source_filename = values[0]



