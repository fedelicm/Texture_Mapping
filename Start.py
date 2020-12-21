import tkinter as Tk
from tkinter import Checkbutton
from tkinter import StringVar, IntVar, Checkbutton
from tkinter.filedialog import askopenfilename
import sys
import tkinter.font as tkf

import TextureMapper
import threading



if(__name__ == "__main__"):
    window = Tk.Tk()
    window.resizable(False, False)
    window.title("Texture Mapper")
    sfm_location = None
    mesh_location = None
    texture_size = 8192
    console_output = ""
    std = None
    Console = None
    correct_mesh_file = False
    correct_sfm_file = False
    txt_size_text = None
    sfm_file_text = StringVar()
    sfm_file_text.set("SFM file: ")
    mesh_file_text = StringVar()
    mesh_file_text.set("Mesh file: ")
    sfm_file_label = Tk.Label(window, textvariable=sfm_file_text, relief=Tk.RAISED, width=60,
                              height=3, wraplength=400, justify=Tk.LEFT, anchor='w')
    mesh_file_label = Tk.Label(window, textvariable=mesh_file_text, relief=Tk.RAISED, width=60,
                               height=3, wraplength=400, justify=Tk.LEFT, anchor='w')
    size_label_text = StringVar()
    size_label_text.set("Texture Map Size: ")
    size_label = Tk.Label(window, textvariable=size_label_text, relief=Tk.RAISED, width=40,
                               height=2, wraplength=300, justify=Tk.LEFT, anchor='w')

def sel_mesh():
    global correct_mesh_file
    global mesh_location

    text = askopenfilename(filetypes=[("Wavefront .OBJ","*.obj")])

    if len(text)>0:
        correct_mesh_file = True
        mesh_file_text.set(text)
        mesh_location = text
    elif(correct_mesh_file and len(text)<=0):
        return

    if(correct_mesh_file and correct_sfm_file):
        start_btn.config(state=Tk.NORMAL)
    else:
        start_btn.config(state=Tk.DISABLED)

def sel_sfm():
    global correct_sfm_file
    global sfm_location
    text = askopenfilename(filetypes=[("Meshroom SFM","*.sfm")])

    if len(text)>0:
        correct_sfm_file = True
        sfm_file_text.set(text)
        sfm_location = text
    elif(correct_sfm_file and len(text)<=0):
        return

    if(correct_mesh_file and correct_sfm_file):
        start_btn.config(state=Tk.NORMAL)
    else:
        start_btn.config(state=Tk.DISABLED)


class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.config(state=Tk.NORMAL)
        self.widget.insert("end", str)
        self.widget.see("end")
        self.widget.config(state=Tk.DISABLED)

    def flush(self):
        pass

mesh_sel_btn = Tk.Button(
    text="Browse",
    width=8,
    height=2,
    bg="grey",
    fg="yellow",
    command = sel_mesh,
)

sfm_sel_btn = Tk.Button(
    text="Browse",
    width=8,
    height=2,
    bg="grey",
    fg="yellow",
    command = sel_sfm,
)

def popupmsg(msg):
    popup = Tk.Tk()
    popup.wm_title("Error")
    label = Tk.Label(popup, text=msg)
    label.pack(side="top", fill="x", pady=10)
    B1 = Tk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()

def start():
    val = int(txt_size_text.get())
    if(val != 2048 and val != 4096 and val != 8192 and val !=16384):
        popupmsg("Texture Size should be either 2048, 4096, 8192, or 16384")
    else:
        start_btn.config(state=Tk.DISABLED)
        threading.Thread(target=TextureMapper.Start, args=(mesh_location, sfm_location, val)).start()


start_btn = Tk.Button(
    text="Start",
    width=8,
    height=2,
    bg="grey",
    fg="yellow",
    command = start,
    state=Tk.DISABLED
)

def ValidateValue(S):
    if S in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        return True
    return False

if(__name__ == "__main__"):
    mesh_file_label.grid(row=0, column=0)
    mesh_sel_btn.grid(row=0,column=3)
    sfm_file_label.grid(row=1, column=0)
    sfm_sel_btn.grid(row=1,column=3)
    start_btn.grid(row=4,column=3)

    vcmd = (window.register(ValidateValue), '%S')
    Console = Tk.Text(window, height = 10, width=60, state=Tk.DISABLED)
    txt_size_text = Tk.Entry(window, width=10)
    txt_size_text.insert(0, "8192")
    txt_size_text.config(validate='key', vcmd=vcmd)
    Console.grid(row=3,column=0)
    txt_size_text.grid(row=4, column=0)
    txt_size_text.insert(Tk.INSERT, "8192")
    size_label.grid(row=4, column=0)

    txt_size_text.insert(0, "8192")

    sys.stdout = TextRedirector(Console)
    window.mainloop()
