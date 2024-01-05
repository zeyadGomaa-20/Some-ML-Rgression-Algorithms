from tkinter import messagebox
from tkinter import *
import tkinter as tk
from linear import plot1, lin
from KNR import knr, plot2
from class_tree import classT , plot3
from regressionTree import ret , plot4
from SVR import svr , plot5


def exit_app():
    msg_box = messagebox.askquestion("Exit Application", "Are you sure you want to exit the application?", icon="warning")
    if msg_box == "yes":
        window.destroy()
    else:
        messagebox.showinfo("Return", "You will now return to the application screen")

def show_page1():
    page2.pack_forget()
    page1.pack()

def show_page2(command):
    page1.pack_forget()
    page2.pack()

    buttonP1.config(command=command)

# Create the main window
window = tk.Tk()

# Create Page 1 and 2
page1 = tk.Frame(window)
page2 = tk.Frame(window)
label1 = tk.Label(page1, text="Hi Doctor Nesma \nClick The button which do what you want." , background="#C19534" , height=4 , width=110 , fg='black',font=('impact' , 20))
label2 = tk.Label(page2, text="Results Doctor." , background="#C19534" , height=4 , width=110 , fg='black',font=('impact' , 20))
label1.pack()
label2.pack()

# Page two contents
acc = Label(page2, width=1000, bg='#C19534', border=10 , justify='center')
acc.pack()
cmm = Label(page2, width=1000, bg='#C19534', border=10 , justify='center')
cmm.pack()
ss = Label(page2, width=1000, bg='#C19534', border=10 , justify='center')
ss.pack()
sss = Label(page2, width=1000, bg='#C19534', border=10 , justify='center')
sss.pack()
mr = Label(page2, width=1000, bg='#C19534', border=10 , justify='center')
mr.pack()
ms = Label(page2, width=1000, bg='#C19534', border=10 , justify='center')
ms.pack()
ma = Label(page2, width=1000, bg='#C19534', border=10 , justify='center')
ma.pack()



def execute_lin():
    lin(acc, cmm, ss, sss, mr, ms, ma)
    show_page2(plot1)

def execute_knr():
    knr(acc, cmm, ss, sss, mr, ms, ma)
    show_page2(plot2)

def execute_classT():
    classT(acc, cmm, ss, sss, mr, ms, ma)
    show_page2(plot3)

def execute_ret():
    ret(acc, cmm, ss, sss, mr, ms, ma)
    show_page2(plot4)

def execute_svr():
    svr(acc, cmm, ss, sss, mr, ms, ma)
    show_page2(plot5)


# Page one buttons
button1 = tk.Button(page1, text="Linear Regression", command=execute_lin , bg='#453A42' , font= 15 ,fg='black' , activebackground='#CCD0F3' , height=4 , width=123)
button1.pack()

button2 = tk.Button(page1, text="K-Nearest Neighbors Regression", command=execute_knr , bg='#453A42' , font= 15 ,fg='black' , activebackground='#CCD0F3' , height=4 , width=123)
button2.pack()

button3 = tk.Button(page1, text="Classification Tree", command=execute_classT , bg='#453A42' , font= 15 ,fg='black' , activebackground='#CCD0F3' , height=4 , width=123)
button3.pack()

button4 = tk.Button(page1, text="Regression Tree", command=execute_ret , bg='#453A42' , font= 15 ,fg='black' , activebackground='#CCD0F3' , height=4 , width=123)
button4.pack()

button5 = tk.Button(page1, text="SVM Regression", command=execute_svr , bg='#453A42' , font= 15 ,fg='black' , activebackground='#CCD0F3' , height=4 , width=123)
button5.pack()


# Page two buttons
buttonP1 = tk.Button(page2, text="Plotting"  , bg='#453A42' , font= 15 ,fg='black' , activebackground='#CCD0F3' , height=4 , width=123)
buttonP1.pack()

buttonP2 = tk.Button(page2, text="Back" , command=show_page1  , bg='#453A42' , font= 15 ,fg='black' , activebackground='#CCD0F3' , height=4 , width=123)
buttonP2.pack()


# Show page1
show_page1()

#GUI window settings
window.title("Machine Learning Algorithms")
window.geometry("1370x750")
window.config(bg="#C19534")
window.protocol("WM_DELETE_WINDOW", exit_app)
window.mainloop()
