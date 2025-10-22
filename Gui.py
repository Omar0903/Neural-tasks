from tkinter import *
from tkinter import ttk
from functions import RunModel, featureMap, classMap


MainScreen = Tk()
MainScreen.geometry('1200x720')
MainScreen.resizable(False, False)
MainScreen.title('Neural Networks Task')
MainScreen.config(background='silver')

fr1 = Frame(MainScreen, width=1150, height=470, bg='white')
fr2 = Frame(fr1, width=500, height=180, bg='silver')
fr3 = Frame(fr1, width=570, height=180, bg='silver')

Lb1 = Label(fr1, text=' Welcome to neural networks task 1 ', fg='black', bg='silver', font=25, width=30)
Lb2 = Label(fr2, text='Enter number of epochs (m)', fg='black', bg='white', font=15, width=25)
Lb3 = Label(fr2, text='Enter MSE threshold', fg='black', bg='white', font=15, width=25)
Lb4 = Label(fr2, text='Enter learning rate (eta)', fg='black', bg='white', font=15, width=25)
Lb5 = Label(fr2, text='Add bias', fg='black', bg='white', font=15, width=25)
Lb6 = Label(fr3, text='Select two features', fg='black', bg='white', font=15, width=25)
Lb7 = Label(fr3, text='Select two classes', fg='black', bg='white', font=15, width=25)
Lb8 = Label(fr3, text='Choose algorithm', fg='black', bg='white', font=15, width=25)


En1 = Entry(fr2, fg='black', bg='white', font=15, width=15)
En2 = Entry(fr2, fg='black', bg='white', font=15, width=15)
En3 = Entry(fr2, fg='black', bg='white', font=15, width=15)


biasOption = StringVar(value="Yes")
radio1 = Radiobutton(fr2, text="Yes", variable=biasOption, value="Yes")
radio2 = Radiobutton(fr2, text="No", variable=biasOption, value="No")


cmbo1 = ttk.Combobox(fr3, value=list(featureMap.keys()))
cmbo1.set('Culmen Length and Culmen Depth')

cmbo2 = ttk.Combobox(fr3, value=list(classMap.keys()))
cmbo2.set('Adelie and Gentoo')

cmbo3 = ttk.Combobox(fr3, value=('Perceptron', 'Adaline'))
cmbo3.set('Perceptron')


fr1.place(x=25, y=20)
fr2.place(x=30, y=80)
fr3.place(x=550, y=80)

Lb1.place(x=360, y=20)
Lb2.place(x=40, y=20)
Lb3.place(x=40, y=60)
Lb4.place(x=40, y=100)
Lb5.place(x=40, y=140)
radio1.place(x=320, y=140)
radio2.place(x=400, y=140)

Lb6.place(x=40, y=20)
Lb7.place(x=40, y=80)
Lb8.place(x=40, y=140)
En1.place(x=320, y=20)
En2.place(x=320, y=60)
En3.place(x=320, y=100)
cmbo1.place(x=300, y=20, height=20, width=250)
cmbo2.place(x=300, y=80, height=20, width=250)
cmbo3.place(x=300, y=140, height=20, width=250)


runBtn = Button(fr1, text="Run Model", bg="green", fg="white", font=15,
                command=lambda: RunModel(En1, En2, En3, cmbo1, cmbo2, cmbo3, biasOption))
runBtn.place(x=950, y=400, width=150, height=40)


MainScreen.mainloop()
