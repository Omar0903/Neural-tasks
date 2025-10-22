from tkinter import *
from tkinter import ttk
from functions import RunModel, featureMap, classMap


MainScreen = Tk()
MainScreen.geometry('1200x720')
MainScreen.resizable(False, False)
MainScreen.title('Neural Networks Task')
MainScreen.config(background='silver')

fr1 = Frame(MainScreen, width=1150, height=600, bg='white')
fr2 = Frame(fr1, width=500, height=180, bg='silver')
fr3 = Frame(fr1, width=570, height=180, bg='silver')
fr4 = Frame(fr1, width=500, height=300, bg='silver')


Lb1 = Label(fr1, text=' Welcome to neural networks task 1 ', fg='black', bg='silver', font=25, width=30)
Lb2 = Label(fr2, text='Enter number of epochs (m)', fg='black', bg='white', font=15, width=25)
Lb3 = Label(fr2, text='Enter MSE threshold', fg='black', bg='white', font=15, width=25)
Lb4 = Label(fr2, text='Enter learning rate (eta)', fg='black', bg='white', font=15, width=25)
Lb5 = Label(fr2, text='Add bias', fg='black', bg='white', font=15, width=25)
Lb6 = Label(fr3, text='Select two features', fg='black', bg='white', font=15, width=25)
Lb7 = Label(fr3, text='Select two classes', fg='black', bg='white', font=15, width=25)
Lb8 = Label(fr3, text='Choose algorithm', fg='black', bg='white', font=15, width=25)

Lb9  = Label(fr4, text=' Entre Sample ', fg='black', bg='white', font=25, width=25)
Lb10 = Label(fr4, text='Species', fg='black', bg='white', font=15, width=25)
Lb11 = Label(fr4, text='Culmen Depth', fg='black', bg='white', font=15, width=25)
Lb12 = Label(fr4, text='Culmen Length', fg='black', bg='white', font=15, width=25)
Lb13 = Label(fr4, text='Body Mass', fg='black', bg='white', font=15, width=25)
Lb14 = Label(fr4, text='Flipper Length', fg='black', bg='white', font=15, width=25)
Lb15 = Label(fr4, text='Origin Location', fg='black', bg='white', font=15, width=25)
Lb16 = Label(fr4, text='Values', fg='black', bg='white', font=15, width=15)



En1 = Entry(fr2, fg='black', bg='white', font=15, width=15)
En2 = Entry(fr2, fg='black', bg='white', font=15, width=15)
En3 = Entry(fr2, fg='black', bg='white', font=15, width=15)

En4 = Entry(fr4, fg='black', bg='white', font=15, width=15)
En5 = Entry(fr4, fg='black', bg='white', font=15, width=15)
En6 = Entry(fr4, fg='black', bg='white', font=15, width=15)
En7 = Entry(fr4, fg='black', bg='white', font=15, width=15)
En8 = Entry(fr4, fg='black', bg='white', font=15, width=15)
En9 = Entry(fr4, fg='black', bg='white', font=15, width=15)


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
fr4.place(x=30,y=280)

Lb1.place(x=380, y=20)
Lb2.place(x=40, y=20)
Lb3.place(x=40, y=60)
Lb4.place(x=40, y=100)
Lb5.place(x=40, y=140)


Lb6.place(x=40, y=20)
Lb7.place(x=40, y=80)
Lb8.place(x=40, y=140)

Lb9.place(x=40, y=20)
Lb10.place(x=40, y=60)
Lb11.place(x=40, y=100)
Lb12.place(x=40, y=140)
Lb13.place(x=40, y=180)
Lb14.place(x=40, y=220)
Lb15.place(x=40, y=260)
Lb16.place(x=320, y=20)


radio1.place(x=320, y=140)
radio2.place(x=400, y=140)

En1.place(x=320, y=20)
En2.place(x=320, y=60)
En3.place(x=320, y=100)

En4.place(x=320, y=60)
En5.place(x=320, y=100)
En6.place(x=320, y=140)
En7.place(x=320, y=180)
En8.place(x=320, y=220)
En9.place(x=320, y=260)

cmbo1.place(x=300, y=20, height=20, width=250)
cmbo2.place(x=300, y=80, height=20, width=250)
cmbo3.place(x=300, y=140, height=20, width=250)


runBtn = Button(fr1, text="Run Model", bg="green", fg="white", font=15,
                command=lambda: RunModel(En1, En2, En3, cmbo1, cmbo2, cmbo3, biasOption))
runBtn.place(x=950, y=400, width=150, height=40)

testSampleBtn = Button(fr1, text="test sample", bg="green", fg="white", font=15,
                command=lambda: RunModel(En1, En2, En3, cmbo1, cmbo2, cmbo3, biasOption))
testSampleBtn.place(x=750, y=400, width=150, height=40)

MainScreen.mainloop()
