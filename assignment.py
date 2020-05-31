
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import sys


file =pd.read_csv("2abc.csv")
print (file.head())
print("\n")

print("Press 2 for 2D plots ")
print("Press 3 for 3D plots ")

def scattor3D(head1,head2,head3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x=list(head1)
    y=list(head2)
    z=list(head3)
    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig("C:/Users/Aamir Aman/source/repos/PythonApplication1/plots/scatter3D.png")
    plt.show()  

def bar3D(head1,head2,head3):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    xpos = head1
    ypos = head2
    x=len(xpos)
    y=len(ypos)
    zpos = np.zeros(x)
    dx = np.ones(x)
    dy = np.ones(y)
    dz = head3

    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_zlabel('Z Label')
    plt.savefig("C:/Users/Aamir Aman/source/repos/PythonApplication1/plots/scatter3D.png")
    plt.show()

def threecolumns():
    new=input('enter first column  ')
    new1=input('enter second column  ')
    new2=input('enter third column  ')

    head1 = np.array(file[new])
    head2= np.array(file[new1])
    head3= np.array(file[new2])
   
    print("chose your plot\n")
    print("Press 11 for scatter3D plot")
    print("Press 22 for bar3D plot")
    print("Press 0 to quit")
    print("For switching to 2D plots press 9 ")
   
    while True:
        choice11=int(input())
        if choice11==11:
            scattor3D(head1,head2,head3)
        elif choice11==22:
            bar3D(head1,head2,head3)
        elif choice11==9:
            twocolumns()
        elif choice11==0:
            sys.exit(0)
            
    
def linear_plot(head1, head2):
        plt.plot(list(head1),list(head2),label='drugs')
        plt.ylabel('y-axis')
        plt.xlabel('X-axis')
        plt.title('linear plot')
        plt.legend()
        plt.savefig("C:/Users/Aamir Aman/source/repos/PythonApplication1/plots/linear.png")
        plt.show()

def scatter_plot(head1,head2):
    sns.scatterplot(list(head1),list(head2),s=100,label='drugs')
    plt.ylabel('y-axis')
    plt.xlabel('X-axis')
    plt.title('scatter plot')
    plt.legend()
    plt.savefig("C:/Users/Aamir Aman/source/repos/PythonApplication1/plots/scatter.png")
    plt.show()

def bar_plot(head1,head2):
    new=np.arange(len(head1))
    sns.barplot(new,list(head2),
        palette = 'hls',
        saturation = 10, 
       label='drugs'
        )   

    plt.ylabel('y-axis')
    plt.xlabel('x-axis')
    plt.title('Bar Plot')
    plt.legend(fontsize=10)
    plt.savefig("C:/Users/Aamir Aman/source/repos/PythonApplication1/plots/bar.png")
    plt.show()                       
                               
def histogram(head2,bins=10):
    sns.set()
    plt.hist(head2,bins=10,label='drugs')
    plt.ylabel('y-axis')
    plt.xlabel('X-axis')
    plt.title('Histogram')
    plt.legend()
    plt.savefig("C:/Users/Aamir Aman/source/repos/PythonApplication1/plots/histogram.png")
    plt.show()

def twocolumns():
    global new
    global new1
    new=input('enter first column  ')
    new1=input('enter second column  ')

    head1 = np.array(file[new])
    head2= np.array(file[new1]) 

    print("\n")
    print("chose your Graph\n")
    print("Press 1 for linear plot")
    print("Press 2 for Scattor plot")
    print("Press 3 for Bar plot")
    print("Press 4 for Histogram")
    print("Press 5 to get all plots")
    print("For switching to 3D plots press 9 ")
    print("Press 0 to exit\n")


    while True: 
   
        choice =int( input())
    
        if choice == 1: 
            linear_plot(head1, head2)
        elif choice == 2:
            scatter_plot(list(head1),list(head2))
        elif choice == 3:
            bar_plot(head1, head2)
        elif choice == 4:
            histogram(head2,bins=10)
        elif choice == 5:
            linear_plot(head1, head2)
            scatter_plot(list(head1),list(head2))
            bar_plot(head1, head2)
            histogram(head2,bins=10)
        elif choice==9:
             threecolumns()
        elif choice==0:
            sys.exit(0)

choice1=int(input())
if choice1==2:
    twocolumns()
elif choice1==3:
    threecolumns()
