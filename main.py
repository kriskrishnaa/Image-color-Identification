from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import sys
import utils
import numpy as np
import cv2
from PIL import Image, ImageStat
from tkinter import *
import PIL.Image,PIL.ImageTk
import PIL.Image
from tkinter.filedialog import askopenfilename
#from tkinter import filedialog
#from filedialog import askopenfilename


put = "yes"
while put == "yes":

    def b1():
        global put
        put = "yes"
        app.destroy()

    def b2():
        global put
        put = "no"
        app.destroy()
        mgui.destroy()

    def close(clas,output):
        global disp
        disp = Tk()
        disp.title("Result")
        disp.geometry("340x150")
        if output == '':
            label = Label(disp, text=clas, borderwidth=2, relief="raised", width=30, height=2)
        else:
            label = Label(disp, text=output, borderwidth=2, relief="raised", width=30, height=2)
        label.place(x=15,y=20)
        button4 = Button(disp, text="OK", width=10, command=pop)
        button4.place(x=120,y=80)
        disp.mainloop()

    def pop():
        global app
        disp.destroy()
        app = Tk()
        app.title("POPUP")
        app.geometry("340x150")
        label = Label(app, text="DO YOU WANT TO CONTINUE",height=0, width=50).place(x=.6,y=20)
        button1 = Button(app, text="YES", width=10, command=b1).place(x=70,y=80)
        button2 = Button(app, text="NO", width=10, command=b2).place(x=180,y=80)
        app.mainloop()

    def centroid_histogram(clt):
            # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins = numLabels)

        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()

        # return the histogram
        return hist


    def plot_colors(hist, centroids):
        # initialize the bar chart representing the relative frequency
        # of each of the colors
        bar = np.zeros((50, 300, 3), dtype = "uint8")
        startX = 0

        # loop over the percentage of each cluster and the color of
        # each cluster
        for (percent, color) in zip(hist, centroids):
                # plot the relative percentage of each cluster
                endX = startX + (percent * 300)
                cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                        color.astype("uint8").tolist(), -1)
                startX = endX
        
        # return the bar chart
        return bar


    
    def proportion(path,clas):
        # load the image and convert it from BGR to RGB so that
        # we can dispaly it with matplotlib

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # show our image
        plt.figure()
        plt.axis("off")
        plt.imshow(image)

        image = image.reshape((image.shape[0] * image.shape[1], 3))

        # cluster the pixel intensities
        clt = KMeans(3)
        clt.fit(image)


        # build a histogram of clusters and then create a figure
        # representing the number of pixels labeled to each color
        hist = utils.centroid_histogram(clt)
        #print(hist)
        #bar = utils.plot_colors(hist, clt.cluster_centers_)

        # show our color bart
        
        plt.figure()
        plt.axis("off")
        #plt.imshow(bar)
        #plt.show()
        global output
        hist.sort()
        temp = hist[2]*100 + hist[1]*100
        print(hist)
        output = "GRAYSCALE %: {name}".format(name=temp - hist[0])
        close(clas,output)
        
        

    def classify(file, thumb_size=40, MSE_cutoff=22, adjust_color_bias=True):
        pil_img = PIL.Image.open(file)
        bands = pil_img.getbands()
        if bands == ('R','G','B') or bands== ('R','G','B','A'):
            thumb = pil_img.resize((thumb_size,thumb_size))
            SSE, bias = 0, [0,0,0]
            if adjust_color_bias:
                bias = ImageStat.Stat(thumb).mean[:3]
                bias = [b - sum(bias)/3 for b in bias ]
            for pixel in thumb.getdata():
                mu = sum(pixel)/3
                SSE += sum((pixel[i] - mu - bias[i])*(pixel[i] - mu - bias[i]) for i in [0,1,2])
            MSE = float(SSE)/(thumb_size*thumb_size)
            if MSE <= MSE_cutoff:
                print("grayscale\t")
                clas = "GRAYSCALE"
            else:
                print("Color\t\t\t")
                clas = "COLOR"
            #print("( MSE=",MSE,")")
        global output
        output = ''
        if clas == "GRAYSCALE":
            proportion(file,clas)
        else:
            close(clas,output)
            pop()

    def choose():
        global path1
        path1 = askopenfilename()
        print(path1)

    def call():
        classify(path1)
        

        


    global mgui
    mgui = Tk()
    mgui.geometry( "600x400" )
    mgui.title("CLASSIFY")
    #mgui.state('zoomed')
    mutton=Button(mgui,text="CHOOSE",fg="#f87305",width="15",height="3",command = choose).place(x=220,y=134)
    mutton=Button(mgui,text="CHECK",fg="#f87305",width="15",height="3",command = call).place(x=220,y=194)
    mgui.mainloop()
        
