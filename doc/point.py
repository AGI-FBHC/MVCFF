import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("/home/yang/sda/github/MVCFF/doc/img/O75821.png")

fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("Click on the image")

def onclick(event):
    if event.xdata and event.ydata:
        print("x =", event.xdata, ", y =", event.ydata)

fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()