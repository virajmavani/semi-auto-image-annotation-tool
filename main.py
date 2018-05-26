import os

from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import random

import keras

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import cv2
import os
import numpy as np
import tensorflow as tf
import config 



def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())

model_path = os.path.join('.', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')

model = models.load_model(model_path, backbone_name='resnet50')
# print(model.summary())


class MainGUI:
    def __init__(self, master):
        self.parent = master
        self.parent.title("Semi Automatic Image Annotation Tool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=False, height=False)

        # Initialize class variables
        self.img = None
        self.tkimg = None
        self.imageDir = ''
        self.imageList = []
        self.cur = 0
        self.bboxIdList = []
        self.bboxList = []
        self.bboxId = None
        self.zoomImgId = None
        self.hl = None
        self.vl = None
        self.filename = None
        self.objectLabelList = []

        # initialize mouse state
        self.STATE = {'x': 0, 'y': 0}
        self.STATE_COCO = {'click': 0}

        # initialize annotation file
        self.anno_filename = 'annotations.csv'
        self.annotation_file = open('annotations/' + self.anno_filename, 'w+')
        self.annotation_file.write("")
        self.annotation_file.close()

        # ------------------ GUI ---------------------

        # Control Panel
        self.ctrlPanel = Frame(self.frame)
        self.ctrlPanel.grid(row=0, column=0, sticky=W + N)
        self.openBtn = Button(self.ctrlPanel, text='Open', command=self.open_image)
        self.openBtn.pack(fill=X, side=TOP)
        self.openDirBtn = Button(self.ctrlPanel, text='Open Dir', command=self.open_image_dir)
        self.openDirBtn.pack(fill=X, side=TOP)
        self.nextBtn = Button(self.ctrlPanel, text='Next -->', command=self.open_next)
        self.nextBtn.pack(fill=X, side=TOP)
        self.previousBtn = Button(self.ctrlPanel, text='<-- Previous', command=self.open_previous)
        self.previousBtn.pack(fill=X, side=TOP)
        self.saveBtn = Button(self.ctrlPanel, text='Save', command=self.save)
        self.saveBtn.pack(fill=X, side=TOP)
        self.semiAutoBtn = Button(self.ctrlPanel, text="Show Suggestions", command=self.automate)
        self.semiAutoBtn.pack(fill=X, side=TOP)
        self.disp = Label(self.ctrlPanel, text='Coordinates:')
        self.disp.pack(fill=X, side=TOP)
        self.mb = Menubutton(self.ctrlPanel, text="COCO Classes for Suggestions", relief=RAISED)
        self.mb.pack(fill=X, side=TOP)
        self.mb.menu = Menu(self.mb, tearoff=0)
        self.mb["menu"] = self.mb.menu
        self.addCocoBtn = Button(self.ctrlPanel, text="+", command=self.add_labels_coco)
        self.addCocoBtn.pack(fill=X, side=TOP)
        self.zoomPanelLabel = Label(self.ctrlPanel, text="Zoom Panel")
        self.zoomPanelLabel.pack(fill=X, side=TOP)
        self.zoomcanvas = Canvas(self.ctrlPanel, width=150, height=150)
        self.zoomcanvas.pack(fill=X, side=TOP, anchor='center')

        # Image Editing Region
        self.canvas = Canvas(self.frame, width=500, height=500)
        self.canvas.grid(row=0, column=1, sticky=W + N)
        self.canvas.bind("<Button-1>", self.mouse_click)
        self.canvas.bind("<Motion>", self.mouse_move, "+")
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_release)
        self.parent.bind("Escape", self.cancel_bbox)

        # Labels and Bounding Box Lists Panel
        self.listPanel = Frame(self.frame)
        self.listPanel.grid(row=0, column=2, sticky=W + N)
        self.listBoxNameLabel = Label(self.listPanel, text="List of Objects").pack(fill=X, side=TOP)
        self.objectListBox = Listbox(self.listPanel, width=40)
        self.objectListBox.pack(fill=X, side=TOP)
        self.delObjectBtn = Button(self.listPanel, text="Delete", command=self.del_bbox)
        self.delObjectBtn.pack(fill=X, side=TOP)
        self.clearAllBtn = Button(self.listPanel, text="Clear All", command=self.clear_bbox)
        self.clearAllBtn.pack(fill=X, side=TOP)
        self.classesNameLabel = Label(self.listPanel, text="Classes").pack(fill=X, side=TOP)
        self.textBox = Entry(self.listPanel, text="Enter label")
        self.textBox.pack(fill=X, side=TOP)

        self.addLabelBtn = Button(self.listPanel, text="+", command=self.add_label).pack(fill=X, side=TOP)
        self.delLabelBtn = Button(self.listPanel, text="-", command=self.del_label).pack(fill=X, side=TOP)

        self.labelListBox = Listbox(self.listPanel)
        self.labelListBox.pack(fill=X, side=TOP)

        self.cocoLabels = config.labels_to_names.values()
        self.cocoIntVars = []

        for idxcoco, label_coco in enumerate(self.cocoLabels):
            self.cocoIntVars.append(IntVar())
            self.mb.menu.add_checkbutton(label=label_coco, variable=self.cocoIntVars[idxcoco])
        # print(self.cocoIntVars)

    def open_image(self):
        self.filename = filedialog.askopenfilename(title="Select Image", filetypes=(("jpeg files", "*.jpg"),
                                                                               ("all files", "*.*")))
        self.load_image(self.filename)

    def open_image_dir(self):
        self.imageDir = filedialog.askdirectory(title="Select Dataset Directory")
        self.imageList = os.listdir(self.imageDir)
        self.imageList = sorted(self.imageList)
        # print(self.imageList)
        self.load_image(self.imageDir + '/' + self.imageList[self.cur])

    def load_image(self, file):
        self.img = Image.open(file)

        # Resize to Pascal VOC format
        w, h = self.img.size
        if w >= h:
            baseW = 500
            wpercent = (baseW/float(w))
            hsize = int((float(h) * float(wpercent)))
            self.img = self.img.resize((baseW, hsize), Image.BICUBIC)
        else:
            baseH = 500
            wpercent = (baseH/float(h))
            wsize = int((float(w) * float(wpercent)))
            self.img = self.img.resize((wsize, baseH), Image.BICUBIC )

        self.tkimg = ImageTk.PhotoImage(self.img)
        self.canvas.create_image(0, 0, image=self.tkimg, anchor=NW)
        self.clear_bbox()

    def open_next(self, event=None):
        self.save()
        if self.cur < len(self.imageList):
            self.cur += 1
            self.load_image(self.imageDir + '/' + self.imageList[self.cur])

    def open_previous(self, event=None):
        self.save()
        if self.cur > 0:
            self.cur -= 1
            self.load_image(self.imageDir + '/' + self.imageList[self.cur])

    def save(self):
        if self.filename is None:
            self.annotation_file = open('annotations/' + self.anno_filename, 'a')
            for idx, item in enumerate(self.bboxList):
                self.annotation_file.write(self.imageDir + '/' + self.imageList[self.cur] + ',' +
                                      ','.join(map(str, self.bboxList[idx])) + ',' + str(self.objectLabelList[idx])
                                      + '\n')
            self.annotation_file.close()
        else:
            self.annotation_file = open('annotations/' + self.anno_filename, 'a')
            for idx, item in enumerate(self.bboxList):
                self.annotation_file.write(self.filename + ',' + ','.join(map(str, self.bboxList[idx])) + ','
                                      + str(self.objectLabelList[idx]) + '\n')
            self.annotation_file.close()

    def mouse_click(self, event):
        self.STATE['x'], self.STATE['y'] = event.x, event.y

    def mouse_drag(self, event):
        self.mouse_move(event)
        if self.bboxId:
            self.canvas.delete(self.bboxId)
        self.bboxId = self.canvas.create_rectangle(self.STATE['x'], self.STATE['y'],
                                                   event.x, event.y,
                                                   width=2,
                                                   outline=config.COLORS[len(self.bboxList) % len(config.COLORS)])

    def mouse_move(self, event):
        self.disp.config(text='x: %d, y: %d' % (event.x, event.y))
        self.zoom_view(event)
        if self.tkimg:
            if self.hl:
                self.canvas.delete(self.hl)
            self.hl = self.canvas.create_line(0, event.y, self.tkimg.width(), event.y, width=2)
            if self.vl:
                self.canvas.delete(self.vl)
            self.vl = self.canvas.create_line(event.x, 0, event.x, self.tkimg.height(), width=2)

    def mouse_release(self, event):
        x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
        y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
        self.bboxList.append((x1, y1, x2, y2))
        self.bboxIdList.append(self.bboxId)
        self.bboxId = None
        labelidx = self.labelListBox.curselection()
        label = self.labelListBox.get(labelidx)
        self.objectLabelList.append(str(label))
        self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (x1, y1, x2, y2) + ': ' + str(label))
        self.objectListBox.itemconfig(len(self.bboxIdList) - 1, fg=config.COLORS[(len(self.bboxIdList) - 1) % len(config.COLORS)])

    def zoom_view(self, event):
        try:
            if self.zoomImgId:
                self.zoomcanvas.delete(self.zoomImgId)
            self.zoomImg = self.img.copy()
            draw = ImageDraw.Draw(self.zoomImg)
            draw.point((event.x, event.y), fill=(0,0,0))
            self.zoomImgCrop = self.zoomImg.crop(((event.x-25), (event.y-25), (event.x+25), (event.y+25)))
            self.zoomImgCrop = self.zoomImgCrop.resize((150, 150))
            self.tkZoomImg = ImageTk.PhotoImage(self.zoomImgCrop)
            self.zoomImgId = self.zoomcanvas.create_image(0, 0, image=self.tkZoomImg, anchor=NW)
        except:
            pass

    def update_bbox(self, event):
        pass

    def cancel_bbox(self, event):
        if self.STATE['click'] == 1:
            if self.bboxId:
                self.canvas.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0

    def del_bbox(self):
        sel = self.objectListBox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        self.canvas.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.objectListBox.delete(idx)

    def clear_bbox(self):
        for idx in range(len(self.bboxIdList)):
            self.canvas.delete(self.bboxIdList[idx])
        self.objectListBox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []
        self.objectLabelList = []

    def add_label(self):
        if self.textBox.get() is not '':
            curr_label_list = self.labelListBox.get(0, END)
            curr_label_list = list(curr_label_list)
            if self.textBox.get() not in curr_label_list:
                self.labelListBox.insert(END, str(self.textBox.get()))
            self.textBox.delete(0, 'end')

    def del_label(self):
        labelidx = self.labelListBox.curselection()
        self.labelListBox.delete(labelidx)

    def add_labels_coco(self):
        for listidxcoco, list_label_coco in enumerate(self.cocoLabels):
            if self.cocoIntVars[listidxcoco].get():
                curr_label_list = self.labelListBox.get(0, END)
                curr_label_list = list(curr_label_list)
                if list_label_coco not in curr_label_list:
                    self.labelListBox.insert(END, str(list_label_coco))

    def automate(self):
        opencvImage = cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2BGR)
        image = preprocess_image(opencvImage)
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        for idx, (box, label, score) in enumerate(zip(boxes[0], labels[0], scores[0])):
            curr_label_list = self.labelListBox.get(0, END)
            curr_label_list = list(curr_label_list)
            if score < 0.5:
                continue

            if config.labels_to_names[label] not in curr_label_list:
                continue

            b = box.astype(int)

            self.bboxId = self.canvas.create_rectangle(b[0], b[1],
                                                       b[2], b[3],
                                                       width=2,
                                                       outline=config.COLORS[len(self.bboxList) % len(config.COLORS)])
            self.bboxList.append((b[0], b[1], b[2], b[3]))
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.objectLabelList.append(str(config.labels_to_names[label]))
            self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (b[0], b[1], b[2], b[3]) + ': ' +
                                      str(config.labels_to_names[label]))
            self.objectListBox.itemconfig(len(self.bboxIdList) - 1, fg=config.COLORS[(len(self.bboxIdList) - 1) % len(config.COLORS)])


if __name__ == '__main__':
    root = Tk()
    imgicon = PhotoImage(file='icon.png')
    root.tk.call('wm', 'iconphoto', root._w, imgicon)
    tool = MainGUI(root)
    root.mainloop()
