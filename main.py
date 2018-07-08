"""
Copyright {2018} {Viraj Mavani}

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0
"""

from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

import keras

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image

# import miscellaneous modules
import os
import numpy as np
import tensorflow as tf
import config
import math


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


keras.backend.tensorflow_backend.set_session(get_session())

model_path = os.path.join('.', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')

model = models.load_model(model_path, backbone_name='resnet50')


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
        self.imageDirPathBuffer = ''
        self.imageList = []
        self.imageTotal = 0
        self.imageCur = 0
        self.cur = 0
        self.bboxIdList = []
        self.bboxList = []
        self.bboxPointList = []
        self.o1 = None
        self.o2 = None
        self.o3 = None
        self.o4 = None
        self.bboxId = None
        self.currLabel = None
        self.editbboxId = None
        self.zoomImgId = None
        self.zoomImg = None
        self.zoomImgCrop = None
        self.tkZoomImg = None
        self.hl = None
        self.vl = None
        self.editPointId = None
        self.filename = None
        self.filenameBuffer = None
        self.objectLabelList = []
        self.EDIT = False

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
        self.zoomPanelLabel = Label(self.ctrlPanel, text="Precision View Panel")
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
        self.parent.bind("<Key-Left>", self.open_previous)
        self.parent.bind("<Key-Right>", self.open_next)
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

        # STATUS BAR
        self.statusBar = Frame(self.frame, width=500)
        self.statusBar.grid(row=1, column=1, sticky=W + N)
        self.processingLabel = Label(self.statusBar, text="                      ")
        self.processingLabel.pack(side="left", fill=X)
        self.imageIdxLabel = Label(self.statusBar, text="                      ")
        self.imageIdxLabel.pack(side="right", fill=X)

    def open_image(self):
        self.filename = filedialog.askopenfilename(title="Select Image", filetypes=(("jpeg files", "*.jpg"),
                                                                                    ("all files", "*.*")))
        if not self.filename:
            return None
        self.filenameBuffer = self.filename
        self.load_image(self.filenameBuffer)

    def open_image_dir(self):
        self.imageDir = filedialog.askdirectory(title="Select Dataset Directory")
        if not self.imageDir:
            return None
        self.imageList = os.listdir(self.imageDir)
        self.imageList = sorted(self.imageList)
        self.imageTotal = len(self.imageList)
        self.filename = None
        self.imageDirPathBuffer = self.imageDir
        self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])

    def load_image(self, file):
        self.img = Image.open(file)
        self.imageCur = self.cur + 1
        self.imageIdxLabel.config(text='  ||   Image Number: %d / %d' % (self.imageCur, self.imageTotal))
        # Resize to Pascal VOC format
        w, h = self.img.size
        if w >= h:
            baseW = 500
            wpercent = (baseW / float(w))
            hsize = int((float(h) * float(wpercent)))
            self.img = self.img.resize((baseW, hsize), Image.BICUBIC)
        else:
            baseH = 500
            wpercent = (baseH / float(h))
            wsize = int((float(w) * float(wpercent)))
            self.img = self.img.resize((wsize, baseH), Image.BICUBIC)

        self.tkimg = ImageTk.PhotoImage(self.img)
        self.canvas.create_image(0, 0, image=self.tkimg, anchor=NW)
        self.clear_bbox()

    def open_next(self, event=None):
        self.save()
        if self.cur < len(self.imageList):
            self.cur += 1
            self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])
        self.processingLabel.config(text="                      ")
        self.processingLabel.update_idletasks()

    def open_previous(self, event=None):
        self.save()
        if self.cur > 0:
            self.cur -= 1
            self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])
        self.processingLabel.config(text="                      ")
        self.processingLabel.update_idletasks()

    def save(self):
        if self.filenameBuffer is None:
            self.annotation_file = open('annotations/' + self.anno_filename, 'a')
            for idx, item in enumerate(self.bboxList):
                self.annotation_file.write(self.imageDirPathBuffer + '/' + self.imageList[self.cur] + ',' +
                                           ','.join(map(str, self.bboxList[idx])) + ',' + str(self.objectLabelList[idx])
                                           + '\n')
            self.annotation_file.close()
        else:
            self.annotation_file = open('annotations/' + self.anno_filename, 'a')
            for idx, item in enumerate(self.bboxList):
                self.annotation_file.write(self.filenameBuffer + ',' + ','.join(map(str, self.bboxList[idx])) + ','
                                           + str(self.objectLabelList[idx]) + '\n')
            self.annotation_file.close()

    def mouse_click(self, event):
        # Check if Updating BBox
        if self.canvas.find_enclosed(event.x - 5, event.y - 5, event.x + 5, event.y + 5):
            self.EDIT = True
            self.editPointId = int(self.canvas.find_enclosed(event.x - 5, event.y - 5, event.x + 5, event.y + 5)[0])
        else:
            self.EDIT = False

        # Set the initial point
        if self.EDIT:
            idx = self.bboxPointList.index(self.editPointId)
            self.editbboxId = self.bboxIdList[math.floor(idx/4.0)]
            self.bboxId = self.editbboxId
            pidx = self.bboxIdList.index(self.editbboxId)
            pidx = pidx * 4
            self.o1 = self.bboxPointList[pidx]
            self.o2 = self.bboxPointList[pidx + 1]
            self.o3 = self.bboxPointList[pidx + 2]
            self.o4 = self.bboxPointList[pidx + 3]
            if self.editPointId == self.o1:
                a, b, c, d = self.canvas.coords(self.o3)
            elif self.editPointId == self.o2:
                a, b, c, d = self.canvas.coords(self.o4)
            elif self.editPointId == self.o3:
                a, b, c, d = self.canvas.coords(self.o1)
            elif self.editPointId == self.o4:
                a, b, c, d = self.canvas.coords(self.o2)
            self.STATE['x'], self.STATE['y'] = int((a+c)/2), int((b+d)/2)
        else:
            self.STATE['x'], self.STATE['y'] = event.x, event.y

    def mouse_drag(self, event):
        self.mouse_move(event)
        if self.bboxId:
            self.canvas.delete(self.bboxId)
            self.canvas.delete(self.o1)
            self.canvas.delete(self.o2)
            self.canvas.delete(self.o3)
            self.canvas.delete(self.o4)
        if self.EDIT:
            self.bboxId = self.canvas.create_rectangle(self.STATE['x'], self.STATE['y'],
                                                       event.x, event.y,
                                                       width=2,
                                                       outline=config.COLORS[(len(self.bboxList) - 1) % len(config.COLORS)])
        else:
            self.bboxId = self.canvas.create_rectangle(self.STATE['x'], self.STATE['y'],
                                                       event.x, event.y,
                                                       width=2,
                                                       outline=config.COLORS[len(self.bboxList) % len(config.COLORS)])

    def mouse_move(self, event):
        self.disp.config(text='x: %d, y: %d' % (event.x, event.y))
        self.zoom_view(event)
        if self.tkimg:
            # Horizontal and Vertical Line for precision
            if self.hl:
                self.canvas.delete(self.hl)
            self.hl = self.canvas.create_line(0, event.y, self.tkimg.width(), event.y, width=2)
            if self.vl:
                self.canvas.delete(self.vl)
            self.vl = self.canvas.create_line(event.x, 0, event.x, self.tkimg.height(), width=2)
            # elif (event.x, event.y) in self.bboxBRPointList:
            #     pass

    def mouse_release(self, event):
        try:
            labelidx = self.labelListBox.curselection()
            self.currLabel = self.labelListBox.get(labelidx)
        except:
            pass
        if self.EDIT:
            self.update_bbox()
            self.EDIT = False
        x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
        y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
        self.bboxList.append((x1, y1, x2, y2))
        o1 = self.canvas.create_oval(x1 - 3, y1 - 3, x1 + 3, y1 + 3, fill="red")
        o2 = self.canvas.create_oval(x2 - 3, y1 - 3, x2 + 3, y1 + 3, fill="red")
        o3 = self.canvas.create_oval(x2 - 3, y2 - 3, x2 + 3, y2 + 3, fill="red")
        o4 = self.canvas.create_oval(x1 - 3, y2 - 3, x1 + 3, y2 + 3, fill="red")
        self.bboxPointList.append(o1)
        self.bboxPointList.append(o2)
        self.bboxPointList.append(o3)
        self.bboxPointList.append(o4)
        self.bboxIdList.append(self.bboxId)
        self.bboxId = None
        self.objectLabelList.append(str(self.currLabel))
        self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (x1, y1, x2, y2) + ': ' + str(self.currLabel))
        self.objectListBox.itemconfig(len(self.bboxIdList) - 1,
                                      fg=config.COLORS[(len(self.bboxIdList) - 1) % len(config.COLORS)])
        self.currLabel = None

    def zoom_view(self, event):
        try:
            if self.zoomImgId:
                self.zoomcanvas.delete(self.zoomImgId)
            self.zoomImg = self.img.copy()
            self.zoomImgCrop = self.zoomImg.crop(((event.x - 25), (event.y - 25), (event.x + 25), (event.y + 25)))
            self.zoomImgCrop = self.zoomImgCrop.resize((150, 150))
            self.tkZoomImg = ImageTk.PhotoImage(self.zoomImgCrop)
            self.zoomImgId = self.zoomcanvas.create_image(0, 0, image=self.tkZoomImg, anchor=NW)
            hl = self.zoomcanvas.create_line(0, 75, 150, 75, width=2)
            vl = self.zoomcanvas.create_line(75, 0, 75, 150, width=2)
        except:
            pass

    def update_bbox(self):
        idx = self.bboxIdList.index(self.editbboxId)
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.objectListBox.delete(idx)
        self.currLabel = self.objectLabelList[idx]
        self.objectLabelList.pop(idx)
        idx = idx*4
        self.canvas.delete(self.bboxPointList[idx])
        self.canvas.delete(self.bboxPointList[idx+1])
        self.canvas.delete(self.bboxPointList[idx+2])
        self.canvas.delete(self.bboxPointList[idx+3])
        self.bboxPointList.pop(idx)
        self.bboxPointList.pop(idx)
        self.bboxPointList.pop(idx)
        self.bboxPointList.pop(idx)

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
        self.canvas.delete(self.bboxPointList[idx * 4])
        self.canvas.delete(self.bboxPointList[(idx * 4) + 1])
        self.canvas.delete(self.bboxPointList[(idx * 4) + 2])
        self.canvas.delete(self.bboxPointList[(idx * 4) + 3])
        self.bboxPointList.pop(idx * 4)
        self.bboxPointList.pop(idx * 4)
        self.bboxPointList.pop(idx * 4)
        self.bboxPointList.pop(idx * 4)
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.objectLabelList.pop(idx)
        self.objectListBox.delete(idx)

    def clear_bbox(self):
        for idx in range(len(self.bboxIdList)):
            self.canvas.delete(self.bboxIdList[idx])
        for idx in range(len(self.bboxPointList)):
            self.canvas.delete(self.bboxPointList[idx])
        self.objectListBox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []
        self.objectLabelList = []
        self.bboxPointList = []

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
        self.processingLabel.config(text="Processing     ")
        self.processingLabel.update_idletasks()
        open_cv_image = np.array(self.img)
        # Convert RGB to BGR
        opencvImage= open_cv_image[:, :, ::-1].copy()
        # opencvImage = cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2BGR)
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
            o1 = self.canvas.create_oval(b[0] - 3, b[1] - 3, b[0] + 3, b[1] + 3, fill="red")
            o2 = self.canvas.create_oval(b[2] - 3, b[1] - 3, b[2] + 3, b[1] + 3, fill="red")
            o3 = self.canvas.create_oval(b[2] - 3, b[3] - 3, b[2] + 3, b[3] + 3, fill="red")
            o4 = self.canvas.create_oval(b[0] - 3, b[3] - 3, b[0] + 3, b[3] + 3, fill="red")
            self.bboxPointList.append(o1)
            self.bboxPointList.append(o2)
            self.bboxPointList.append(o3)
            self.bboxPointList.append(o4)
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.objectLabelList.append(str(config.labels_to_names[label]))
            self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (b[0], b[1], b[2], b[3]) + ': ' +
                                      str(config.labels_to_names[label]))
            self.objectListBox.itemconfig(len(self.bboxIdList) - 1,
                                          fg=config.COLORS[(len(self.bboxIdList) - 1) % len(config.COLORS)])
        self.processingLabel.config(text="Done              ")


if __name__ == '__main__':
    root = Tk()
    imgicon = PhotoImage(file='icon.gif')
    root.tk.call('wm', 'iconphoto', root._w, imgicon)
    tool = MainGUI(root)
    root.mainloop()
