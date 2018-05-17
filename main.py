import os
import ntpath
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import random

import keras

from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import preprocess_image

# import miscellaneous modules
import cv2
import os
import numpy as np
import tensorflow as tf


COLORS = ['red', 'blue', 'yellow', 'pink', 'cyan', 'green', 'black']


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())

model_path = os.path.join('.', 'snapshots', 'resnet50_coco_best_v2.0.2.h5')

model = keras.models.load_model(model_path, custom_objects=custom_objects)
# print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


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
        self.hl = None
        self.vl = None
        self.filename = None
        self.objectLabelList = []

        # initialize mouse state
        self.STATE = {'click': 0, 'x': 0, 'y': 0}
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
        # self.mb.menu.bind("<Button-1>", self.add_labels_coco)
        # self.mb.menu.focus_set()

        # Image Editing Region
        self.canvas = Canvas(self.frame, width=500, height=500)
        self.canvas.grid(row=0, column=1, sticky=W + N)
        self.canvas.bind("<Button-1>", self.mouse_click)
        self.canvas.bind("<Motion>", self.mouse_move)
        self.parent.bind("Escape", self.cancel_bbox)
        # self.parent.bind("s", self.cancel_bbox)
        # self.parent.bind("a", self.open_previous)  # press 'a' to go backward
        # self.parent.bind("d", self.open_next)  # press 'd' to go forward

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

        self.cocoLabels = labels_to_names.values()
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
            self.img = self.img.resize((baseW, hsize), Image.ANTIALIAS)
        else:
            baseH = 500
            wpercent = (baseH/float(h))
            wsize = int((float(w) * float(wpercent)))
            self.img = self.img.resize((wsize, baseH), Image.ANTIALIAS)

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
        if self.STATE['click'] == 0:
            self.STATE['x'], self.STATE['y'] = event.x, event.y
            self.STATE['click'] = 1
        else:
            self.STATE['click'] = 0
            x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
            y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
            self.bboxList.append((x1, y1, x2, y2))
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            labelidx = self.labelListBox.curselection()
            label = self.labelListBox.get(labelidx)
            self.objectLabelList.append(str(label))
            self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (x1, y1, x2, y2) + ': ' + str(label))
            self.objectListBox.itemconfig(len(self.bboxIdList) - 1, fg=COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
        # self.STATE['click'] = 1 - self.STATE['click']

    def mouse_move(self, event):
        self.disp.config(text='x: %d, y: %d' % (event.x, event.y))
        if self.tkimg:
            if self.hl:
                self.canvas.delete(self.hl)
            self.hl = self.canvas.create_line(0, event.y, self.tkimg.width(), event.y, width=2)
            if self.vl:
                self.canvas.delete(self.vl)
            self.vl = self.canvas.create_line(event.x, 0, event.x, self.tkimg.height(), width=2)
        if self.STATE['click'] == 1:
            if self.bboxId:
                self.canvas.delete(self.bboxId)
            self.bboxId = self.canvas.create_rectangle(self.STATE['x'], self.STATE['y'],
                                                       event.x, event.y,
                                                       width=2,
                                                       outline=COLORS[len(self.bboxList) % len(COLORS)])

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
        _, _, boxes, nms_classification = model.predict_on_batch(np.expand_dims(image, axis=0))
        predicted_labels = np.argmax(nms_classification[0, :, :], axis=1)
        scores = nms_classification[0, np.arange(nms_classification.shape[1]), predicted_labels]
        for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
            curr_label_list = self.labelListBox.get(0, END)
            curr_label_list = list(curr_label_list)
            if score < 0.5:
                continue

            if labels_to_names[label] not in curr_label_list:
                continue

            b = boxes[0, idx, :].astype(int)

            self.bboxId = self.canvas.create_rectangle(b[0], b[1],
                                                       b[2], b[3],
                                                       width=2,
                                                       outline=COLORS[len(self.bboxList) % len(COLORS)])
            self.bboxList.append((b[0], b[1], b[2], b[3]))
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.objectLabelList.append(str(labels_to_names[label]))
            self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (b[0], b[1], b[2], b[3]) + ': ' + str(labels_to_names[label]))
            self.objectListBox.itemconfig(len(self.bboxIdList) - 1, fg=COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])


if __name__ == '__main__':
    root = Tk()
    imgicon = PhotoImage(file='icon.png')
    root.tk.call('wm', 'iconphoto', root._w, imgicon)
    tool = MainGUI(root)
    root.mainloop()
