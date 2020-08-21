'''
Interactive Image Segmentation using GrabCut algorithm.

Please first select a picture.
Draw a rectangle around the object using mouse right button.
then press the following keys to draw mask using mouse left button.

Key 'n' - To update the segmentation

Key '0' - To select areas (adding masks) of sure background
Key '1' - To select areas (adding masks) of sure foreground

Key 'r' - Reset
Key 'Esc' - Finish adding masks
'''

# the grabcut function and some macroes are in this file
from grabcut_func import *

import sys
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox


# setting up flags
rect = (0, 0, 1, 1)
drawing = False # flag for drawing curves
rectangle = False # flag for drawing rect
rect_over = False # flag to check if rect drawn
rect_or_mask = 100 # flag for selecting rect or mask mode
value = DRAW_NULL # drawing initialized to NULL
thickness = 3 # brush thickness
IS_THERE_A_RESULT = False # whether the user has started grabcut segmentation

# ------------------------------
# Instantiate object, create window window
window = tk.Tk()

# Name the window
window.title('GrabCut')

# Set the size of the window
win_len = 400
win_wid = 500
window.geometry(str(win_wid) + 'x' + str(win_len))
# ------------------------------

# ------------------------------
# Set label (description) on GUI
description = tk.Label(window, text='Implementation of GrabCut', font=('Arial', 13), width=50, height=2)

# Place the label
description.pack()

# Set label (copy right) on GUI
copy_right = tk.Label(window, text='Copyright © 2019 Anton Wang. All rights reserved. ', font=('Arial', 7), width=50, height=2)

# Place the label
copy_right.pack(side='bottom')

# ------------------------------
# Shape position of button and button slogan
frame_len = 100
button_len = 15
label_len = 100
button_interval = 20

# Instruction
INSTRUCTION_var = tk.StringVar()

# setting up flags of buttons
IMPORT_PIC_on_hit = False

def onmouse(event, x, y, flags, param):
    global img, img2, drawing, value, mask, rectangle, rect, rect_or_mask, ix, iy, rect_over
    global IS_THERE_A_RESULT

    # Draw Rectangle
    if event == cv.EVENT_RBUTTONDOWN:
        rectangle = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv.rectangle(img, (ix, iy), (x, y), BLUE, 2)
            rect = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))
            rect_or_mask = 0

    elif event == cv.EVENT_RBUTTONUP:
        rectangle = False
        cv.rectangle(img, (ix, iy), (x, y), BLUE, 2)
        rect = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))
        rect_or_mask = 0

        IS_THERE_A_RESULT = True # from here you can save the result
        messagebox.showinfo("Next step!", "Now press the key 'n' if no further change and wait for a few seconds. \n Then you can draw the masks for foreground (press key '1') or background (press key '0').")


    # draw touchup curves

    if event == cv.EVENT_LBUTTONDOWN:
        if rect_over == False:
            messagebox.showinfo("Error!", "Please first draw rectangle!")
        else:
            drawing = True
            cv.circle(img, (x, y), thickness, value['color'], -1)
            if value['val']!=-1:
                cv.circle(mask, (x, y), thickness, value['val'], -1)

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            cv.circle(img, (x, y), thickness, value['color'], -1)
            if value['val'] != -1:
                cv.circle(mask, (x, y), thickness, value['val'], -1)

    elif event == cv.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv.circle(img, (x, y), thickness, value['color'], -1)
            if value['val'] != -1:
                cv.circle(mask, (x, y), thickness, value['val'], -1)

def conducting_grabcut(filename):
    global img, img2, drawing, value, mask, rectangle, rect, rect_or_mask, ix, iy, rect_over, output

    img = cv.imread(filename)
    img2 = img.copy()  # a copy of original image
    mask = np.zeros(img.shape[:2], dtype=np.uint8)  # mask initialized to PR_BG
    output = 255*np.ones(img.shape, np.uint8)  # output image to be shown

    # input and output windows
    cv.namedWindow('output')
    cv.namedWindow('input')
    cv.setMouseCallback('input', onmouse)
    cv.moveWindow('input', img.shape[1] + 10, 90)

    operation_count=0
    MAX_OPERATION=100

    while(operation_count<MAX_OPERATION):

        cv.imshow('output', output)
        cv.imshow('input', img)

        key = cv.waitKey(1)

        if key==27:
            cv.destroyAllWindows()
            break
        elif key==ord('0'):
            value = DRAW_BG
            messagebox.showinfo("Attention!", "Draw masks for background. \n Please press the key 'n' after drawing the masks and wait for a few seconds.")
            operation_count = operation_count + 1
        elif key==ord('1'):
            value = DRAW_FG
            messagebox.showinfo("Attention!", "Draw masks for foreground. \n Please press the key 'n' after drawing the masks and wait for a few seconds.")
            operation_count = operation_count + 1
        elif key == ord('r'):  # reset
            rect = (0, 0, 1, 1)
            drawing = False
            rectangle = False
            rect_or_mask = 100
            rect_over = False
            value = DRAW_FG
            img = img2.copy()
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            output = np.zeros(img.shape, np.uint8)
            operation_count = operation_count + 1
        elif key==ord('n'):
            if (rect_or_mask == 0):  # grabcut with rect
                rect_over = True
                GRABCUT = grabcut(img2, mask, rect)
                rect_or_mask = 1
            elif rect_or_mask == 1:  # grabcut with mask
                GRABCUT.draw_mask()
            messagebox.showinfo("Attention!","Processing complete.")
            operation_count=operation_count+1

        mask2 = np.where((mask == DRAW_FG['val']) + (mask == DRAW_PR_FG['val']), 255, 0).astype('uint8')
        output = cv.bitwise_and(img2, img2, mask=mask2)

# function of button 'Select a picture'
def IMPORT_PIC():
    global IMPORT_PIC_on_hit
    global pic_filename
    pic_filename = askopenfilename()
    if pic_filename != '':
        IMPORT_PIC_on_hit = True
        conducting_grabcut(pic_filename)
    else:
        IMPORT_PIC_on_hit = False

# function of button ''
def SAVE_GRABCUT_PIC():
    global SAVE_GRABCUT_PIC_on_hit
    global IS_THERE_A_RESULT
    global img, img2, output

    if IS_THERE_A_RESULT == True:
        bar = np.zeros((img.shape[0], 5, 3), np.uint8)
        cv.imwrite('grabcut_output.png', np.hstack((img2, bar, img, bar, output)))
        messagebox.showinfo("Attention!","Result saved as image.")
    else:
        messagebox.showinfo("Error!", "You haven't done anything.")

frame = tk.Frame(window)
frame.place(x=win_wid / 2, y=frame_len, anchor='s')

# Place button on the window
B_IMPORT_PIC = tk.Button(frame, text='Select a picture', font=('Arial', 12), width=button_len, height=1, command=IMPORT_PIC)
B_IMPORT_PIC.pack(padx=button_interval, side='left')

B_SAVE_GRABCUT_PIC = tk.Button(frame, text='Save the result', font=('Arial', 12), width=button_len, height=1, command=SAVE_GRABCUT_PIC)
B_SAVE_GRABCUT_PIC.pack(padx=button_interval, side='left')

INSTRUCTION_var.set(__doc__)
L_Instruction = tk.Label(window, textvariable=INSTRUCTION_var, justify = 'left', font=('Arial', 12), width=label_len, height=13)
L_Instruction.pack(side='bottom')

# Main window loop display
window.mainloop()