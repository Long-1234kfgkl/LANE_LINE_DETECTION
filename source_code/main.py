import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd 
import os

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2]  
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image, mask

def get_vertices(image):
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0, rows]
    top_left     = [cols*0, rows*0.6]
    bottom_right = [cols*1, rows]
    top_right    = [cols*1, rows*0.6] 
    
    ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return ver

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def limit_theta(line1s):
  LINE_NEW = []
  for i in range(len(line1s)):
    TAM_1 = line1s[i][0][0] - line1s[i][0][2]
    TAM_2 = line1s[i][0][1] - line1s[i][0][3]

    if TAM_1 ==0 or TAM_2 == 0:
      continue
    m = (line1s[i][0][3] - line1s[i][0][1])/(line1s[i][0][2] - line1s[i][0][0])
    THETA = np.degrees(np.arctan(-TAM_1 / TAM_2))
    if THETA < 0:
      THETA = 180 + THETA
    if m <0:
      if THETA >=15 and THETA <=75:
        LINE_NEW.append(line1s[i])
    else:
      if THETA >=105 and THETA <=175:
        LINE_NEW.append(line1s[i])
  LINE_NEW = np.array(LINE_NEW)
  return LINE_NEW

def find_left_right_line(LINE_NEW, img):
  try: 
    left_lines = []
    LINES_1 = []
    right_lines = []
    LINES_2 = []
    rows, cols = img.shape[:2]
    for line in LINE_NEW:
        for x1,y1,x2,y2 in line:
            if x1 == x2:
                continue 
            else:
                m = (y2 - y1) / (x2 - x1)

                if m < 0:
                    # if x2 <= int(cols/2) and x1 <= int(cols/2):
                    if x1<= int(cols/2) and x1<x2 and y1 >y2:
                      c = y2 - m * x2
                      y1 = int(rows*0.6)
                      y2 = rows
                      if m == 0:
                          continue
                      x2 = int((y2 - c)/m)
                      x1 = int((y1 - c)/m)
                      if x1 >= int(cols/2):
                        x1 = int(x1 - (x1 - (cols/2)))  
                        y1 = int(m*x1+c)
                      LEFT_TD = int((x2+x1)/2)
                      left_lines.append(LEFT_TD)
                      LINES_1.append([[x1,y1,x2,y2]])
                elif m >= 0:
                    # if x1 >= int(cols/2) and x2 >= int(cols/2):
                    if x2 >=int(cols/2) and x1<x2 and y1<y2:
                      c = y1 - m * x1
                      y2 = rows
                      y1 = int(rows*0.6)
                      if m==0: 
                          continue
                      x2 = int((y2 - c)/m)
                      x1 = int((y1 - c)/m)
                      if x1 <= int(cols/2):
                        x1 = int(x1 + ((cols/2) - x1))
                        y1 = int(m*x1+c)
                      RIGHT_TD = int((x2+x1)/2)
                      right_lines.append(RIGHT_TD)
                      LINES_2.append([[x1,y1,x2,y2]])
    tam1 = np.argmax(left_lines)
    LEFT_LINE_NEW = LINES_1[tam1].copy()
    tam2 = np.argmin(right_lines)
    RIGHT_LINE_NEW = LINES_2[tam2].copy()

    LEFT_RIGHT_LINE = np.array([LEFT_LINE_NEW] + [RIGHT_LINE_NEW])
    return LEFT_RIGHT_LINE
  except:
    return

def slope_line(LEFT_RIGHT_LINE, img):
  try:
    LEFT_RIGHT_LINE_NEW = []
    for x1,y1,x2,y2 in LEFT_RIGHT_LINE[0]:
      LEFT_RIGHT_LINE_NEW.append((x1,y1))
      LEFT_RIGHT_LINE_NEW.append((x2,y2))
    for x1,y1,x2,y2 in LEFT_RIGHT_LINE[1]:
      LEFT_RIGHT_LINE_NEW.append((x2,y2))
      LEFT_RIGHT_LINE_NEW.append((x1,y1))
    MASK = np.zeros_like(img)
    cv2.fillPoly(MASK, pts = np.array([LEFT_RIGHT_LINE_NEW],'int32'), color = (0,255,0))
    output_img = cv2.addWeighted(img,1.,MASK,0.4,0.)
    return output_img
  except:
    return img

def test(img, canny_img):
  ysize = img.shape[0]
  xsize = img.shape[1]
  color_select = np.copy(img)

  # Define color selection criteria
  ###### MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
  red_threshold = 200
  green_threshold = 200
  blue_threshold = 200
  ######

  rgb_threshold = [red_threshold, green_threshold, blue_threshold]
  thresholds = (img[:,:,0] < rgb_threshold[0]) \
            | (img[:,:,1] < rgb_threshold[1]) \
            | (img[:,:,2] < rgb_threshold[2])
  color_select[thresholds] = [0,0,0]


  color_select = grayscale(color_select)
  canny_img_new = cv2.bitwise_or(canny_img, color_select)
  return canny_img_new

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
def lane_finding_pipeline(img):
    gray_img = grayscale(img)
    gaussian_img = gaussian_blur(img = gray_img, kernel_size = 5)
    canny_img = canny(img = gaussian_img, low_threshold =50, high_threshold = 100)
    canny_img_new = test(img = img, canny_img = canny_img)
    masked_img, mask = region_of_interest(img = canny_img_new, vertices = get_vertices(img))
    line1s = hough_lines(masked_img, 1, np.pi/180, 30, 30, 30)
    LINE_NEW = limit_theta(line1s)
    LEFT_RIGHT_LINE = find_left_right_line(LINE_NEW, img)
    output_img = slope_line(LEFT_RIGHT_LINE, img)
    return output_img
