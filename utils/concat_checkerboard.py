import os
import cv2 
import numpy as np

dir_path = r'input_data/checkerboard/nonlinear_C1'
count = 0
for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        count += 1

font                    = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText     = (10,150)
topRightCornerOfText    = (3200, 150)
bottomRightCornerOfText = (1800,2000)
veryBottomCornerOfText  = (1730,2090)
fontScale1              = 5
fontScale2              = 4
fontScale3              = 3.4
fontColor1              = (0,0,255)
fontColor2              = (255,255,255)
fontColor3              = (0,255,0)
thickness               = 10
lineType                = 2

on_1 = np.load('input_Data/checkerboard_on/C1_initial.npy')
on_2 = np.load('input_Data/checkerboard_on/C2_initial.npy')
on_3 = np.load('input_Data/checkerboard_on/C3_initial.npy')
on_4 = np.load('input_Data/checkerboard_on/C4_initial.npy')
on_5 = np.load('input_Data/checkerboard_on/C5_initial.npy')
# error = np.load('output_3d/error.npy')

for i in range (1, count): 
    print ("Concating image " + str(i) + "......")

    c1 = cv2.imread('input_Data/checkerboard/nonlinear_C1/'+str(i)+'.jpg')
    c2 = cv2.imread('input_Data/checkerboard/nonlinear_C2/'+str(i)+'.jpg')
    c3 = cv2.imread('input_Data/checkerboard/nonlinear_C3/'+str(i)+'.jpg')
    c4 = cv2.imread('input_Data/checkerboard/nonlinear_C4/'+str(i)+'.jpg')
    c5 = cv2.imread('input_Data/checkerboard/nonlinear_C5/'+str(i)+'.jpg')

    cv2.putText(c1,'Camera 1', topLeftCornerOfText, font, fontScale1,fontColor1,thickness,lineType)
    cv2.putText(c2,'Camera 2', topLeftCornerOfText, font, fontScale1,fontColor1,thickness,lineType)
    cv2.putText(c3,'Camera 3', topLeftCornerOfText, font, fontScale1,fontColor1,thickness,lineType)
    cv2.putText(c4,'Camera 4', topLeftCornerOfText, font, fontScale1,fontColor1,thickness,lineType)
    cv2.putText(c5,'Camera 5', topLeftCornerOfText, font, fontScale1,fontColor1,thickness,lineType)

    if on_1[i][0] == 1:
        cv2.putText(c1,'ON', topRightCornerOfText, font, fontScale1,fontColor3,thickness,lineType)
    if on_2[i][0] == 1:
        cv2.putText(c2,'ON', topRightCornerOfText, font, fontScale1,fontColor3,thickness,lineType)
    if on_3[i][0] == 1:
        cv2.putText(c3,'ON', topRightCornerOfText, font, fontScale1,fontColor3,thickness,lineType)
    if on_4[i][0] == 1:
        cv2.putText(c4,'ON', topRightCornerOfText, font, fontScale1,fontColor3,thickness,lineType)
    if on_5[i][0] == 1:
        cv2.putText(c5,'ON', topRightCornerOfText, font, fontScale1,fontColor3,thickness,lineType)
    
    # cv2.putText(c5, 'Frame error: '+str(error[i]), bottomRightCornerOfText,font, fontScale2,fontColor2,thickness,lineType)

    # if i == count-1:
    #     cv2.putText(c5, 'Total average error: 9.001184924709639',veryBottomCornerOfText,font, fontScale3,fontColor2,thickness,lineType)

    c1_c2 = cv2.hconcat([c1,c2])
    c3_c4 = cv2.hconcat([c3,c4])
    c1_c2_c3_c4 = cv2.vconcat([c1_c2, c3_c4])
    c5 = cv2.resize(c5, dsize=(7680,4320), interpolation= cv2.INTER_CUBIC)
    c1_c2_c3_c4_c5 = cv2.hconcat([c1_c2_c3_c4,c5])
    cv2.imwrite('output_data/checkerboard_nl_concat/'+str(i)+'_concat.jpg',c1_c2_c3_c4_c5 )
