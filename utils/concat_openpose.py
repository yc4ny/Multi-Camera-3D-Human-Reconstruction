import os
import cv2 
import numpy as np

# def concat():
dir_path = r'C1/C1_dlt'
count = 0
for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        count += 1

font                    = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText     = (10,150)
bottomRightCornerOfText = (1800,2000)
veryBottomCornerOfText  = (1730,2090)
fontScale1              = 5
fontScale2              = 3
fontColor1              = (0,0,255)
fontColor2              = (255,0,0)
fontColor3              = (0,255,0)
thickness               = 10
lineType                = 2

dlt_error = np.load('concat/dlt_error.npy')
nl_error = np.load('concat/nl_error.npy')

sum = 0
for a in range (0, dlt_error.shape[0]):
    sum += dlt_error[a]
dlt_error_average = sum/ dlt_error.shape[0]+1    

sum = 0
for a in range (0, nl_error.shape[0]):
    sum += nl_error[a]
nl_error_average = sum/ (nl_error.shape[0]+1)


for i in range (1, count): 
    print ("Concating image " + str(i) + "......")

    dlt = cv2.imread('C1/C1_dlt/'+str(i)+'_reproject.jpg')
    nl = cv2.imread('C1/C1_nonlinear/'+str(i)+'_reproject.jpg')
    dlt_1 = cv2.imread('C2/C2_dlt/'+str(i)+'_reproject.jpg')
    nl_1 = cv2.imread('C2/C2_nonlinear/'+str(i)+'_reproject.jpg')
    kp_1 = cv2.imread('3dkeypoint_viz/1/' + str(i-1) + '_output.jpg')
    kp_2 = cv2.imread('3dkeypoint_viz/2/' + str(i-1) + '_output.jpg')
    kp_3 = cv2.imread('3dkeypoint_viz/3/' + str(i-1) + '_output.jpg') 
    kp_4 = cv2.imread('3dkeypoint_viz/4/' + str(i-1) + '_output.jpg')
    kp_1 = cv2.resize(kp_1, dsize=(3840,2160), interpolation= cv2.INTER_CUBIC)
    kp_2 = cv2.resize(kp_2, dsize=(3840,2160), interpolation= cv2.INTER_CUBIC)
    kp_3 = cv2.resize(kp_3, dsize=(3840,2160), interpolation= cv2.INTER_CUBIC)
    kp_4 = cv2.resize(kp_4, dsize=(3840,2160), interpolation= cv2.INTER_CUBIC)
    cv2.putText(dlt,'Camera 1: DLT', topLeftCornerOfText, font, fontScale1,fontColor1,thickness,lineType)
    cv2.putText(nl,'Camera 1: Non Linear Optimization', topLeftCornerOfText, font, fontScale1,fontColor2,thickness,lineType)
    cv2.putText(nl_1,'Camera 2: Non Linear Optimization', topLeftCornerOfText, font, fontScale1,fontColor2,thickness,lineType)
    cv2.putText(dlt_1,'Camera 2: DLT', topLeftCornerOfText, font, fontScale1,fontColor1,thickness,lineType)
    cv2.putText(dlt_1,'Frame Reprojection Error: '+ str(dlt_error[i-1]), bottomRightCornerOfText, font, fontScale2,fontColor3,thickness,lineType)
    cv2.putText(nl_1,'Frame Reprojection Error: '+ str(nl_error[i-1]), bottomRightCornerOfText, font, fontScale2,fontColor3,thickness,lineType)

    if i == count-1:
        cv2.putText(nl_1,'Average Reprojection Error: '+ str(nl_error_average), veryBottomCornerOfText, font, fontScale2,fontColor3,thickness,lineType)
        cv2.putText(dlt_1,'Average Reprojection Error: '+ str(dlt_error_average), veryBottomCornerOfText, font, fontScale2,fontColor3,thickness,lineType)

    im_h_1 = cv2.hconcat([dlt_1,nl_1])
    im_h = cv2.hconcat([dlt, nl])
    im_v = cv2.vconcat([im_h, im_h_1]) 
    kp1 = cv2.hconcat([kp_1, kp_2])
    kp2 = cv2.hconcat([kp_3, kp_4])
    kp_c = cv2.vconcat([kp1,kp2])
    final = cv2.vconcat([im_v, kp_c])

    cv2.imwrite('concat/'+str(i)+'_concat.jpg',final)
