import cv2
import numpy as np

class GMM():
    def __init__(self,firstframe):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.model = cv2.createBackgroundSubtractorMOG2()
        fgmk = self.model.apply(firstframe)

    def update(self,frame):
         # 运用高斯模型进行拟合，在两个标准差内设置为0，在两个标准差外设置为255
        fgmk = self.model.apply(frame)

        #th = cv2.threshold(np.copy(fgmk), 254, 255, cv2.THRESH_BINARY)[1] 
        #th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2) # 图像形态学操作-腐蚀操作 cv2.erode
        #dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2) # cv2.dilate () 膨胀：将前景物体变大，理解成将图像断开裂缝变小（在图片上画上黑色印记，印记越来越小）
        # 使用形态学的开运算做背景的去除
        fgmk = cv2.morphologyEx(fgmk, cv2.MORPH_OPEN, self.kernel)
        # cv2.findContours计算fgmk的轮廓
        contours = cv2.findContours(fgmk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        lx=frame.shape[0]/2
        ly=frame.shape[1]/2
        if len(contours)==1:
            for c in contours:
                length = cv2.arcLength(c, True)
                area = cv2.contourArea(c)
                if (length > 30 and area<500):
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    lx,ly=x,y
                    cv2.imwrite(f"t.jpg",frame)
                    return [x,y,w,h]
        elif len(contours)>1:
            bboxs=[]
            for c in contours:
            # 第七步：进行人的轮廓判断，使用周长，符合条件的画出外接矩阵的方格
                length = cv2.arcLength(c, True)
                area = cv2.contourArea(c)
                if (length > 30 and area<500):
                    bboxs.append(cv2.boundingRect(c))
        
            bboxs.sort(key=lambda b:(b[0]-lx)**2+(b[1]-ly)**2)
            (x, y, w, h) = bboxs[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            lx,ly=x,y
            cv2.imwrite(f"t.jpg",frame)
            cv2.imwrite(f"tt.jpg",fgmk)
            return [x,y,w,h]
        return None
