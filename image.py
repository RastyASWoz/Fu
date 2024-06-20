import cv2
import numpy as np
def myshow(img):
    cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('custom window', img)
    cv2.resizeWindow('custom window',1920,1080)
    cv2.waitKey()
    cv2.destroyAllWindows()

def get_curve_points(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    #img = cv2.floodFill(img, None, [0, 0], 0, 90, 255, cv2.FLOODFILL_FIXED_RANGE)[1]
    img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)[1]
    #myshow(img)

    # Apply Canny edge detection to find the contours
    edges = cv2.Canny(img, 100, 200)

    # Find the contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # sort all the contours found and use the largest one
    contours = sorted(contours, key=len)
    curve_points = np.array(contours[-1].squeeze().tolist())

    img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img2 = cv2.drawContours(img2, contours, -1, [0, 255, 255], 1)
    #myshow(img2)
    
    # 得到在边界内的中心点，对应傅里叶级数的 a_0 项，而此时即可将其消去
    centerpoint =np.array( [curve_points[:,0].mean(),curve_points[:,1].mean()])
    
    #将中心点移到原点
    curve_points =  (curve_points - centerpoint)/200
    curve_points[:,1] = -curve_points[:,1]
    return curve_points
