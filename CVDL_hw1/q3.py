import cv2
import numpy as np

# 讀取左右影像
left_image = cv2.imread("/ssd6/Roy/CVRL_hw1/Dataset_CvDl_Hw1/Q3_Image/imL.png", cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread("/ssd6/Roy/CVRL_hw1/Dataset_CvDl_Hw1/Q3_Image/imR.png", cv2.IMREAD_GRAYSCALE)

# 檢查影像是否正確讀取
if left_image is None or right_image is None:
    raise ValueError("請確認左右影像的路徑是否正確。")

# 設置StereoBM參數 
stereo = cv2.StereoBM_create(numDisparities=16 * 5, blockSize=5 )  # numDisparities和blockSize可根據需求調整

# 計算視差圖
disparity = stereo.compute(left_image, right_image)

# 正規化視差圖以便於顯示
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# 或者，將視差圖儲存為圖像文件
output_filename = "/ssd6/Roy/CVRL_hw1/Dataset_CvDl_Hw1/Q3_Image/disparity_map.png"
cv2.imwrite(output_filename, disparity_normalized)
print(f"Saved disparity map: {output_filename}")
