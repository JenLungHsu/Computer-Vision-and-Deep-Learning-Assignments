import cv2
import numpy as np

# 讀取兩張影像
image1 = cv2.imread("/ssd6/Roy/CVRL_hw1/Dataset_CvDl_Hw1/Q4_Image/Left.jpg")
image2 = cv2.imread("/ssd6/Roy/CVRL_hw1/Dataset_CvDl_Hw1/Q4_Image/Right.jpg")

# 檢查影像是否正確讀取
if image1 is None or image2 is None:
    raise ValueError("請確認影像的路徑是否正確。")

# 轉換為灰階影像
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 初始化 SIFT 檢測器
sift = cv2.SIFT_create()

# 4.1 在兩張影像中檢測 SIFT 特徵點和計算描述子
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# 在影像上繪製特徵點
image1_with_keypoints = cv2.drawKeypoints(gray1, keypoints1, None, color=(0, 255, 0))
image2_with_keypoints = cv2.drawKeypoints(gray2, keypoints2, None, color=(0, 255, 0))

# 或者將特徵點圖儲存為圖片文件
cv2.imwrite("/ssd6/Roy/CVRL_hw1/Dataset_CvDl_Hw1/Q4_Image/imageL_keypoints.png", image1_with_keypoints)
cv2.imwrite("/ssd6/Roy/CVRL_hw1/Dataset_CvDl_Hw1/Q4_Image/imageR_keypoints.png", image2_with_keypoints)
print("Saved images with keypoints.")

# 4.2 使用 BFMatcher 進行特徵匹配
# 使用 BFMatcher 和 knnMatch 找到匹配點
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 根據距離過濾出好的匹配點
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])  # 注意此處是 [m]，以便於使用 drawMatchesKnn

# 繪製匹配點
matched_image = cv2.drawMatchesKnn(gray1, keypoints1, gray2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 儲存匹配結果影像
cv2.imwrite("/ssd6/Roy/CVRL_hw1/Dataset_CvDl_Hw1/Q4_Image/matched_image.png", matched_image)
print("Saved matched image.")
