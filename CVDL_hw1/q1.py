import cv2
import numpy as np
import glob
import re

# 設定棋盤格的格點數量（如11x8）
chessboard_size = (11, 8)

# 準備物體點，例如 (0,0,0), (1,0,0), (2,0,0) ..., (10,7,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 存儲物體點和影像點
objpoints = []  # 3D點於真實世界
imgpoints = []  # 2D點於影像平面

# 讀取棋盤格影像
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

images = sorted(glob.glob('/ssd6/Roy/CVRL_hw1/Dataset_CvDl_Hw1/Q1_Image/*.bmp'), key=natural_sort_key)  # 替換成您的影像路徑

# 印出圖片順序
print("Image loading order:")
for idx, fname in enumerate(images, start=1):
    print(f"{idx}: {fname}")


# 偵測棋盤格角點
for i, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 找到棋盤格的角點
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 如果找到角點，則新增物體點和影像點
    if ret:
        objpoints.append(objp)

        # 提升角點的精確度
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 繪製角點
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)

        # 儲存結果圖片
        output_filename = f'/ssd6/Roy/CVRL_hw1/Dataset_CvDl_Hw1/Q1_Image/output_image_{i+1}.png'
        cv2.imwrite(output_filename, img)
        print(f"儲存圖片：{output_filename}")

# 計算影像大小
h, w = gray.shape[:2]

# 計算內參矩陣、畸變係數、旋轉向量和位移向量
ret, intrinsic_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

# 顯示內參矩陣和畸變係數
print("Intrinsic Matrix:\n", intrinsic_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# 計算並顯示外參矩陣
print("Extrinsic Matrices:")
for i in range(len(rvecs)):
    # 使用Rodrigues轉換將旋轉向量轉換為旋轉矩陣
    rotation_matrix, _ = cv2.Rodrigues(rvecs[i])
    extrinsic_matrix = np.hstack((rotation_matrix, tvecs[i]))
    print(f"Extrinsic Matrix for Image {i+1}:\n", extrinsic_matrix)

# 無畸變影像
for i, fname in enumerate(images):
    img = cv2.imread(fname)
    # 使用內參矩陣和畸變係數進行無畸變處理
    undistorted_img = cv2.undistort(img, intrinsic_matrix, dist_coeffs)
    
    # 顯示並儲存無畸變影像
    output_filename = f'/ssd6/Roy/CVRL_hw1/Dataset_CvDl_Hw1/Q1_Image/undistorted_image_{i+1}.png'
    cv2.imwrite(output_filename, undistorted_img)
    print(f"Undistorted image saved: {output_filename}")
