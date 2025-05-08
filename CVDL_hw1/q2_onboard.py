import cv2
import numpy as np
import glob

# 設定棋盤格和字母的資料庫路徑
chessboard_size = (11, 8)
database_path = "/Users/xurenlong/Desktop/CVRL_hw1/Dataset_CvDl_Hw1/Q2_Image/Q2_db/alphabet_db_onboard.txt"
images = sorted(glob.glob("/Users/xurenlong/Desktop/CVRL_hw1/Dataset_CvDl_Hw1/Q2_Image/*.bmp"))  # 替換成您的影像路徑

# 步驟 1：校正影像以獲取相機參數
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objpoints = []  # 3D 點於真實世界
imgpoints = []  # 2D 點於影像平面

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001))
        imgpoints.append(corners2)

h, w = gray.shape[:2]
ret, ins, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

# 讀取字母的 3D 座標
def load_letter_coordinates(letter, db_path):
    fs = cv2.FileStorage(db_path, cv2.FILE_STORAGE_READ)
    char_points = fs.getNode(letter).mat()
    fs.release()
    # 確保座標點的資料型別為 float32，並且轉換為 N x 3 的形狀
    char_points = np.array(char_points, dtype=np.float32).reshape(-1, 3)
    return char_points

# 設定字母佈局為兩行三列，從左至右排列
def get_word_coordinates(word):
    coordinates = []
    positions = [
        (7, 5, 0), (4, 5, 0), (1, 5, 0),  # 第一行
        (7, 2, 0), (4, 2, 0), (1, 2, 0)   # 第二行
    ]
    
    for i, letter in enumerate(word):
        if i < 6:  # 只放置前6個字母
            char_points = load_letter_coordinates(letter, database_path)
            x_offset, y_offset, z_offset = positions[i]  # 根據位置進行平移
            translated_points = char_points + np.array([x_offset, y_offset, z_offset])
            coordinates.append(translated_points)
    return coordinates

# 設置線條顏色為紅色，並加粗
def project_word_on_image(img, word, ins, dist, rvec, tvec):
    coordinates = get_word_coordinates(word)
    for char_points in coordinates:
        for i in range(0, len(char_points), 2):
            start_point = char_points[i].reshape(1, -1, 3)
            end_point = char_points[i + 1].reshape(1, -1, 3)
            projected_start, _ = cv2.projectPoints(start_point, rvec, tvec, ins, dist)
            projected_end, _ = cv2.projectPoints(end_point, rvec, tvec, ins, dist)
            start_2d = tuple(np.int32(projected_start).reshape(-1, 2)[0])
            end_2d = tuple(np.int32(projected_end).reshape(-1, 2)[0])
            # 使用紅色和粗線條
            cv2.line(img, start_2d, end_2d, (0, 0, 255), 10)

# 步驟 5：顯示結果影像
def show_result_onboard(word):
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        project_word_on_image(img, word, ins, dist, rvecs[i], tvecs[i])
        
        # 顯示影像
        cv2.imshow(f"Projected Word Image {i+1}", img)
        
        # 按下任意鍵切換到下一張圖
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 使用範例
if __name__ == "__main__":
    show_result_onboard("OPENCV")
