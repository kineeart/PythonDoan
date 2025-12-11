# Cấu trúc thư mục
1. Data: chứa các dữ liệu keypoint
2. DataImage: Dữ liệu gốc bao gồm
  nc: 9
  names: ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'L']
3. balance_classes.py: để sắp xếp các lớp đảm bảo có đủ dữ liệu ở mỗi lớp
4. export_keypoint_images.py: xuất hình ảnh có keypoint
_____________________________________________________________
# Cách chạy
1. vào file caidat.txt và cài đặt theo các bước bên trong
2. chạy trainning.ipynb để huấn luyện
3. chạy python run_inference.py để thực hiện test trên camera
_______________________________________________
# Lưu ý khi thực hiện
1. nếu tay phải không thực hiện được thì thực hiện bằng tay trái
