đầu tiên em chạy file run_mutiple solution
cái file này chạy các giải thuật mỗi giải thuật 15 lần
sau đó nó lưu best fit của 15 lần chạy và loss của thằng tốt nhất trong 15 lần chạy vào các các thư mục convergence và stability
sau đó em chạy 1 file get_experiment_infor
để đọc các thông tin từ các file đã lưu và chỉnh sửa vào 1 cấu trúc dữ liệu dạng dict lồng nhau
cho tiện để vẽ bảng latex
file thông tin chung này ở thư mục overall/all_algo_infor.pkl
tiếp theo em chạy file gen latex table.py để lấy thông tin từ fiel all_algo_infor và  tạo 4 bảng latex
có file plot_stab_conv.py là plot 30 cái convergenc và stability
à còn có file gen ra mấy cái bảng stability cho mlp






