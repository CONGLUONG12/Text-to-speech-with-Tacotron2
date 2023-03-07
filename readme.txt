step 1:docker build . -t apitts
step 2:docker run -it --rm --gpus device=0 apitts:latest nvidia-smi
step 3: post vào url http://172.17.0.2:5000 json có dng:
{
    "text": "Xin chào quý khách! Xin mời vào! Quý khách đã đặt bàn chưa ạ? Quý khách đặt bàn lúc mấy giờ ạ? Quý khách có mấy người ạ? Mời quý khách đi theo tôi. Mời quý khách ngồi đây ạ! Đây là menu món ăn ạ! Bây giờ quý khách muốn gọi món chưa ạ? Quý khách dùng gì ạ? Quý khách muốn dùng đồ uống gì ạ? Quý khách có bị dị ứng gì không ạ? Quý khách vui lòng đợi một chút! Thức ăn ra rồi đây ạ"
}
