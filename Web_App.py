import streamlit as st
import tensorflow as tf
import time
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from streamlit_option_menu import option_menu

# Link emoji https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Nhận diện bệnh cây cà chua", page_icon= ":tomato:",layout = 'wide')

# Navigation menu
with st.sidebar:
    selected = option_menu(
        menu_title = "",
        options = ["Home", "Guide", "Author", "Contact"],
        icons = ["house", "book", "file-earmark-person", "envelope"],
        default_index = 0,
        styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "blue", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#DDE0E6"},
            }
    )

# Load model
model = tf.keras.models.load_model("ModelTomato.h5")

# Select mode 
#----------------------------------Home---------------------------------------
if selected == "Home":
    st.title('Nhận diện bệnh cây cà chua và đưa ra biện pháp khắc phục')
    st.divider()
    uploaded_file = st.file_uploader("Chọn ảnh muốn nhận diện tại đây:", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
         # Convert file size
        imga = image.load_img(uploaded_file,target_size=(256,256))
        st.image(imga, channels="RGB")
        img = image.load_img(uploaded_file,target_size=(100,100))

        # Convert to array
        img = img_to_array(img)
        img = img.reshape(1,100,100,3)
        img = img.astype('float32')
        img = img/255
        st.divider()

        # Button detection
        Button_detect = st.button("Detect")
        st.divider()

        if Button_detect:
            with st.spinner("Please wait, Running!!!"):
                time.sleep(2)
            prediction = model.predict(img).argmax()
            y_pred = model.predict(img)
            # Phân loại bệnh
            if prediction == 1:
                st.write("**Cây cà chua của bạn bình thường**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")
                st.divider()

            elif prediction == 2:
                st.write("**Cây cà chua của bạn bị Cháy lá**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")
                st.write("**Biện pháp khắc phục:**")
                st.write("ㅤㅤ-ㅤLựa chọn hạt giống chuẩn, chất lượng rõ nguồn gốc để hạn chế sâu bệnh.")
                st.write("ㅤㅤ-ㅤVệ sinh vườn trước khi trồng vì bệnh đốm vòng có thể còn trên tàn dư thực vật hay trong đất.")
                st.write("ㅤㅤ-ㅤLuân canh cây trồng: Lựa chọn giống cây khác họ cà chua để trồng sau")
                st.write("ㅤㅤ-ㅤSử dụng chế phẩm sinh học Emina-P phun cho cây cà chua theo liều lượng 500ml chế phẩm hòa thêm 18 lít nước phun định kỳ 3-5 ngày/lần.")
                st.divider()

            elif prediction == 3:
                st.write("**Cây cà chua của bạn bị Đốm do nhện đỏ gây ra**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")
                st.write("**Biện pháp khắc phục:**")
                st.write("ㅤㅤ-ㅤÁp dụng những biện pháp canh tác và kỹ thuật chăm sóc hợp lý như: bón phân đầy đủ và cân đối, tỉa bớt cành để cây thông thoáng, giữ ẩm cho cây trong mùa khô nóng.")
                st.write("ㅤㅤ-ㅤSử dụng biện pháp sinh học tự nhiên hoặc biện pháp hóa học để khắc phục bệnh.")
                st.write("ㅤㅤ-ㅤTưới phun sương thường xuyên.")
                st.write("ㅤㅤ-ㅤDùng bột để bết dính chân nhện đỏ và làm bít lỗ thở của chúng. Tinh dầu bạc hà, dầu ăn, dầu khoáng,… là những nguyên liệu tự nhiên thường được sử dụng để tiêu diệt nhện đỏ ở quy mô nhỏ.")
                st.divider()

            elif prediction == 4:
                st.write("**Cây cà chua của bạn bị Đốm lá do vi khuẩn Septoria**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")
                st.write("**Biện pháp khắc phục:**")
                st.write("ㅤㅤ-ㅤTỉa bỏ các lá bệnh, đốt bỏ xác lá cây bệnh và tiêu hủy tàn dư cây trồng sau mỗi vụ mùa. Nấm có thể sống trong tàn dư cây trồng trong 3-4 năm. ")
                st.write('ㅤㅤ-ㅤTrồng cây đúng mật độ, tạo khoảng cách cho cây có độ thông thoáng.')
                st.write('ㅤㅤ-ㅤTránh tưới nước lên lá, nên dùng hệ thống tưới nhỏ giọt, dùng bạt phủ nông nghiệp để che phủ đất có thể hạn chế được bệnh.')
                st.write('ㅤㅤ-ㅤBón phân đầy đủ và cân đối giúp cây sinh trưởng khỏe.')
                st.write('ㅤㅤ-ㅤCác loại thuốc sau có thể dùng để kiểm soát được bệnh:')
                st.write('ㅤㅤㅤ+ㅤHoạt chất Chlorothalonil : Daconil 75WP')
                st.write('ㅤㅤㅤ+ㅤHoạt chất Azoxystrobin như Overamis 300SC.')
                st.write('ㅤㅤㅤ+ㅤThuốc Mighty 560SC phối trộn của 2 hoạt chất trên ( Azoxystrobin 60g/l + Chlorothalonil 500g/l)')
                st.write('ㅤㅤㅤ+ㅤHoạt chất Mancozeb như Dithane 80WP, Penncozeb 75DF, Manzate 75DF…')
                st.write('ㅤㅤㅤNên phun sau khi mưa và phun lặp lại sau 5-7 ngày.')
                st.write('ㅤㅤㅤChú ý : Khi sử dụng thuốc luôn luôn làm theo chỉ dẩn trên nhãn.')
                st.divider()

            elif prediction == 5:
                st.write("**Cây cà chua của bạn bị Đốm vi khuẩn (Bacterial spot)**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")
                st.write("**Biện pháp khắc phục:**")
                st.write("ㅤㅤ-ㅤSử dụng các giống cây có sức đề kháng bệnh tại địa phương.")
                st.write("ㅤㅤ-ㅤGiám sát vườn thường xuyên, đặc biệt là trong những ngày thời tiết âm u.")
                st.write("ㅤㅤ-ㅤLoại bỏ và đốt sạch bất cứ cây giống hay bộ phận nào của cây xuất hiện các đốm trên lá.")
                st.write("ㅤㅤ-ㅤoại bỏ cỏ dại trong và quanh vườn.")
                st.write("ㅤㅤ-ㅤSử dụng lớp phủ trên mặt đất để tránh cho đất nhiễm mầm bệnh từ cây.")
                st.write("ㅤㅤ-ㅤVệ sinh sạch sẽ công cụ và trang thiết bị làm vườn.")
                st.write("ㅤㅤ-ㅤKhông sử dụng hệ thống tưới phun từ trên cao và tránh làm việc trong vườn khi tán lá còn ẩm ướt.")
                st.divider()

                         
            elif prediction == 6:
                st.write("**Cây cà chua của bạn bị Đốm vòng (Early blight)**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")
                st.write("**Biện pháp khắc phục:**")
                st.write("ㅤㅤ-ㅤPhòng trừ bệnh úa sớm cà chua chủ yếu bằng biện pháp canh tác. Thực hiện chế độ luân canh trong khoảng 2-3 năm, không luân canh với cây họ Cà. Bón phân cân đối, cần chú trọng phân kali để cây sinh trưởng tốt.")
                st.write("ㅤㅤ-ㅤLưu thông khí – Cần cung cấp nhiều không gian giữa các cây, trồng đúng mật độ tạo nên tiểu khí hậu khô ráo")
                st.write("ㅤㅤ-ㅤKhi bệnh chớm xuất hiện trên đồng ruộng, dùng thuốc  sau:")
                st.write("ㅤㅤㅤㅤㅤ+ㅤHoạt chất Azoxystrobin như Amista  pha liều 25ml/ 25 lít nước")
                st.write("ㅤㅤㅤㅤㅤ+ㅤHoặc  hỗn hợp cũa Azoxystrobin và Difenconazol. như Amista top. ..")
                st.write("ㅤㅤㅤㅤㅤ+ㅤHoạt chất Chlorothalonil như Daconil…")
                st.write("ㅤㅤㅤㅤㅤ+ㅤCác thuốc có hoạt chất Copper Oxychloride.")
                st.write("ㅤㅤㅤㅤㅤ+ㅤHoạt chất Mancozeb như  Manzate…")
                st.divider()

            elif prediction == 7:
                st.write("**Cây cà chua của bạn bị Héo muộn (Late blight)**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")
                st.write("**Biện pháp khắc phục:**")
                st.write("ㅤㅤ-ㅤLoại bỏ và tiêu hủy các cây quanh khu vực cây bị nhiễm bệnh ngay lập tức và không tạo phân trộn hữu cơ từ các bộ phận của cây đã bị nhiễm bệnh.")
                st.write("ㅤㅤ-ㅤPhun các loại thuốc diệt nấm có gốc mandipropamid, chlorothalonil, fluazinam, mancozeb để đối phó với bệnh.")
                st.write("ㅤㅤ-ㅤKhông nên trồng cà chua và khoai tây cạnh nhau.")
                st.write("ㅤㅤ-ㅤGiữ cây khô ráo thông qua hệ thống thoát nước và thông gió của vườn cây.")
                st.write("ㅤㅤ-ㅤNên luân canh trong hai đến ba năm với các loài cây không phải là ký chủ của nấm.")
                st.divider()

            elif prediction == 8:
                st.write("**Cây cà chua của bạn bị Khảm lá do vi khuẩn Mosaic**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")
                st.write("**Biện pháp khắc phục:**")
                st.write("ㅤㅤ-ㅤTrồng giống kháng bệnh.")
                st.write("ㅤㅤ-ㅤBón phân đầy đủ, cân đối cho cây sinh trưởng tốt.")
                st.write("ㅤㅤ-ㅤVệ sinh tay, dụng cụ (dao, kéo) trước và sau mỗi lần cắt tỉa cành.")
                st.write("ㅤㅤ-ㅤNhổ bỏ, tiêu hủy cây bệnh.")
                st.write("ㅤㅤ-ㅤPhun thuốc trừ côn trùng chích hút.")
                st.divider()

            elif prediction == 9:
                st.write("**Cây cà chua của bạn bị Mốc lá (Leaf mold)**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")
                st.write("**Biện pháp khắc phục:**")
                st.write("ㅤㅤ-ㅤThu dọn sạch tàn dư cây bệnh, vệ sinh đồng ruộng.")
                st.write("ㅤㅤ-ㅤTrồng giống kháng bệnh.")
                st.write("ㅤㅤ-ㅤLàm giàn, cắt tỉa lá phía gốc, tăng độ thôn thoáng trong luống cà chua có tác dụng làm giảm mức độ bệnh.")
                st.write("ㅤㅤ-ㅤSử dụng các loại thuốc BVTV có hoạt chất sau để phòng trị: Azoxystrobin, Difenoconazole Validacin, Hexaconazole…")
                st.divider()

            elif prediction == 10:
                st.write("**Cây cà chua của bạn bị Vàng xoăn lá do vi khuẩn Curl**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")
                st.write("**Biện pháp khắc phục:**")
                st.write("ㅤㅤ-ㅤThường xuyên kiểm tra vườn, nhổ bỏ cây bệnh đem tiêu huỷ.")
                st.write("ㅤㅤ-ㅤLuân canh với cây trồng không phải là ký chủ của rệp, phát hiện bệnh sớm và tiêu hủy cây bệnh.")
                st.write("ㅤㅤ-ㅤVệ sinh đồng ruộng, trừ cỏ dại.")
                st.write("ㅤㅤ-ㅤDùng các loại thuốc được phép sử dụng, theo nồng độ khuyến cáo và theo nguyên tắc 4 đúng (đúng thuốc, đúng thời điểm, đúng nồng độ và liều lượng, đúng cách)")
                st.write("ㅤㅤ-ㅤPhòng trừ bọ phấn Bemissia tabaci để tiêu diệt môi giới truyền bệnh bằng các loại thuốc Applaud 10WP, Baythroid 5SL, Trebon 10EC, Pegasus 500SC, Fastac.")
                st.divider()
    
#----------------------------------Guide---------------------------------------
if selected == "Guide":
    st.title('Guide')

#----------------------------------About---------------------------------------
if selected == "Author":
    st.title('About the Author:')
    st.write('ㅤㅤㅤLưu Quang Hội')
    st.write('ㅤㅤㅤStudent ID:ㅤ20146124')
    st.write('ㅤㅤㅤMajor:ㅤMechatronics')
    st.write('ㅤㅤㅤFME, HCMC University of Technology and Education')
    st.caption('This is Project of Artificial Intelligence subject')
    st.divider()

#----------------------------------Contact---------------------------------------
if selected == "Contact":
    st.title('Contact with me:')
    st.write('ㅤㅤㅤ:label: Facebookㅤ:ㅤhttps://www.facebook.com/hoi.luuquang.5 ')
    st.write('ㅤㅤㅤ:envelope: Gmailㅤ:ㅤluuquanghoi99@gmail.com')
    st.write('ㅤㅤㅤ:telephone_receiver: Zaloㅤ:ㅤ0335427xxx  ')
    st.divider()
st.write('ㅤ')
st.write('ㅤ')
st.write('ㅤ')
st.write('ㅤ')
st.write('ㅤ')
st.write('ㅤ')
st.text('ㅤㅤㅤㅤㅤㅤCopyright © 2023 by Luu Quang Hoi | All rights reserved!')