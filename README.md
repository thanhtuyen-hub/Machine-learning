# NHẬP MÔN HỌC MÁY - ĐỀ TÀI CUỐI KÌ
#### Họ và tên: Nguyễn Thành Tuyến - MSSV: 52000867
### 1)	Tìm hiểu, so sánh các phương pháp Optimizer trong huấn luyện mô hình học máy ?
| Loại mô hình | Mục tiêu của mô hình | Loại dữ liệu | Ưu điểm và nhược điểm |
|-------|-------|-------|-------|
| Stochastic Gradient Descent (SGD)  | Cập nhật trọng số của mô hình để giảm thiểu hàm mất mát dựa trên một điểm dữ liệu duy nhất trong mỗi lần cập nhật.  | Dữ liệu lớn, không có yêu cầu về đồng nhất, thích hợp cho bài toán phân loại và hồi quy.  | Ưu điểm: Dễ triển khai, hoạt động tốt trên dữ liệu lớn. <br> Nhược điểm: Có thể bị mắc kẹt trong các điểm cực tiểu cục bộ, không hiệu quả trên các bề mặt lỗi không đồng nhất. |
| Mini-batch Gradient Descent  | Cải thiện SGD bằng cách kết hợp Momentum và RMSProp, điều chỉnh tỷ lệ học và độ động tự động.  | Phù hợp với cả dữ liệu số và dữ liệu hạng mục.<br> Đặc biệt hiệu quả khi có thể tận dụng tính chất song song của GPU.  | Ưu điểm: Giảm thiểu tác động của nhiễu và tăng tính ổn định.Có thể triển khai trên GPU.<br> Nhược điểm: Vẫn có thể mắc kẹt ở các điểm tối thiểu cục bộ.  |
| RMSProp (Root Mean Square Propagation)  | Giảm tác động của gradient nhiễu bằng cách điều chỉnh tỷ lệ học cho mỗi tham số.  | Hiệu quả khi xử lý các vấn đề không đồng nhất về tỷ lệ học trong dữ liệu số và dữ liệu hạng mục.  | Ưu điểm: Hiệu quả trong việc xử lý các vấn đề không đồng nhất về tỷ lệ học.<br> Nhược điểm: Cần chọn tỷ lệ học thủ công.  |
| Adagrad  | Điều chỉnh tỷ lệ học cho từng tham số tùy thuộc vào lịch sử các gradient trước đó.  | Thích hợp cho dữ liệu số.<br> Tốt khi các đặc trưng có thể có độ quan trọng khác nhau.  | Ưu điểm: Tích hợp tỷ lệ học tự điều chỉnh cho mỗi trọng số.<br> Nhược điểm: Có thể dẫn đến vấn đề "đói thông tin" nếu tỷ lệ học quá nhanh.  |
| Adadelta  | Cải thiện Adagrad bằng cách sử dụng một biến độ động để giảm độ nhạy của tỷ lệ học.  | Phù hợp với dữ liệu số và có thể làm giảm vấn đề "đói thông tin" của Adagrad.  | Ưu điểm: Không yêu cầu chọn tỷ lệ học. Đối phó tốt với vấn đề "đói thông tin".<br>Nhược điểm: Yêu cầu thêm bộ nhớ để lưu trữ các trung bình bình phương của gradient.  |
| Nadam  | Kết hợp Adam với Nesterov Momentum để cải thiện khả năng hội tụ.  | Phổ quát và có thể sử dụng cho cả dữ liệu số và dữ liệu hạng mục. <br>Hiệu quả trên các mô hình học sâu.  | Ưu điểm: Kết hợp Adam với Nesterov Momentum.<br>Nhược điểm: Có thể phức tạp và đòi hỏi nhiều thử nghiệm.  |
| Rectified Adam (RAdam)  | Giảm vấn đề về độ động của Adam thông qua việc thêm một bước đầu tiên.  | Phổ quát và có thể sử dụng cho cả dữ liệu số và dữ liệu hạng mục. <br>Thường hiệu quả trên các bài toán học sâu.  | Ưu điểm: Giảm vấn đề về độ động của Adam.<br>Nhược điểm: Đôi khi không ổn định trên một số bài toán.  |
| FTRL (Follow-the-Regularized Leader)  | Tối ưu hóa mô hình trong trường hợp dữ liệu lớn và thưa thớt.  | Thích hợp cho cả dữ liệu số và dữ liệu hạng mục. <br>Đặc biệt hữu ích khi muốn kiểm soát overfitting.  | Ưu điểm: Hỗ trợ tính năng "L1" regularization. <br>Nhược điểm: Có thể yêu cầu nhiều thời gian để đặt các siêu tham số.  |
***
### 2)	Tìm hiểu về Continual Learning và Test Production khi xây dựng một giải pháp học máy để giải quyết một bài toán nào đó.
#### <pre> 2.1) Continual Learning</pre>
Continual Learning là một phương pháp trong lĩnh vực học máy nhằm giải quyết vấn đề khi mô hình phải liên tục học từ dữ liệu mới mà không quên đi kiến thức đã học từ dữ liệu cũ. Điều này quan trọng trong các ứng dụng thực tế khi dữ liệu thường xuyên thay đổi hoặc khi mô hình cần mở rộng để học được từ nhiều nguồn thông tin khác nhau.<br>
#### Ví dụ:<br>
Giả sử chúng ta đang xây dựng một ứng dụng nhận diện hình ảnh cho việc phân loại động vật. Ban đầu, tôi có một bộ dữ liệu đào tạo với các hình ảnh của mèo và chó. Sau khi triển khai mô hình, ứng dụng của tôi hoạt động tốt trên việc nhận diện mèo và chó trong điều kiện thông thường.<br>
Tuy nhiên, sau một thời gian, tôi quyết định mở rộng ứng dụng để nhận diện thêm các loài động vật khác như chim và cá. Tôi có thêm dữ liệu mới về chim và cá, và tôi muốn cập nhật mô hình mà không làm giảm hiệu suất của nó trong việc nhận diện mèo và chó.<br>
Lúc này tôi sẽ áp dụng Continual Learning bằng các phương pháp:<br>
* Lãng quên (Catastrophic Forgetting): Nếu đơn giản đào tạo lại mô hình với dữ liệu mới về chim và cá mà không giữ lại kiến thức về mèo và chó, mô hình có thể quên cách nhận diện mèo và chó một cách hiệu quả, gây ra lãng quên.<br>
* Tương tác giữa các Nhiệm vụ (Task Interference): Việc thêm dữ liệu mới có thể tác động đến trọng số của mô hình liên quan đến việc nhận diện mèo và chó, và có thể làm giảm hiệu suất trên các nhiệm vụ trước đó.<br>

Để giải quyết vấn đề này, tôi sẽ thể sử dụng các kỹ thuật Continual Learning như Regularization Techniques hoặc sử dụng Memory Replay. Memory Replay giúp mô hình "nhớ lại" một số mẫu dữ liệu cũ để giữ lại kiến thức đã học trước đó trong quá trình đào tạo với dữ liệu mới.<br>

| Ưu điểm | Nhược điểm |
|-------|-------|
| •	Duy trì Kiến Thức: Mô hình có khả năng tích luỹ và duy trì kiến thức từ các   dữ liệu mới mà nó gặp phải, giúp nó không quên đi kiến thức đã học trước đó.<br>•	Tính Linh Hoạt: Có khả năng thích ứng với sự biến đổi của dữ liệu mà không yêu cầu quá trình huấn luyện lại từ đầu.<br>•Tiết Kiệm Tài Nguyên:•	Tiết kiệm tài nguyên so với việc huấn luyện lại toàn bộ mô hình trên toàn bộ dữ liệu. | <br>•	Mô hình có thể quên đi thông tin quan trọng từ quá khứ khi được huấn luyện trên dữ liệu mới, gọi là hiện tượng "lãng phí đột ngột".<br>•	Hiệu Suất Giảm Đi: Hiệu suất trên các nhiệm vụ cũ có thể giảm đi khi mô hình thích ứng với dữ liệu mới. |
####     <pre>2.1.1) Lãng quên (Catastrophic Forgetting)</pre>
Lãng quên (Catastrophic Forgetting) là hiện tượng trong Continual Learning khi mô hình máy học quên hoặc giảm hiệu suất đối với các nhiệm vụ đã học trước đó khi nó đang học nhiệm vụ mới. Hiện tượng này là một trong những thách thức lớn nhất khi xây dựng và triển khai các hệ thống máy học có khả năng học liên tục.<br>
Có một số nội dung chính quan trọng trong lãng quên: <br>
* Mất mát Tri thức: Mô hình có thể quên đi tri thức quan trọng đã học từ nhiệm vụ trước đó, gây mất mát thông tin quan trọng và làm giảm hiệu suất trên nhiệm vụ cũ.    <br>
* Tương tác giữa các Nhiệm vụ (Task Interference): Việc học một nhiệm vụ mới có thể ảnh hưởng đến khả năng của mô hình trong việc giải quyết nhiệm vụ cũ, tạo ra tình trạng tương tác giữa các nhiệm vụ và gây ra hiện tượng lãng quên.<br>
* Phân phối Dữ liệu Thay đổi: Nếu phân phối của dữ liệu thay đổi quá nhanh, mô hình có thể không thích ứng được, và hiệu suất trên các nhiệm vụ trước đó có thể giảm đáng kể.<br>
* Trọng số Quan trọng: Các trọng số của mô hình có thể được điều chỉnh quá mức đối với nhiệm vụ mới, làm mất đi sự cân bằng và ổn định của mô hình trên các nhiệm vụ cũ.<br>
* Kỹ thuật Phòng ngừa (Mitigation Techniques): Các kỹ thuật phòng ngừa như Regularization Techniques (ví dụ: Elastic Weight Consolidation), Memory Replay, và Network Architecture Design được thiết kế để giảm thiểu hiện tượng lãng quên.  <br>
####     <pre>2.1.1) Hiệu suất giữa các Dữ liệu không cân đối (Imbalanced Data Performance)</pre>
Hiệu suất giữa các dữ liệu không cân đối (Imbalanced Data Performance) là một khía cạnh quan trọng cần được xem xét. Đây là tình trạng khi có sự chênh lệch lớn về số lượng mẫu giữa các lớp dữ liệu. Hiện tượng này có thể ảnh hưởng đến khả năng học của mô hình, đặc biệt là trong trường hợp một số lớp dữ liệu có số lượng mẫu ít. <br>
Có một số nội dung quan trọng trong hiệu suất giữa các dữ liệu không cần đối <br>
* Mất cân bằng giữa các Lớp (Class Imbalance): Một số lớp có thể có số lượng mẫu lớn hơn đáng kể so với các lớp khác, tạo ra tình trạng mất cân bằng giữa các lớp dữ liệu. <br>
* Ảnh hưởng đến Hiệu suất Trung bình: Mô hình có thể hiệu quả trên các lớp có số lượng mẫu nhiều hơn, trong khi hiệu suất trên các lớp có số lượng mẫu ít có thể giảm đáng kể. <br>
* Hiệu ứng Tăng cường (Boosting Effect): Trong mô hình học liên tục, việc tăng cường dữ liệu cho các lớp ít mẫu có thể giúp cân bằng dữ liệu và làm tăng hiệu suất trên những lớp này. <br>
* Regularization Techniques: Các kỹ thuật regularization như Class-Balanced Loss hay Focal Loss được sử dụng để làm giảm thiểu ảnh hưởng của mất cân bằng dữ liệu lên quá trình đào tạo. <br>
* Phân phối không Đồng nhất của Dữ liệu Mới: Khi có sự thay đổi trong dữ liệu mới được thêm vào, có thể xảy ra hiện tượng mất cân bằng giữa các lớp, đặc biệt là khi một số loại dữ liệu mới xuất hiện với tần suất cao hơn.<br>
#### <pre> 2.1) Test Production</pre>
Test Production là quá trình tạo ra các bộ kiểm tra (test sets) để đánh giá hiệu suất của mô hình máy học. Khi xây dựng giải pháp học máy, việc đảm bảo rằng mô hình hoạt động hiệu quả trên các tình huống mới và đa dạng là rất quan trọng.<br>
Một số cách thức trong Test Production bao gồm:
* Tạo dữ liệu kiểm tra đa dạng: Đảm bảo rằng bộ kiểm tra đủ đa dạng và phản ánh các điều kiện thực tế mà mô hình có thể gặp phải.<br>
* Kiểm tra tình huống biên (Edge Cases Testing): Đảm bảo rằng mô hình hoạt động đúng trên các tình huống đặc biệt, biên giới. <br>
* Kiểm tra đối với dữ liệu không cân đối: Đánh giá hiệu suất của mô hình trên các lớp dữ liệu với số lượng mẫu không đồng đều. <br>
* Kiểm tra Continual Learning: Đảm bảo rằng mô hình không gặp vấn đề lãng quên khi thêm dữ liệu mới. <br>

| Ưu điểm | Nhược điểm |
|-------|-------|
| •	Chất Lượng Dự Đoán: Kiểm thử trong môi trường thực tế giúp đảm bảo rằng mô hình có khả năng dự đoán chính xác trên dữ liệu mới và không gặp vấn đề không mong muốn.<br>•	Đảm Bảo An Toàn: Test Production giúp phát hiện và giải quyết các vấn đề liên quan đến an toàn và độ tin cậy của mô hình.<br>•	Tối Ưu Hóa Hiệu Suất: Cho phép tối ưu hóa hiệu suất của mô hình trong môi trường sản xuất, nơi mà yêu cầu về hiệu suất và đáng tin cậy cao. | •	Chi Phí và Thời Gian: Kiểm thử trong môi trường thực tế có thể tốn kém về chi phí và thời gian, đặc biệt là khi đối mặt với các hệ thống lớn và phức tạp.<br>•	Khó Khăn Trong Việc Tạo Các Điều Kiện Kiểm Thử Đầy Đủ: Có thể khó khăn để tạo ra tất cả các điều kiện kiểm thử cần thiết để mô phỏng mọi khía cạnh của môi trường thực tế. |
***
### 3)	Áp dụng bài toán
Bài toán: Giả sử đang xây dựng một ứng dụng dự đoán giá nhà (bài toán dự đoán giá nhà) và chúng ta muốn áp dụng Continual Learning để cập nhật mô hình của mình khi có thêm dữ liệu mới về thị trường bất động sản. Đồng thời, chúng ta muốn thực hiện Test Production để đảm bảo mô hình vẫn hoạt động đúng trên dữ liệu thực tế. <br>
| Continual Learning | Test Production |
|-------|-------|
| • Nếu chúng ta huấn luyện mô hình của mình ban đầu trên một bộ dữ liệu lớn với thông tin về giá nhà, diện tích, số phòng ngủ, vị trí, và các đặc trưng khác.<br>Triển khai Mô hình: Mô hình đã được triển khai và đang được sử dụng để dự đoán giá nhà trên thị trường.<br>• Dữ liệu mới xuất hiện: Hàng tháng, sẽ nhận được dữ liệu mới về giá nhà từ các nguồn tin khác nhau, bao gồm thông tin về các khu vực mới và các biến động mới trên thị trường.<br>• Continual Learning Update: Ap dụng Continual Learning để cập nhật mô hình của mình với dữ liệu mới. Điều này giúp mô hình không chỉ học được từ dữ liệu mới mà còn duy trì khả năng dự đoán tốt trên các khu vực và biến động mà nó đã học trước đó.<br>• Kiểm tra hiệu suất: Chúng ta kiểm tra hiệu suất của mô hình trên bộ dữ liệu kiểm tra mới để đảm bảo rằng nó vẫn dự đoán chính xác giá nhà trên các khu vực mới và đã biết trước. | • Dữ liệu Kiểm thử Thực tế: Chúng ta sử dụng một bộ dữ liệu kiểm thử thực tế, có thể được lấy từ các giao dịch bất động sản gần đây, để kiểm tra mô hình của mình trên dữ liệu mà nó chưa từng thấy.<br> • Chạy Dự đoán: Mình sẽ chạy mô hình của mình trên bộ dữ liệu kiểm thử thực tế để đánh giá hiệu suất thực tế của nó.<br>• Đánh giá Kết quả: Chúng ta sẽ so sánh kết quả dự đoán với giá nhà thực tế để đánh giá khả năng tổng quát của mô hình trên thị trường thực tế.<br>• Điều chỉnh và Cải tiến: Nếu có sai số lớn hoặc hiệu suất kém, chúng ta có thể điều chỉnh mô hình hoặc thực hiện cải tiến dựa trên phản hồi từ kết quả kiểm thử.|

**Kết luận:** Áp dụng Continual Learning giúp mô hình duy trì khả năng học liên tục từ dữ liệu mới, giữ lại thông tin quan trọng về giá nhà ở các khu vực và điều kiện thị trường khác nhau. Test Production giúp đảm bảo rằng mô hình hoạt động đúng trên dữ liệu thực tế, đồng thời cung cấp cơ hội để điều chỉnh và cải tiến mô hình dựa trên thông tin thực tế từ thị trường bất động sản. Điều này giúp duy trì và cải thiện chất lượng dự đoán của mô hình theo thời gian.
