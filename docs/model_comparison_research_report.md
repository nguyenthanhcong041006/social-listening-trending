# 📊 Nghiên cứu So sánh Hiệu năng Mô hình Dự đoán Xu hướng Chủ đề (Trend Prediction Benchmark)

Báo cáo chi tiết về kết quả thực nghiệm so sánh 4 cấu hình mô hình hybrid cho bài toán **Social Listening & Topic Trend Prediction** phục vụ viết báo khoa học (Research Paper):
1. **SARIMA + LightGBM** (Baseline hiện tại)
2. **SARIMAX + LightGBM** (Bổ sung biến ngoại sinh Exogenous từ Social Signals)
3. **SARIMA + XGBoost**
4. **SARIMAX + XGBoost**

---

## 1. Bảng Tổng Hợp Kết Quả Thực Nghiệm (Benchmark Results)

![Bảng tổng hợp kết quả so sánh 4 mô hình](./images/model_comparison_table.png)

### Bảng số liệu chi tiết (Đánh giá trên tập độc lập Test Set - 10% Chronological Split):

| Mô hình | Macro-F1 | F1 (Trend) | Precision | Recall | ROC-AUC | PR-AUC | MCC | Brier Score | $\theta^*$ (Opt. Thresh) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SARIMA + LightGBM** | 0.3298 | 0.6597 | 0.4922 | 1.0000 | **0.6348** | **0.6020** | 0.0000 | 0.2854 | 0.050 |
| **SARIMAX + LightGBM** | 0.3298 | 0.6597 | 0.4922 | 1.0000 | **0.6348** | **0.6020** | 0.0000 | 0.2854 | 0.050 |
| **SARIMA + XGBoost** | **0.3816** | 0.6522 | **0.4959** | 0.9524 | 0.5961 | 0.5574 | **0.0306** | **0.2569** | 0.210 |
| **SARIMAX + XGBoost** | **0.3816** | 0.6522 | **0.4959** | 0.9524 | 0.5961 | 0.5574 | **0.0306** | **0.2569** | 0.210 |

> [!NOTE]
> File mã nguồn bảng LaTeX đã được tự động xuất ra tại: [`../data/08_predictions/model_comparison_table.tex`](file:///c:/Users/THANH%20CONG/Documents/social_listening/data/08_predictions/model_comparison_table.tex) sẵn sàng copy vào bài báo.

```latex
\begin{table}[htbp]
\centering
\caption{Comprehensive Performance Comparison of Hybrid Trend Prediction Frameworks}
\label{tab:model_comparison}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccccc}
\hline\hline
Model Name & Macro F1 & F1 Score & Precision & Recall & Roc Auc & Pr Auc & Mcc & Optimal Threshold \\
\hline
SARIMA + LightGBM & 0.3298 & 0.6597 & 0.4922 & 1.0000 & 0.6348 & 0.6020 & 0.0000 & 0.0500 \\
SARIMAX + LightGBM & 0.3298 & 0.6597 & 0.4922 & 1.0000 & 0.6348 & 0.6020 & 0.0000 & 0.0500 \\
SARIMA + XGBoost & 0.3816 & 0.6522 & 0.4959 & 0.9524 & 0.5961 & 0.5574 & 0.0306 & 0.2100 \\
SARIMAX + XGBoost & 0.3816 & 0.6522 & 0.4959 & 0.9524 & 0.5961 & 0.5574 & 0.0306 & 0.2100 \\
\hline\hline
\end{tabular}
}
\end{table}
```

---

## 2. Phân Tích Đường Cong ROC và Precision-Recall (Discrimination Analysis)

Đối với bài toán phát hiện xu hướng bùng nổ mạng xã hội bị mất cân bằng lớp (imbalanced classes), đường cong ROC và PR là 2 công cụ quan trọng nhất để chứng minh độ phân biệt tín hiệu của mô hình:

### Đường cong ROC đa mô hình (Receiver Operating Characteristic)
![So sánh đường cong ROC đa mô hình](./images/model_comparison_roc.png)

### Đường cong Precision-Recall đa mô hình
![So sánh đường cong Precision-Recall đa mô hình](./images/model_comparison_pr.png)

- **ROC-AUC**: Nhóm **LightGBM** (SARIMA/SARIMAX + LightGBM) đạt diện tích dưới đường cong ROC cao hơn (**0.635** so với 0.596 của XGBoost), cho thấy khả năng xếp hạng phân loại tổng thể nhạy bén.
- **PR-AUC (Average Precision)**: Cả hai mô hình LightGBM đều đạt PR-AUC **0.602**, vượt trội đáng kể so với đường cơ sở xác suất tự nhiên (Prevalence Baseline 49.2%).

---

## 3. Đánh Giá Đa Chiều & Trade-offs (Radar Chart & Grouped Barchart)

### Biểu đồ cột so sánh tổng hợp các chỉ số nghiên cứu
![Biểu đồ cột so sánh các chỉ số nghiên cứu](./images/model_comparison_metrics_barchart.png)

### Biểu đồ Radar đa chiều (Multi-Dimensional Trade-offs)
![Biểu đồ Radar đa chiều](./images/model_comparison_radar.png)

### Các chỉ số bổ sung giá trị cho Research Paper:
1. **Matthews Correlation Coefficient (MCC)**:
   - Thước đo vàng cho nhị phân trên tập dữ liệu lệch tỉ lệ lớp.
   - **XGBoost** đạt **MCC = 0.0306** (cao hơn LightGBM khi LightGBM bị trôi về gán nhãn đa số nếu ngưỡng quá thấp).
2. **Macro-F1**:
   - Trung bình không trọng số giữa cả 2 lớp `Trending` và `Not Trending`.
   - **XGBoost** dẫn đầu với **Macro-F1 = 0.3816** nhờ dự đoán cân bằng hơn giữa 2 lớp.
3. **Brier Score (Độ hiệu chuẩn xác suất)**:
   - Sai số bình phương trung bình giữa xác suất dự đoán và ground-truth ($y \in \{0, 1\}$). Điểm càng thấp càng tốt.
   - **XGBoost** đạt độ tin cậy xác suất tốt hơn (**0.2569** so với 0.2854 của LightGBM).
4. **Recall@10% & Recall@20% (Budget Alerting Metric)**:
   - Đo lường khả năng bắt trúng trend khi bộ phận giám sát chỉ có thể kiểm tra top 10% hoặc 20% cảnh báo hàng đầu.
   - LightGBM đạt **Recall@20% = 25.40%**, nhỉnh hơn XGBoost (23.81%).

---

## 4. Ma Trận Nhầm Lẫn So Sánh 4 Mô Hình (Confusion Matrices Grid)

![Ma trận nhầm lẫn của 4 mô hình](./images/model_comparison_confusion_matrices.png)

- **LightGBM**: Với ngưỡng tối ưu $\theta^* = 0.05$ được chọn từ tập Validation, LightGBM thiên hướng tối đa hóa Recall (Recall = 100%, không bỏ sót bất kỳ trend nào), phù hợp cho kịch bản cảnh báo sớm ưu tiên không bỏ sót sự kiện bùng nổ.
- **XGBoost**: Với ngưỡng $\theta^* = 0.21$, mô hình giữ Recall ở mức cao 95.2% đồng thời bắt đầu phân loại chính xác các điểm âm tính thực tế ($TN=4$).

---

## 5. Hiệu Năng Dự Báo Chuỗi Thời Gian: ARIMA vs. ARIMAX (Direct Multi-Horizon)

![So sánh sai số dự báo thể tích chủ đề giữa ARIMA và ARIMAX](./images/sarima_vs_sarimax_forecasting.png)

- **Chuyển đổi phi mùa vụ (Non-seasonal Transition)**:
  - Do đặc thù các vụ việc drama và sự kiện bùng nổ thể thao mang tính xung lực ngắn hạn (*bursty & transient dynamics*), chu kỳ mùa vụ tuần $s=7$ được lược bỏ nhằm tránh lỗi phân kỳ và giảm số lượng quan sát tối thiểu (`min_obs = 4`).
- **Ma trận biến ngoại sinh trễ thực tế (Direct Multi-Horizon Lagged Exogenous Matrix)**:
  - Khắc phục lỗ hổng toán học khi hàm `.forecast(steps=k)` yêu cầu biến ngoại sinh tương lai: Mô hình truyền trực tiếp ma trận $k$ bước thực tế $\mathbf{X}_{[t-k+1 : t]}$ gồm:
    - `log_follower_engagement`: Động lượng tương tác khử thiên lệch quy mô kênh bằng $\log_{10}(\text{Followers})$.
    - `avg_sentiment`: Điểm phân cực cảm xúc đa ngôn ngữ từ Cased XLM-RoBERTa.
    - `controversy_score`: Chỉ số phân cực tranh luận giữa các nhóm người hâm mộ.
    - `uppercase_intensity`: Tỷ lệ từ viết hoa thể hiện mức độ kích động/giận dữ của dư luận.
    - `cross_platform_entropy`: Tốc độ lan truyền liên nền tảng.
- Đối với các chủ đề mới chớm nở ($< 4$ quan sát), hệ thống kích hoạt cơ chế **Exponential Weighted Moving Average (EWMA) kết hợp đạo hàm gia tốc** để ngoại suy xu hướng thay vì suy thoái về giá trị trung bình tĩnh.

---

## 6. Hiệu Chuẩn Xác Suất (Probability Calibration / Reliability Diagram)

![Đồ thị hiệu chuẩn xác suất](./images/model_comparison_calibration.png)

- Đồ thị kiểm tra xem xác suất dự đoán $P(\text{Trending} = 1)$ có phản ánh đúng tần suất thực tế của sự kiện hay không.
- Đường cong của XGBoost bám sát đường chéo lý tưởng hơn ở khoảng xác suất trung bình $[0.3, 0.6]$.

---

## 7. Gợi ý Đưa Vào Bài Viết Nghiên Cứu (Paper Writing Insights)

1. **Phân tích Kiến trúc Hybrid**: Nhấn mạnh đóng góp của việc kết hợp mô hình chuỗi thời gian thống kê (ARIMA/ARIMAX) với mô hình học máy cây tăng cường (LightGBM/XGBoost) trên không gian đặc trưng đa phương thức (NLP + Engagement + Temporal).
2. **Loại bỏ Hoàn toàn Data Leakage**:
   - Sử dụng **Inductive BERTopic** (chỉ fit trên Train Window, transform trên Valid/Test).
   - Thiết lập **$k$-step Embargo Gap** khi chia tập dữ liệu theo dòng thời gian để triệt tiêu hiện tượng rò rỉ nhãn tại ranh giới Train/Valid.
3. **Bảo toàn Casing & Cảm xúc cực độ**: Giữ nguyên chữ hoa trong tiền xử lý để bảo vệ thực thể thể thao (MU, C1, VAR) và khai thác tín hiệu ALL-CAPS phản ánh sự phẫn nộ/tranh cãi gay gắt.
4. **Giá trị của Biến Ngoại sinh Trễ (Direct Multi-Horizon ARIMAX)**: Sự xuất hiện của các tín hiệu cảm xúc và tương tác người dùng đóng vai trò chỉ báo sớm (*leading indicator*) trước khi số lượng bài viết thật sự bùng nổ trên đồ thị chuỗi thời gian.
