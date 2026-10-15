import cv2
import numpy as np
import pandas as pd
import streamlit as st

# 設定網頁標題與排版
st.set_page_config(page_title="小學科學數字化探究 - 麵包黴菌檢測器", page_icon="🍞", layout="centered")

st.title("🍞 小學科學數字化探究：麵包黴菌生長速率自動分析系統 (智慧形狀過濾版)")
st.markdown("本系統已升級至**智慧過濾架構**：**1. 完美捕捉大片黑黴菌與黃綠色黴菌；2. 自動識別並剃除白色麵包表面的細長陰影與雜點，杜絕誤判！**")

# 選擇實驗組別
groups = [
    "A組：乾燥環境 (室溫)", 
    "B組：濕潤環境 (室溫 25°C)", 
    "C組：濕潤環境 (雪櫃 4°C)", 
    "D組：濕潤環境 (溫暖處 35°C)"
]
selected_group = st.selectbox("📌 請選擇您要分析的實驗組別：", groups)

st.divider()

# 檔案上傳區
uploaded_files = st.file_uploader(
    f"📤 上傳【{selected_group}】從第1天到第5天的實驗相片（可多選）", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    data = []
    
    # 依照檔案名稱排序（確保天數順序正確）
    uploaded_files = sorted(uploaded_files, key=lambda x: x.name)
    
    st.subheader(f"🖼️ {selected_group} - 智慧過濾標註結果")
    st.info("💡 **檢視說明：** 系統現已能精準鎖定塊狀黴菌，並自動排除麵包縫隙陰影與背景雜訊。")
    
    for uploaded_file in uploaded_files:
        # 讀取圖片轉為 OpenCV 格式
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # --- 1. 高清保邊去噪 (維持邊緣清晰) ---
        filtered_img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        hsv = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2HSV)
        
        # --- 2. 寬頻多色域閥值 (確保黃綠色、亮綠色與大片黑灰色黴菌全數涵蓋) ---
        # (A) 黃綠色與綠色黴菌
        lower_yg = np.array([15, 25, 20])
        upper_yg = np.array([85, 255, 240])
        mask_yg = cv2.inRange(hsv, lower_yg, upper_yg)
        
        # (B) 黑色/深灰大片黴菌 (適度放寬明度與飽和度，捕捉深色黴菌)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 100, 80])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        raw_mask = cv2.bitwise_or(mask_yg, mask_black)
        
        # --- 3. 形態學處理：閉運算（把分散的黴菌斑塊融合成扎實區塊） ---
        kernel_close = np.ones((5, 5), np.uint8)
        closed_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # --- 4. 智慧輪廓與形狀過濾 (核心：剔除細長陰影與極小雜點) ---
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(closed_mask)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 過濾掉太小的雜點（例如麵包屑，面積小於 30 像素）
            if area > 30:
                # 計算長寬比，排除過於細長的線條（如麵包切片縫隙、袋子摺痕陰影）
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
                
                # 若長寬比小於 8（代表是扎實的塊狀，而非細長陰影），則保留為真實黴菌
                if aspect_ratio < 8.0:
                    cv2.drawContours(final_mask, [cnt], -1, 255, -1)
        
        # --- 5. 視覺化紅色高亮標註 ---
        visual_img = img.copy()
        visual_img[final_mask > 0] = [0, 0, 255]
        
        # 半透明疊加融合
        alpha = 0.45
        blended_img = cv2.addWeighted(visual_img, alpha, img, 1 - alpha, 0)
        
        # 轉回 RGB 格式以供 Streamlit 顯示
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blended_rgb = cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)
        
        # --- 6. 計算真實面積百分比 ---
        mold_pixels = np.count_nonzero(final_mask)
        total_pixels = img.shape[0] * img.shape[1]
        
        mold_percentage = round((mold_pixels / total_pixels) * 100, 1)
        if mold_percentage > 100.0:
            mold_percentage = 100.0
            
        data.append({
            "實驗天數/檔案": uploaded_file.name,
            "黴菌覆蓋率 (%)": mold_percentage
        })
        
        # 在網頁上左右並排展示
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_rgb, caption=f"原圖：{uploaded_file.name}", use_container_width=True)
            with col2:
                st.image(blended_rgb, caption=f"智慧過濾標註 (覆蓋率: {mold_percentage}%)", use_container_width=True)
            st.markdown("---")

    # 轉換為數據表格
    df = pd.DataFrame(data)
    
    st.subheader("📊 數據統計表")
    st.dataframe(df, use_container_width=True)
    
    # 繪製生長速率折線圖
    st.subheader("📈 黴菌生長速率折線圖")
    st.line_chart(df.set_index("實驗天數/檔案")["黴菌覆蓋率 (%)"])
    
    st.success("✅ 智慧形狀過濾升級完畢！細長陰影已被自動剃除，大片黑黴菌與塊狀黴菌將能完美呈現。")
else:
    st.info("💡 請先在上方的檔案上傳區，選取並上傳該組別的實驗相片（支援多張同時選取）。")
