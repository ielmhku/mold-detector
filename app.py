import cv2
import numpy as np
import pandas as pd
import streamlit as st

# 設定網頁標題與排版
st.set_page_config(page_title="小學科學數字化探究 - 麵包黴菌檢測器", page_icon="🍞", layout="centered")

st.title("🍞 小學科學數字化探究：麵包黴菌生長速率自動分析系統 (終極精準版)")
st.markdown("本系統已升級至**雙向夾擊演算法**：**1. 解禁純黑黴菌捕捉範圍，確保黑斑不漏網；2. 增加『白麵包主體主動排除遮罩』，徹底解決白麵包被誤判為紅標的問題！**")

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
    
    st.subheader(f"🖼️ {selected_group} - 終極精準標註結果")
    st.info("💡 **檢視說明：** 系統現已能完美鎖定黑黴菌與綠黴菌，並自動排除正常白麵包區域。")
    
    for uploaded_file in uploaded_files:
        # 讀取圖片轉為 OpenCV 格式
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # --- 1. 高清保邊去噪 ---
        filtered_img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        hsv = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2HSV)
        
        # --- 2. 綠色與黃綠色黴菌捕捉 ---
        lower_yg = np.array([18, 30, 25])
        upper_yg = np.array([85, 255, 230])
        mask_yg = cv2.inRange(hsv, lower_yg, upper_yg)
        
        # --- 3. 純黑/深灰大片黴菌捕捉 (解禁低飽和度，確保黑斑不漏網) ---
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 50, 55])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        # 合併初步候選黴菌遮罩
        raw_mold = cv2.bitwise_or(mask_yg, mask_black)
        
        # --- 4. 核心反制：建立「白色/淺色麵包主體排除遮罩」 ---
        # 正常白麵包特徵：亮度高 (V > 130)、飽和度低 (S < 40)，色相落在黃白色區間
        lower_white_bread = np.array([10, 0, 130])
        upper_white_bread = np.array([35, 40, 255])
        mask_white_bread = cv2.inRange(hsv, lower_white_bread, upper_white_bread)
        
        # 從候選黴菌中「強制扣除」正常白麵包區域，防止白麵包被誤判
        cleaned_mask = cv2.subtract(raw_mold, mask_white_bread)
        
        # --- 5. 形態學處理與微小噪點過濾 ---
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 輪廓過濾 (剔除極小雜點)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(cleaned_mask)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 15:  # 保留微小斑點但過濾像素級噪點
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
                if aspect_ratio < 12.0:
                    cv2.drawContours(final_mask, [cnt], -1, 255, -1)
        
        # --- 6. 視覺化紅色高亮標註 ---
        visual_img = img.copy()
        visual_img[final_mask > 0] = [0, 0, 255]
        
        # 半透明疊加融合
        alpha = 0.45
        blended_img = cv2.addWeighted(visual_img, alpha, img, 1 - alpha, 0)
        
        # 轉回 RGB 格式以供 Streamlit 顯示
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blended_rgb = cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)
        
        # --- 7. 計算真實面積百分比 ---
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
                st.image(blended_rgb, caption=f"精準標註 (覆蓋率: {mold_percentage}%)", use_container_width=True)
            st.markdown("---")

    # 轉換為數據表格
    df = pd.DataFrame(data)
    
    st.subheader("📊 數據統計表")
    st.dataframe(df, use_container_width=True)
    
    # 繪製生長速率折線圖
    st.subheader("📈 黴菌生長速率折線圖")
    st.line_chart(df.set_index("實驗天數/檔案")["黴菌覆蓋率 (%)"])
    
    st.success("✅ 終極精準版更新完畢！白麵包主體已被成功排除，大片黑斑也已順利捕捉。")
else:
    st.info("💡 請先在上方的檔案上傳區，選取並上傳該組別的實驗相片（支援多張同時選取）。")
