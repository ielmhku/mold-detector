import cv2
import numpy as np
import pandas as pd
import streamlit as st

# 設定網頁標題與排版
st.set_page_config(page_title="小學科學數字化探究 - 麵包黴菌檢測器", page_icon="🍞", layout="centered")

st.title("🍞 小學科學數字化探究：麵包黴菌生長速率自動分析系統 (高清保邊銳利版)")
st.markdown("本系統已全面優化：**完全移除模糊感，改用『雙邊濾波（Bilateral Filter）』保邊去噪技術，確保原圖與標註線條保持極致清晰、不失真！**")

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
    
    st.subheader(f"🖼️ {selected_group} - 高清銳利視覺化標註結果")
    st.info("💡 **檢視說明：** 左側為原汁原味的高清原圖（保持完美清晰度），右側為 AI 標註圖。紅色區域精準對應黴菌邊界，絕不模糊。")
    
    for uploaded_file in uploaded_files:
        # 讀取圖片轉為 OpenCV 格式
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # --- 1. 使用保邊去噪 (Bilateral Filter) 替代模糊的降噪 ---
        # 既能過濾微小雜點，又能完美保留麵包與黴菌的清晰邊界（絕對不糊、不失真）
        filtered_img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        
        # --- 2. 轉換為 HSV 色彩空間 ---
        hsv = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2HSV)
        
        # --- 3. 精準防誤判色域閥值 (避開正常金黃麵包皮與陰影) ---
        lower_yg = np.array([25, 50, 40])
        upper_yg = np.array([85, 255, 210])
        mask_yg = cv2.inRange(hsv, lower_yg, upper_yg)
        
        lower_black = np.array([0, 15, 20])
        upper_black = np.array([180, 150, 60])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        # 合併遮罩
        raw_mask = cv2.bitwise_or(mask_yg, mask_black)
        
        # --- 4. 輕量形態學濾波 (保持邊緣銳利度) ---
        kernel = np.ones((2, 2), np.uint8)
        clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # --- 5. 視覺化紅色高亮標註 (直接基於高清原圖疊加) ---
        visual_img = img.copy()
        visual_img[clean_mask > 0] = [0, 0, 255]
        
        # 半透明疊加融合
        alpha = 0.45
        blended_img = cv2.addWeighted(visual_img, alpha, img, 1 - alpha, 0)
        
        # 轉回 RGB 格式以供 Streamlit 顯示
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blended_rgb = cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)
        
        # --- 6. 計算真實面積百分比 ---
        mold_pixels = np.count_nonzero(clean_mask)
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
                st.image(img_rgb, caption=f"高清原圖：{uploaded_file.name}", use_container_width=True)
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
    
    st.success("✅ 系統已更新為「高清保邊模式」！圖片不再模糊，邊界清晰銳利，數據也更精準。")
else:
    st.info("💡 請先在上方的檔案上傳區，選取並上傳該組別的實驗相片（支援多張同時選取）。")
