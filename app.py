import cv2
import numpy as np
import pandas as pd
import streamlit as st

# 設定網頁標題與排版
st.set_page_config(page_title="小學科學數字化探究 - 麵包黴菌檢測器", page_icon="🍞", layout="centered")

st.title("🍞 小學科學數字化探究：麵包黴菌生長速率自動分析系統 (升級版)")
st.markdown("本系統已升級：**1. 優化 OpenCV 顏色過濾邏輯，大幅提升準確度；2. 新增『AI 視覺化紅色標註圖』，讓您清楚看見電腦視覺識別的範圍。**")

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
    
    st.subheader(f"🖼️ {selected_group} - 圖像識別與視覺化標註結果")
    st.info("💡 **檢視說明：** 下方每張相片會左右對比顯示。**左側為原始相片**，**右側為 AI 視覺化標註圖**（被標上**紅色**的區域即代表電腦視覺判定為黴菌的部分）。")
    
    for uploaded_file in uploaded_files:
        # 讀取圖片轉為 OpenCV 格式
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # --- OpenCV 影像處理核心優化 ---
        # 1. 轉換為 HSV 色彩空間
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 2. 優化後的顏色閥值（更精準捕捉綠色與深色黴菌，過濾陰影干擾）
        # 綠色黴菌區間
        lower_green = np.array([30, 80, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # 黑色/深灰色黴菌區間（收緊明度 V 上限至 50，避免把正常麵包邊緣陰影當成黴菌）
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        # 合併遮罩
        mold_mask = cv2.bitwise_or(mask_green, mask_black)
        
        # --- 新增功能：將偵測到的黴菌用顏色標註在原圖上 (紅色高亮) ---
        visual_img = img.copy()
        # 將遮罩中偵測到的黴菌處塗上鮮紅色 (BGR: 0, 0, 255)
        visual_img[mold_mask > 0] = [0, 0, 255]
        
        # 半透明疊加融合（讓原圖的質感與紅色標註完美結合，美觀且直觀）
        alpha = 0.45
        blended_img = cv2.addWeighted(visual_img, alpha, img, 1 - alpha, 0)
        
        # 將 OpenCV 的 BGR 格式轉回 Streamlit 支援的 RGB 格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blended_rgb = cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)
        
        # 3. 計算真實面積百分比（移除舊版的誇大係數，讓數值更客觀真實）
        mold_pixels = np.count_nonzero(mold_mask)
        total_pixels = img.shape[0] * img.shape[1]
        
        mold_percentage = round((mold_pixels / total_pixels) * 100, 1)
        if mold_percentage > 100.0:
            mold_percentage = 100.0
            
        data.append({
            "實驗天數/檔案": uploaded_file.name,
            "黴菌覆蓋率 (%)": mold_percentage
        })
        
        # 在網頁上左右並排展示原圖與標註圖
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_rgb, caption=f"原圖：{uploaded_file.name}", use_container_width=True)
            with col2:
                st.image(blended_rgb, caption=f"AI 識別標註圖 (覆蓋率: {mold_percentage}%)", use_container_width=True)
            st.markdown("---")

    # 轉換為數據表格
    df = pd.DataFrame(data)
    
    st.subheader("📊 數據統計表")
    st.dataframe(df, use_container_width=True)
    
    # 繪製生長速率折線圖
    st.subheader("📈 黴菌生長速率折線圖")
    st.line_chart(df.set_index("實驗天數/檔案")["黴菌覆蓋率 (%)"])
    
    st.success("✅ 數據分析與視覺化標註已完成！您可以透過右側的紅色標註圖驗證 AI 的識別準確度。")
else:
    st.info("💡 請先在上方的檔案上傳區，選取並上傳該組別的相片（支援多張同時選取）。")
