import cv2
import numpy as np
import pandas as pd
import streamlit as st

# 設定網頁標題與排版
st.set_page_config(page_title="小學科學數字化探究 - 麵包黴菌檢測器", page_icon="🍞", layout="centered")

st.title("🍞 小學科學數字化探究：麵包黴菌生長速率自動分析系統 (專業降噪精準版)")
st.markdown("本系統已全面升級：**1. 引入 OpenCV 專業降噪算法消除雜訊與麵包屑干擾；2. 擴展黃色與黑色黴菌的捕捉範圍，解決漏檢問題。**")

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
    
    st.subheader(f"🖼️ {selected_group} - 降噪後圖像識別與視覺化標註結果")
    st.info("💡 **檢視說明：** 左側為經過專業降噪處理的原始相片，右側為 AI 識別標註圖（被標上**紅色**的區域即代表電腦視覺精準捕捉到的各類黴菌）。")
    
    for uploaded_file in uploaded_files:
        # 讀取圖片轉為 OpenCV 格式
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # --- 1. 專業降噪算法 (Denoising) ---
        # 消除相機噪點、光影斑點以及麵包表面的微小碎屑干擾
        denoised_img = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
        
        # --- 2. 轉換為 HSV 色彩空間 ---
        hsv = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2HSV)
        
        # --- 3. 擴展且精準的多色域閥值範圍 ---
        # (A) 黃色、黃綠色與綠色黴菌區間 (將 Hue 下限調至 15，捕捉與麵包皮相似但具備霉斑特徵的黃色黴菌)
        lower_yellow_green = np.array([15, 35, 35])
        upper_yellow_green = np.array([85, 255, 255])
        mask_yg = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)
        
        # (B) 黑色、深灰與深褐色黴菌區間 (擴大明度與飽和度容忍度，確保黑黴菌不漏網)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 80, 65])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        # 合併初步遮罩
        raw_mask = cv2.bitwise_or(mask_yg, mask_black)
        
        # --- 4. 形態學降噪算法 (Morphological Denoising & Filtering) ---
        # 利用開運算消除孤立雜點，閉運算填補黴菌斑塊內部空洞，讓偵測結果扎實平滑
        kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # --- 5. 視覺化紅色高亮標註 ---
        visual_img = denoised_img.copy()
        # 將處理後的乾淨遮罩處塗上鮮紅色 (BGR: 0, 0, 255)
        visual_img[clean_mask > 0] = [0, 0, 255]
        
        # 半透明疊加融合
        alpha = 0.45
        blended_img = cv2.addWeighted(visual_img, alpha, denoised_img, 1 - alpha, 0)
        
        # 轉回 RGB 格式以供 Streamlit 顯示
        img_rgb = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)
        blended_rgb = cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)
        
        # --- 6. 計算真實面積百分比 ---
        mold_pixels = np.count_nonzero(clean_mask)
        total_pixels = denoised_img.shape[0] * denoised_img.shape[1]
        
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
                st.image(img_rgb, caption=f"降噪原圖：{uploaded_file.name}", use_container_width=True)
            with col2:
                st.image(blended_rgb, caption=f"AI 精準標註 (覆蓋率: {mold_percentage}%)", use_container_width=True)
            st.markdown("---")

    # 轉換為數據表格
    df = pd.DataFrame(data)
    
    st.subheader("📊 數據統計表")
    st.dataframe(df, use_container_width=True)
    
    # 繪製生長速率折線圖
    st.subheader("📈 黴菌生長速率折線圖")
    st.line_chart(df.set_index("實驗天數/檔案")["黴菌覆蓋率 (%)"])
    
    st.success("✅ 專業降噪與多色域檢測已完成！您可以重新整理網頁查看升級後的識別效果。")
else:
    st.info("💡 請先在上方的檔案上傳區，選取並上傳該組別的實驗相片（支援多張同時選取）。")
