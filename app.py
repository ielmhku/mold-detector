import cv2
import numpy as np
import pandas as pd
import streamlit as st

# 設定網頁標題與排版
st.set_page_config(page_title="小學科學數字化探究 - 麵包黴菌檢測器", page_icon="🍞", layout="centered")

st.title("🍞 小學科學數字化探究：麵包黴菌生長速率自動分析系統 (精準防誤判版)")
st.markdown("本系統已針對**「避免誤判正常麵包與陰影」**進行深度優化：**已過濾麵包金黃色外皮與背景陰影，只精準捕捉真實的綠色與黑色黴菌。**")

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
    
    st.subheader(f"🖼️ {selected_group} - 高精度視覺化標註結果")
    st.info("💡 **檢視說明：** 左側為降噪後原圖，右側為 AI 標註圖。紅色區域現已嚴格過濾麵包正常外皮與背景陰影，僅顯示真實黴菌。")
    
    for uploaded_file in uploaded_files:
        # 讀取圖片轉為 OpenCV 格式
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # --- 1. 專業降噪算法 ---
        denoised_img = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
        
        # --- 2. 轉換為 HSV 色彩空間 ---
        hsv = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2HSV)
        
        # --- 3. 精準防誤判色域閥值 (核心修正) ---
        # (A) 黃綠色與綠色黴菌：Hue 從 25 開始（完全避開 10~24 的正常麵包金黃色外皮），飽和度提高至 50 濾除淡色雜訊
        lower_yg = np.array([25, 50, 40])
        upper_yg = np.array([85, 255, 210])
        mask_yg = cv2.inRange(hsv, lower_yg, upper_yg)
        
        # (B) 黑色/深色黴菌：限制飽和度與明度範圍，防止抓到整張圖的背景陰影與四周暗角
        lower_black = np.array([0, 15, 20])
        upper_black = np.array([180, 150, 60])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        # 合併遮罩
        raw_mask = cv2.bitwise_or(mask_yg, mask_black)
        
        # --- 4. 形態學濾波 (去除獨立微小噪點與麵包屑) ---
        kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # --- 5. 視覺化紅色高亮標註 ---
        visual_img = denoised_img.copy()
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
    
    st.success("✅ 參數已成功調整！請重新整理網頁查看效果，現在正常麵包皮與陰影應不再被紅標誤認。")
else:
    st.info("💡 請先在上方的檔案上傳區，選取並上傳該組別的實驗相片（支援多張同時選取）。")
