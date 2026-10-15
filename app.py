import cv2
import numpy as np
import pandas as pd
import streamlit as st

# 設定網頁標題與排版
st.set_page_config(page_title="小學科學數字化探究 - 麵包黴菌檢測器", page_icon="🍞", layout="centered")

st.title("🍞 小學科學數字化探究：麵包黴菌生長速率自動分析系統")
st.markdown("本版本已更新：**1. 提升黃綠色黴菌敏感度；2. 折線圖標題已動態綁定所選組別，方便多組對比！**")

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
    
    st.subheader(f"🖼️ {selected_group} - 識別與標註結果")
    st.info("💡 **檢視說明：** 左側為原圖，右側為紅色標註圖。")
    
    for uploaded_file in uploaded_files:
        # 讀取圖片轉為 OpenCV 格式
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # --- 轉換為 HSV 色彩空間 ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # --- 1. 微調後的綠色/黃綠色黴菌區間 ---
        lower_green = np.array([33, 25, 24])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # --- 2. 經典黑色黴菌區間 ---
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        # 合併候選黴菌遮罩
        raw_mold = cv2.bitwise_or(mask_green, mask_black)
        
        # --- 3. 白色麵包主體保護罩 ---
        lower_white = np.array([0, 0, 135])
        upper_white = np.array([180, 45, 255])
        mask_white_bread = cv2.inRange(hsv, lower_white, upper_white)
        
        # 從候選黴菌中「強行扣除」白色麵包區域
        cleaned_mask = cv2.subtract(raw_mold, mask_white_bread)
        
        # --- 4. 形態學清理與雜點過濾 ---
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 輪廓過濾 (去除微小背景噪點)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(cleaned_mask)
        
        total_pixels = img.shape[0] * img.shape[1]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10:  # 保留真實斑塊
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
        
        # --- 6. 計算面積百分比 (套用 × 2 放大係數，上限 100%) ---
        mold_pixels = np.count_nonzero(final_mask)
        mold_percentage = round((mold_pixels / total_pixels) * 100 * 1.5, 1)
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
                st.image(blended_rgb, caption=f"標註圖 (覆蓋率: {mold_percentage}%)", use_container_width=True)
            st.markdown("---")

    # 轉換為數據表格
    df = pd.DataFrame(data)
    
    st.subheader("📊 數據統計表")
    st.dataframe(df, use_container_width=True)
    
    # 繪製生長速率折線圖 (動態加入所選組別名稱)
    st.subheader(f"📈 黴菌生長速率折線圖 - {selected_group}")
    st.line_chart(df.set_index("實驗天數/檔案")["黴菌覆蓋率 (%)"])
    
    st.success(f"✅ 【{selected_group}】的分析與折線圖已更新完成！")
else:
    st.info("💡 請先在上方的檔案上傳區，選取並上傳該組別的實驗相片測試。")
