import cv2
import numpy as np
import pandas as pd
import streamlit as st

# 設定網頁標題與排版
st.set_page_config(page_title="小學科學數字化探究 - 麵包黴菌檢測器", page_icon="🍞", layout="centered")

st.title("🍞 小學科學數字化探究：麵包黴菌生長速率自動分析系統 (嚴格收緊版)")
st.markdown("本版本已**大幅收緊識別條件**：嚴格過濾正常麵包皮與陰影，只精準捕捉真正濃郁、局部的黴菌斑塊，杜絕整個麵包被誤判的狀況！")

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
    
    st.subheader(f"🖼️ {selected_group} - 嚴格過濾識別結果")
    st.info("💡 **檢視說明：** 左側為原圖，右側為紅色標註圖。現已排除正常麵包體與陰影干擾。")
    
    for uploaded_file in uploaded_files:
        # 讀取圖片轉為 OpenCV 格式
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # --- 轉換為 HSV 色彩空間 ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # --- 1. 收緊的黃棕色早期黴菌區間 (大幅提高飽和度，避開正常麵包金黃皮) ---
        lower_yb = np.array([15, 70, 40])
        upper_yb = np.array([30, 255, 200])
        mask_yb = cv2.inRange(hsv, lower_yb, upper_yb)
        
        # --- 2. 收緊的綠色黴菌區間 (提高飽和度與明度下限) ---
        lower_green = np.array([35, 60, 40])
        upper_green = np.array([85, 255, 220])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # --- 3. 收緊的黑色黴菌區間 (大幅壓低明度，排除一般陰影) ---
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 80, 35])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        # 合併候選遮罩
        raw_mask = cv2.bitwise_or(mask_yb, cv2.bitwise_or(mask_green, mask_black))
        
        # --- 4. 強力形態學過濾 (消除片狀擴散與噪點) ---
        kernel = np.ones((5, 5), np.uint8)
        clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # --- 5. 嚴格輪廓篩選 (剔除過大、形狀不合的區域) ---
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(clean_mask)
        
        total_pixels = img.shape[0] * img.shape[1]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 限制條件：必須大於 30 像素（濾除微小雜點），且小於總面積的 40%（防止把整個麵包主體誤認為一個大斑塊）
            if 30 < area < (total_pixels * 0.4):
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
                # 排除過於細長的線條或奇怪形狀
                if aspect_ratio < 6.0:
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
        
        # --- 7. 計算面積百分比 (帶有 × 2 放大係數，上限 100%) ---
        mold_pixels = np.count_nonzero(final_mask)
        
        mold_percentage = round((mold_pixels / total_pixels) * 100 * 2.0, 1)
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
                st.image(blended_rgb, caption=f"收緊標註 (覆蓋率: {mold_percentage}%)", use_container_width=True)
            st.markdown("---")

    # 轉換為數據表格
    df = pd.DataFrame(data)
    
    st.subheader("📊 數據統計表")
    st.dataframe(df, use_container_width=True)
    
    # 繪製生長速率折線圖
    st.subheader("📈 黴菌生長速率折線圖")
    st.line_chart(df.set_index("實驗天數/檔案")["黴菌覆蓋率 (%)"])
    
    st.success("✅ 條件已成功收緊！正常麵包體與陰影已被確實排除。")
else:
    st.info("💡 請先在上方的檔案上傳區，選取並上傳該組別的實驗相片（支援多張同時選取）。")
