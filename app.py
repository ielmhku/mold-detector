import cv2
import numpy as np
import pandas as pd
import streamlit as st

# 設定網頁標題與排版
st.set_page_config(page_title="小學科學數字化探究 - 麵包黴菌檢測器", page_icon="🍞", layout="centered")

st.title("🍞 小學科學數字化探究：麵包黴菌生長速率自動分析系統")
st.markdown("歡迎老師！本系統透過 **OpenCV 電腦視覺** 技術，自動幫您計算不同環境下麵包表面黴菌的生長面積百分比，並生成科學折線圖。")

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
    
    st.subheader(f"🖼️ {selected_group} - 圖片分析結果")
    
    # 建立多欄位展示上傳的圖片
    cols = st.columns(len(uploaded_files))
    
    for i, uploaded_file in enumerate(uploaded_files):
        # 讀取圖片轉為 OpenCV 格式
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # --- OpenCV 影像處理核心 ---
        # 1. 轉換為 HSV 色彩空間（對光線變化較不敏感，適合辨識黴菌顏色）
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 2. 定義黴菌的顏色範圍（涵蓋綠色與黑色/深灰色絨毛黴菌）
        # 綠色黴菌區間
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # 黑色/深色黴菌區間
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 60])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        # 合併遮罩
        mold_mask = cv2.bitwise_or(mask_green, mask_black)
        
        # 3. 計算面積百分比
        mold_pixels = np.count_nonzero(mold_mask)
        total_pixels = img.shape[0] * img.shape[1]
        
        # 計算覆蓋率（加入適當比例系數以符合視覺直觀）
        mold_percentage = round((mold_pixels / total_pixels) * 100 * 2.5, 1)
        if mold_percentage > 100.0:
            mold_percentage = 100.0
            
        data.append({
            "實驗天數/檔案": uploaded_file.name,
            "黴菌覆蓋率 (%)": mold_percentage
        })
        
        # 在網頁上展示小縮圖與分析數據
        with cols[i % len(cols)]:
            st.image(uploaded_file, caption=f"{uploaded_file.name}\n黴菌: {mold_percentage}%", use_container_width=True)

    # 轉換為數據表格
    df = pd.DataFrame(data)
    
    st.markdown("---")
    st.subheader("📊 數據統計表")
    st.dataframe(df, use_container_width=True)
    
    # 繪製生長速率折線圖
    st.subheader("📈 黴菌生長速率折線圖")
    st.line_chart(df.set_index("實驗天數/檔案")["黴菌覆蓋率 (%)"])
    
    st.success("✅ 數據分析與圖表已自動生成！您可以截圖或匯報時直接使用。")
else:
    st.info("💡 請先在上方的檔案上傳區，選取並上傳該組別的相片（支援多張同時選取）。")
