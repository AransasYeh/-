import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# ---------------------------------------------------------
# 頁面配置
# ---------------------------------------------------------
st.set_page_config(
    page_title="台南11號 智慧沼液混灌與 AWD 決策 Agent",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 台南11號水稻 智慧沼液混灌與 AWD 決策專家系統")
st.caption("雲林縣元長鄉微型試驗場址 (0.2 分地 / 192 m²) | 環工實作智慧感測、異常預警與 AI 綜合診斷")

# ---------------------------------------------------------
# 側邊欄：環境背景與水質參數 (預設為台宇檢驗報告元民畜牧場數值)
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 場域與水質基準")
    area_m2 = st.number_input("試驗田面積 (m²)", value=192.04, step=10.0, help="約 0.198 分地")
    
    st.subheader("🧪 沼液分析水質 (Eurofins 報告)")
    slurry_tn = st.number_input("沼液總氮 TN (mg/L)", value=113.0, step=5.0)
    slurry_ec = st.number_input("沼液電導度 EC (mS/cm)", value=1.35, step=0.1)
    slurry_ph = st.number_input("沼液 pH 值", value=6.44, step=0.1)
    
    st.subheader("💧 灌溉水源 (地下水基線)")
    clean_water_ec = st.number_input("地下水 EC (mS/cm)", value=0.15, step=0.05)
    
    st.subheader("☁️ 氣象環境覆寫")
    rain_forecast = st.slider("未來 24 小時預報降雨量 (mm)", 0, 100, 0, step=5)
    ambient_temp = st.slider("日間最高水溫預估 (°C)", 20, 42, 32, step=1)

# ---------------------------------------------------------
# 模組 1：農藝物候規則防護罩 (Agronomic Rules Engine)
# ---------------------------------------------------------
def get_agronomic_rules(dat):
    if dat <= 10:
        stage = "秧苗成活期"
        trigger_h = 1.0     
        target_h = 4.0      
        n_budget = 0.0      
        allow_slurry = False
        desc = "淺水護苗期，促進開根成活，嚴禁實施深度落乾與追施沼液。"
    elif dat <= 35:
        stage = "分糵旺盛期"
        trigger_h = -5.0    
        target_h = 3.5      
        n_budget = 1.80     
        allow_slurry = True
        desc = "早生分糵關鍵期，允許落乾充氧發根；進水以淺水層搭配沼液速效銨態氮。"
    elif dat <= 45:
        stage = "最高分糵末期 (曬田期)"
        trigger_h = -7.0    
        target_h = 4.0
        n_budget = 0.0      
        allow_slurry = False
        desc = "強迫曬田斷氮！切斷無效分糵，抑制基部節間伸長以防孕穗倒伏，兼抑甲烷排放。"
    elif dat <= 70:
        stage = "幼穗分化至抽穗開花期"
        trigger_h = 0.0     
        target_h = 4.0      
        n_budget = 0.30     
        allow_slurry = True
        desc = "水分最高敏感期！禁止深度落乾，維持淺水護穎花；沼液須極度稀釋避免粗蛋白超標。"
    elif dat <= 95:
        stage = "結實成熟期 (灌漿期)"
        trigger_h = -5.0    
        target_h = 3.0
        n_budget = 0.0      
        allow_slurry = False
        desc = "輕度乾濕交替以維持根系活力，絕對禁止施用含氮沼液，否則白米食味品質崩盤。"
    else:
        stage = "收穫前排水期"
        trigger_h = -999.0
        target_h = 0.0
        n_budget = 0.0
        allow_slurry = False
        desc = "收穫前 10-14 天徹底斷水，硬化田面以利機械採收作業。"
        
    return stage, trigger_h, target_h, n_budget, allow_slurry, desc

# ---------------------------------------------------------
# 主畫面：現地感測器數據輸入
# ---------------------------------------------------------
st.subheader("📡 現地即時監測數據輸入 (S00307S0001 / FD-1081)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    dat_input = st.number_input("插秧日齡 (DAT)", value=31, min_value=1, max_value=120, help="試驗田插秧日為 9/10")
with col2:
    current_water_level = st.number_input("現地水位深度 (cm)", value=-5.0, step=0.5, help="土表為 0，土下為負，積水為正")
with col3:
    soil_ec = st.number_input("表土/田水 EC (mS/cm)", value=0.18, step=0.02, help="正常水田應介於 0.2 ~ 0.8")
with col4:
    soil_ph = st.number_input("表土/田水 pH 值", value=7.6, step=0.1, help="注意強光藻類光合作用引起的午後偏鹼")

stage_name, trigger_level, target_level, n_max_stage, slurry_allowed, stage_desc = get_agronomic_rules(dat_input)

# ---------------------------------------------------------
# 模組 2：多維異常警示系統 (Anomaly Detection & Safety Shield)
# ---------------------------------------------------------
alerts = []

# 1. EC 異常警示
if soil_ec > 1.5:
    alerts.append(("error", "🚨 【鹽分障礙 / 肥傷致命警報】表土 EC 達到 {:.2f} mS/cm（超過耐受上限 1.5），根系將發生逆滲透脫水！系統強制鎖死沼液閥門，限注入純清水洗鹽！".format(soil_ec)))
elif soil_ec > 1.0:
    alerts.append(("warning", "⚠️ 【鹽分累積警戒】表土 EC ({:.2f} mS/cm) 偏高，後續混灌必須加大稀釋倍數，防止鹽分離子於表層土壤結聚。".format(soil_ec)))
elif soil_ec < 0.15:
    alerts.append(("warning", "⚠️ 【肥力匱乏 / 淋洗過度警報】表土 EC 僅 {:.2f} mS/cm，顯示土壤可移動養分離子極度匱乏（如九月傳統深水對照組現象），基肥已被嚴重淋洗，急需在安全範圍內追肥。".format(soil_ec)))

# 2. pH 與 氨揮發 (NH3 Stripping) 異常警示
if soil_ph >= 8.0:
    alerts.append(("error", "🚨 【氨氣極速揮發爆發警戒】田水 pH 達 {:.2f} 呈強鹼性！若此時注入含高濃度銨態氮（NH4+）沼液，將引發 NH3 逸散風暴（揮發率可飆破 17%），造成嚴重空氣污染與氮素損失！禁止日間進肥！".format(soil_ph)))
elif soil_ph >= 7.5:
    alerts.append(("warning", "⚠️ 【偏鹼性氨揮發風險】田水 pH 為 {:.2f}，受微藻日間旺盛光合作用影響呈現弱鹼性。建議將灌溉排程推遲至傍晚水溫下降（17:00 後）進行。".format(soil_ph)))
elif soil_ph < 5.5:
    alerts.append(("warning", "⚠️ 【土壤酸化警戒】pH 低於 5.5，容易抑制有益微生物活性並提高重金屬活性，需評估是否暫停酸性肥源。".format(soil_ph)))

# 3. 生育期水理逆境警示
if 46 <= dat_input <= 70 and current_water_level < 0:
    alerts.append(("error", "🚨 【穎花退化 / 空包彈生理危機】水稻目前處於【幼穗分化與抽穗開花期】，對缺水極度敏感！當前水位已降至土下 {:.1f} cm，若不及時回補水層，花粉發育受阻將造成嚴重結實率崩跌！".format(current_water_level)))
elif dat_input > 70 and not slurry_allowed and n_max_stage == 0:
    alerts.append(("info", "ℹ️ 【白米食味品質保護中】水稻已邁入結實成熟期，為避免稻穀粗蛋白質超標劣化口感，系統已將沼液注入權限完全封鎖。"))

# 渲染警示看板
st.markdown("#### 🛡️ 系統安全守護與異常預警看板")
if alerts:
    for alert_type, alert_msg in alerts:
        if alert_type == "error":
            st.error(alert_msg)
        elif alert_type == "warning":
            st.warning(alert_msg)
        else:
            st.info(alert_msg)
else:
    st.success("✅ 目前所有水理與土壤化學環境（EC、pH、水位）皆處於台南11號標準生理安全區間。")

# ---------------------------------------------------------
# 模組 3：水肥優化求解器 (LP Solver)
# ---------------------------------------------------------
need_irrigation = False
if rain_forecast >= 25.0:
    irrigation_trigger_msg = "⛔ 氣象預報未來有顯著降雨 (≥25mm)，強制閉鎖抽水馬達，最大化天然降水利用並防範澇災。"
elif current_water_level <= trigger_level:
    need_irrigation = True
    irrigation_trigger_msg = f"🔔 現地水位 ({current_water_level} cm) 已觸發落乾回補下限 ({trigger_level} cm)，啟動進水排程！"
else:
    irrigation_trigger_msg = f"⏳ 目前處於 AWD 落乾充氧發根階段 (水位 {current_water_level} cm > 下限 {trigger_level} cm)，無須進水。"

st.markdown("---")
st.markdown(f"### 灌溉動作判定：{irrigation_trigger_msg}")

opt_slurry = 0.0
opt_water = 0.0
total_water_volume = 0.0
dilution_ratio = 0.0
supplied_n = 0.0
mixed_ec = clean_water_ec

if need_irrigation:
    delta_h_cm = target_level - current_water_level
    delta_h_m = delta_h_cm / 100.0
    total_water_volume = area_m2 * delta_h_m  # 噸
    
    # 線性規劃求解: Maximize V_slurry
    c = [-1.0, 0.0]
    ec_limit = 1.5
    A_ub = []
    b_ub = []
    
    # 肥分約束 (若 EC 嚴重超標或 pH 過高，強制鎖死沼液)
    if slurry_allowed and n_max_stage > 0 and soil_ec <= 1.5 and soil_ph < 8.0:
        A_ub.append([slurry_tn / 1000.0, 0.0])
        b_ub.append(n_max_stage)
    else:
        A_ub.append([1.0, 0.0])
        b_ub.append(0.0)
        
    # EC 鹽度約束
    A_ub.append([(slurry_ec - ec_limit), (clean_water_ec - ec_limit)])
    b_ub.append(0.0)
    
    # 水量守恆
    A_eq = [[1.0, 1.0]]
    b_eq = [total_water_volume]
    bounds = [(0, total_water_volume), (0, total_water_volume)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success:
        opt_slurry = res.x[0]
        opt_water = res.x[1]
        dilution_ratio = opt_water / opt_slurry if opt_slurry > 0 else 0
        supplied_n = opt_slurry * (slurry_tn / 1000.0)
        mixed_ec = (opt_slurry * slurry_ec + opt_water * clean_water_ec) / total_water_volume
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("回補總需水量", f"{total_water_volume:.2f} 噸", f"水位回升 +{delta_h_cm:.1f} cm")
        m_col2.metric("推薦沼液注入量", f"{opt_slurry:.2f} 噸", f"純氮補注 {supplied_n:.2f} kg N")
        m_col3.metric("清水配注量", f"{opt_water:.2f} 噸", f"稀釋倍數 {dilution_ratio:.1f} 倍" if opt_slurry > 0 else "純水灌溉")
        m_col4.metric("混灌終端水體 EC", f"{mixed_ec:.2f} mS/cm", "低於 1.5 安全極限")

# ---------------------------------------------------------
# 模組 4：AI 專家當前情勢綜合評估報告 (AI Situation Assessment)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🤖 AI 農業與環境工程專家情勢綜合評估")

# 組合動態 AI 評估報告文本
ai_assessment_markdown = f"""
> **【台南11號 智慧田區多維情勢診斷簡報】**
> 
> **1. 作物生育與物候生理診斷 (Physiological Diagnosis)：**
> * 當前水稻處於 **{stage_name} (DAT {dat_input}天)**。此階段台南11號的農藝核心目標為：*{stage_desc}*
> * 水理動態評估：現地水位為 **{current_water_level} cm**（本階段安全回補閾值為 **{trigger_level} cm**）。
>   - {'⚠️ 水稻正遭遇乾旱威脅，深層毛細水已無法滿足蒸散，必須立即進水！' if current_water_level < trigger_level else '✅ 目前水位處於生理安全透氣區間，向水性機制正刺激根系縱向深扎。'}
>
> **2. 土壤理化與養分動態解譯 (Soil Micro-environment)：**
> * **養分與鹽害權衡 (EC = {soil_ec} mS/cm)：** {'土壤溶液鹽分過高，存在嚴重肥傷風險，禁止盲目施肥！' if soil_ec > 1.2 else ('土壤養分極度貧乏，離子活度低，應抓緊回灌期補肥。' if soil_ec < 0.2 else '土壤導電度適中，養分離子交換容量（CEC）運作正常。')}
> * **酸鹼與氣體化學平衡 (pH = {soil_ph}, 水溫 = {ambient_temp}°C)：** 
>   - 日間高水溫結合 {'強鹼性環境，田面水游離 NH3 比例劇增，屬極高氨揮發危險期！' if soil_ph >= 7.8 else '弱酸至中性環境，銨態氮（NH4+）能穩定吸附於土壤膠體，揮發損失可控。'}
>
> **3. 溫室氣體與減碳效益審核 (Carbon & Environmental Audit)：**
> * **甲烷 (CH4) 減排：** 透過將水位壓低至 **{current_water_level} cm**，大氣氧氣進入土壤微孔隙，有效將氧化還原電位（Eh）自厭氧還原態（<-150 mV）提升至氧化態（>+100 mV），成功抑制產甲烷菌發酵活性達 50% 以上。
> * **化肥替代減碳：** 本次排程計算建議沼液消納量 **{opt_slurry:.2f} 噸**，等氮替代合成化肥，估算消除工業製程排碳量約 **{(supplied_n / 6.4) * 25.18:.2f} kg CO2e**。
>
> **4. 田間致動器執行指令建議 (Actuator Prescription)：**
> * **灌溉執行決策：** {'【立刻啟動回補】' if need_irrigation else '【維持待機落乾】'}
> * **閥門開度排程：** {'設定清水與沼液分流閥開度比為 {:.1f} : 1。'.format(dilution_ratio) if opt_slurry > 0 else '沼液閥閉鎖 (0%)，僅允許開啟清水閥。'}
> * **最佳進水窗口：** {'強烈建議預約於【傍晚 17:00 後】啟動進水，避開午後強光高溫，減少 14% 以上的氨揮發損失。' if soil_ph >= 7.5 else '環境條件良好，可依既定排程啟動水泵。'}
"""

st.markdown(ai_assessment_markdown)

# ---------------------------------------------------------
# 專案文獻與歷史數據驗證折疊區
# ---------------------------------------------------------
with st.expander("📚 點擊展開：系統決策依據之農藝文獻與歷史紀錄"):
    st.markdown("""
    * **等氮替代依據：** 依文獻最佳產量平台模型，二期作台南11號純氮推薦量為 12 kg/分地，沼液等氮替代 25%-50% 可顯著提升氮肥利用率 31.4%。
    * **AWD 水理標準：** 依農業改良場規範，砂質壤土與黏土之乾濕交替閾值控制於 -5 cm 至 -15 cm，抽穗期保持淺水 3 cm。
    * **現場歷史實測對照：**
      - `10/11 (DAT 31)`：水位 -2.5cm 補 188ppm 沼液 9.6噸 ➔ 成功建立有效分糵。
      - `10/21 (DAT 41)`：歷經 10 天零補水至 -5.0cm ➔ 曬田斷氮成功抑制節間徒長。
      - `10/28 (DAT 48)`：7 天落乾表土微裂及時回補 14.4噸 ➔ 保護幼穗分化發育。
    """)