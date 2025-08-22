import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from folium.features import DivIcon
from shapely.geometry import Point
import osmnx as ox
import requests
from streamlit_folium import st_folium
import math
import os
import datetime
import glob
from pathlib import Path

# ────────────────────────────── 
# ✅ 환경변수 불러오기 (Streamlit Cloud 호환에 저장된 키 사용)
# ──────────────────────────────
MAPBOX_TOKEN = "pk.eyJ1IjoiZ3VyMDUxMDgiLCJhIjoiY21lZ2k1Y291MTdoZjJrb2k3bHc3cTJrbSJ9.DElgSQ0rPoRk1eEacPI8uQ"

# ──────────────────────────────
# ✅ 1. 대상 shp 파일 탐색
# ──────────────────────────────
patterns = ["./drt_*.shp", "./new_drt.shp"]
shp_files = []
for p in patterns:
    shp_files.extend(glob.glob(p))
shp_files = sorted(set(shp_files))

if not shp_files:
    raise FileNotFoundError("❌ drt_*.shp / new_new_drt.shp 파일을 찾지 못했습니다.")

# ──────────────────────────────
# ✅ 2. 파일별 읽기 + source_file 컬럼 추가
# ──────────────────────────────
gdfs = []
for f in shp_files:
    print("불러오는 중:", f)
    _g = gpd.read_file(f, encoding="euc-kr")
    _g["source_file"] = Path(f).stem
    gdfs.append(_g)

# ──────────────────────────────
# ✅ 3. concat으로 병합
# ──────────────────────────────
gdf = pd.concat(gdfs, ignore_index=True)
gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdfs[0].crs)

# ──────────────────────────────
# ✅ 4. 좌표계 EPSG:4326으로 맞추기
# ──────────────────────────────
if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)

# ──────────────────────────────
# ✅ 5. name 컬럼 보정
# ──────────────────────────────
name_candidates = [c for c in gdf.columns if c.lower() in ["name", "정류장명", "stop_name", "title"]]
if name_candidates:
    name_col = name_candidates[0]
    gdf["name"] = gdf[name_col].astype(str)
else:
    obj_cols = [c for c in gdf.columns if c != "geometry" and gdf[c].dtype == "object"]
    if obj_cols:
        gdf["name"] = gdf[obj_cols[0]].astype(str)
    else:
        gdf["name"] = gdf.apply(lambda r: f"{r.get('source_file','drt')}_{int(r.name)+1}", axis=1)

# ──────────────────────────────
# ✅ 6. 위도/경도 컬럼 생성
# ──────────────────────────────
if gdf.geometry.geom_type.isin(["Point"]).all():
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
else:
    reps = gdf.geometry.representative_point()
    gdf["lon"] = reps.x
    gdf["lat"] = reps.y

# ──────────────────────────────
# ✅ 7. boundary 생성
# ──────────────────────────────
boundary_path = Path("./cb_shp.shp")
if boundary_path.exists():
    boundary = gpd.read_file(boundary_path, encoding="euc-kr").to_crs(epsg=4326)
else:
    try:
        union = gdf.unary_union
        hull = union.convex_hull
        boundary = gpd.GeoDataFrame(geometry=[hull], crs="EPSG:4326")
    except Exception:
        boundary = None

# ──────────────────────────────
# ✅ Session 초기화
# ──────────────────────────────
DEFAULTS = {
    "order": [],
    "segments": [],
    "duration": 0.0,
    "distance": 0.0,
    "messages": [
        {
            "role": "system",
            "content": "당신은 천안시에서 DRT(수요응답형 교통) 최적 노선을 추천해 주는 전문 교통 어시스턴트입니다."
        }
    ],
    "auto_gpt_input": ""
}

# ──────────────────────────────
# ✅ 혼잡도 보정 함수 (차량만 영향, 도보=1.0)
# ──────────────────────────────
def congestion_factor(time_band: str, mode_text: str) -> float:
    if "도보" in mode_text:
        return 1.0
    table = {
        "혼잡(출퇴근)": 1.40,
        "일반":       1.15,
        "심야/한산":   0.90,
    }
    return table.get(time_band, 1.15)

# ──────────────────────────────
# ✅ 페이지 설정 & 스타일
# ──────────────────────────────
st.set_page_config(
    page_title="청풍로드 - 충청북도 맞춤형 AI기반 스마트 관광 가이드",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif; }
.main > div { padding-top: 1.2rem; padding-bottom: 0.5rem; }
header[data-testid="stHeader"] { display: none; }
.stApp { background: #f8f9fa; }
.header-container { display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 2rem; padding: 1rem 0; }
.logo-image { width: 50px; height: 50px; object-fit: contain; }
.main-title { font-size: 2.8rem; font-weight: 700; color: #202124; letter-spacing: -1px; margin: 0; }
.title-underline { width: 100%; height: 3px; background: linear-gradient(90deg, #4285f4, #34a853); margin: 0 auto 2rem auto; border-radius: 2px; }
.section-header { font-size: 1.3rem; font-weight: 700; color: #1f2937; margin-bottom: 20px; display: flex; align-items: center; gap: 8px; padding-bottom: 12px; border-bottom: 2px solid #f3f4f6; }
.stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 10px; padding: 12px 20px; font-size: 0.9rem; font-weight: 600; width: 100%; height: 48px; transition: all 0.3s ease; box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3); }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4); }
.visit-order-item { display: flex; align-items: center; padding: 12px 16px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; margin-bottom: 8px; font-size: 0.95rem; font-weight: 500; transition: all 0.2s ease; box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3); }
.visit-order-item:hover { transform: translateX(4px); box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4); }
.visit-number { background: rgba(255,255,255,0.9); color: #667eea; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 700; margin-right: 12px; flex-shrink: 0; }
.stMetric { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); border: none; border-radius: 12px; padding: 16px 10px; text-align: center; transition: all 0.2s ease; box-shadow: 0 2px 4px rgba(168, 237, 234, 0.3); }
.stMetric:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(168, 237, 234, 0.4); }
.empty-state { text-align: center; padding: 40px 20px; color: #9ca3af; font-style: italic; font-size: 0.95rem; background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); border-radius: 12px; margin: 16px 0; }
.map-container { width: 100% !important; height: 520px !important; border-radius: 12px !important; overflow: hidden !important; position: relative !important; background: transparent !important; border: 2px solid #e5e7eb !important; margin: 0 !important; padding: 0 !important; box-sizing: border-box !important; }
div[data-testid="stIFrame"] { width: 100% !important; max-width: 100% !important; height: 520px !important; position: relative !important; overflow: hidden !important; box-sizing: border-box !important; border-radius: 12px !important; background: transparent !important; border: none !important; margin: 0 !important; padding: 0 !important; }
div[data-testid="stIFrame"] > iframe { width: 100% !important; height: 100% !important; border: none !important; border-radius: 12px !important; max-width: 100% !important; box-sizing: border-box !important; background: transparent !important; margin: 0 !important; padding: 0 !important; }
div[data-testid="stIFrame"] > iframe > html > body > div:empty { display: none !important; }
div[data-testid="stIFrame"] div:empty { display: none !important; }
.folium-map div:empty { display: none !important; }
.leaflet-container .leaflet-control-container div:empty { display: none !important; }
.leaflet-container > div:empty { display: none !important; }
div:empty:not(.leaflet-zoom-box):not(.leaflet-marker-icon):not(.leaflet-div-icon) { display: none !important; }
div[style*="background: white"]:empty, div[style*="background: #fff"]:empty, div[style*="background: #ffffff"]:empty, div[style*="background-color: white"]:empty, div[style*="background-color: #fff"]:empty, div[style*="background-color: #ffffff"]:empty { display: none !important; }
.folium-map { width: 100% !important; height: 100% !important; max-width: 100% !important; max-height: 520px !important; box-sizing: border-box !important; background: transparent !important; margin: 0 !important; padding: 0 !important; border: none !important; }
.leaflet-container { width: 100% !important; height: 100% !important; max-width: 100% !important; max-height: 520px !important; box-sizing: border-box !important; background: transparent !important; margin: 0 !important; padding: 0 !important; border: none !important; }
.stTextInput > div > div > input, .stSelectbox > div > div > select { border: 2px solid #e5e7eb; border-radius: 8px; padding: 10px 14px; font-size: 0.9rem; transition: all 0.2s ease; background: #fafafa; }
.stTextInput > div > div > input:focus, .stSelectbox > div > div > select:focus { border-color: #667eea; background: white; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1); }
.block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1400px; }
.stSuccess { background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border: 1px solid #b8dacd; border-radius: 8px; color: #155724; }
.stWarning { background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); border: 1px solid #f8d7da; border-radius: 8px; color: #856404; }
.stError { background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); border: 1px solid #f1b0b7; border-radius: 8px; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────
# ✅ 헤더
# ──────────────────────────────
st.markdown('''
<div class="header-container">
    <img src="https://raw.githubusercontent.com/JeongWon4034/cheongju/main/cheongpung_logo.png" alt='청풍로드 로고' style ="width:125px; height:125px">
    <div class="main-title">청풍로드 - 충청북도 맞춤형 AI기반 스마트 관광 가이드</div>
</div>
<div class="title-underline"></div>
''', unsafe_allow_html=True)

# ──────────────────────────────
# ✅ 메인 레이아웃 (3컬럼)
# ──────────────────────────────
col1, col2, col3 = st.columns([1.5, 1.2, 3], gap="large")

# ------------------------------
# ✅ [좌] DRT 노선 추천 설정
# ------------------------------
with col1:
    st.markdown('<div class="section-header">🚐 DRT 노선 추천 설정</div>', unsafe_allow_html=True)
    st.caption("출발/경유 정류장을 선택하고 노선을 추천받으세요.")

    # 운행 모드
    st.markdown("**운행 모드**")
    mode = st.radio(
        "", 
        ["차량(운행)", "도보(승객 접근)"],
        horizontal=True, 
        key="mode_key", 
        label_visibility="collapsed"
    )

    # 출발 시각 & 시간대  ←★ 추가
    st.markdown("**출발 시각 & 시간대**")
    dep_time = st.time_input(
        "",
        value=datetime.time(9, 0),
        key="dep_time",
        label_visibility="collapsed",
        help="이 시각을 기준으로 각 정류장의 도착 예정 시간을 계산합니다."
    )
    time_band = st.selectbox(
        "",
        ["혼잡(출퇴근)", "일반", "심야/한산"],
        index=1,
        key="time_band",
        label_visibility="collapsed",
        help="시간대별 혼잡도를 반영해 소요시간을 보정합니다. (차량만 적용)"
    )

    # 출발 정류장
    st.markdown("**출발 정류장**")
    names_list = gdf["juso"].dropna().astype(str).unique().tolist()
    start = st.selectbox(
        "",
        names_list,
        key="start_key",
        label_visibility="collapsed",
        help="DRT 운행을 시작할 정류장을 선택하세요."
    )

    # 경유 정류장
    st.markdown("**경유 정류장 (선택)**")
    wps = st.multiselect(
        "",
        [n for n in names_list if n != st.session_state.get("start_key", "")],
        key="wps_key",
        label_visibility="collapsed",
        help="중간에 들를 정류장을 선택하세요. (복수 선택 가능)"
    )

    # 버튼
    col_btn1, col_btn2 = st.columns(2, gap="small")
    with col_btn1:
        create_clicked = st.button("노선 추천")
    with col_btn2:
        clear_clicked = st.button("초기화")

# ------------------------------
# ✅ 초기화 처리
# ------------------------------
if clear_clicked:
    try:
        keys_to_clear = ["segments", "order", "duration", "distance", "auto_gpt_input", "leg_durations"]
        for k in keys_to_clear:
            if k in st.session_state:
                if k in ["segments", "order", "leg_durations"]:
                    st.session_state[k] = []
                elif k in ["duration", "distance"]:
                    st.session_state[k] = 0.0
                else:
                    st.session_state[k] = ""
        for widget_key in ["mode_key", "start_key", "wps_key", "dep_time", "time_band"]:
            if widget_key in st.session_state:
                del st.session_state[widget_key]
        st.success("✅ 초기화가 완료되었습니다.")
        st.rerun()
    except Exception as e:
        st.error(f"❌ 초기화 중 오류: {str(e)}")

# ------------------------------
# ✅ [중간] 방문순서 + 메트릭
# ------------------------------
with col2:
    st.markdown('<div class="section-header">📍 여행 방문 순서</div>', unsafe_allow_html=True)

    current_order = st.session_state.get("order", [])
    if current_order:
        for i, name in enumerate(current_order, 1):
            st.markdown(f"""
            <div class="visit-order-item">
                <div class="visit-number">{i}</div>
                <div>{name}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state">경로 생성 후 표시됩니다<br>🗺️</div>', unsafe_allow_html=True)

    # ★★★ ETA(도착 예정 시각) 표: 방문 순서 아래 고정 표시 (여기에 넣는 게 포인트)
    if st.session_state.get("segments") and st.session_state.get("leg_durations"):
        try:
            start_dt = datetime.datetime.combine(
                datetime.date.today(),
                st.session_state.get("dep_time", datetime.time(9, 0))
            )
            times = [start_dt]
            for dmin in st.session_state["leg_durations"]:
                times.append(times[-1] + datetime.timedelta(minutes=float(dmin)))

            labels = st.session_state.get("order", [])[:]
            if len(labels) < len(times):
                labels = labels + [f"목적지 {len(times)-len(labels)}"]
            labels = labels[:len(times)]

            st.markdown("**🕒 도착 예정 시간표**")
            eta_df = pd.DataFrame({
                "정류장": labels,
                "도착 예정 시각": [t.strftime("%H:%M") for t in times]
            })
            st.dataframe(eta_df, use_container_width=True)
        except Exception as _e:
            st.warning(f"ETA 계산 중 경고: {str(_e)}")

    st.markdown("---")
    st.metric("⏱️ 소요시간", f"{st.session_state.get('duration', 0.0):.1f}분")
    st.metric("📏 이동거리", f"{st.session_state.get('distance', 0.0):.2f}km")

# ------------------------------
# ✅ [우] 지도
# ------------------------------
with col3:
    st.markdown('<div class="section-header">🗺️ 추천경로 지도시각화</div>', unsafe_allow_html=True)

    # 지도 중심
    try:
        ctr = boundary.geometry.centroid
        clat, clon = float(ctr.y.mean()), float(ctr.x.mean())
        if math.isnan(clat) or math.isnan(clon):
            clat, clon = 36.64, 127.48
    except Exception as e:
        st.warning(f"중심점 계산 오류: {str(e)}")
        clat, clon = 36.64, 127.48

    @st.cache_data
    def load_graph(lat, lon):
        try:
            return ox.graph_from_point((lat, lon), dist=3000, network_type="all")
        except Exception as e:
            st.warning(f"도로 네트워크 로드 실패: {str(e)}")
            try:
                return ox.graph_from_point((36.64, 127.48), dist=3000, network_type="all")
            except:
                return None

    G = load_graph(clat, clon)
    edges = None
    if G is not None:
        try:
            edges = ox.graph_to_gdfs(G, nodes=False)
        except Exception as e:
            st.warning(f"엣지 변환 실패: {str(e)}")

    # 스냅 포인트 생성
    stops = [start] + wps
    snapped = []
    try:
        for nm in stops:
            matching_rows = gdf[gdf["name"] == nm]
            if matching_rows.empty:
                st.warning(f"⚠️ '{nm}' 정보를 찾을 수 없습니다.")
                continue
            r = matching_rows.iloc[0]
            if pd.isna(r.lon) or pd.isna(r.lat):
                st.warning(f"⚠️ '{nm}'의 좌표 정보가 없습니다.")
                continue
            pt = Point(r.lon, r.lat)

            if edges is None or edges.empty:
                snapped.append((r.lon, r.lat))
                continue

            edges["d"] = edges.geometry.distance(pt)
            if edges["d"].empty:
                snapped.append((r.lon, r.lat))
                continue

            ln = edges.loc[edges["d"].idxmin()]
            sp = ln.geometry.interpolate(ln.geometry.project(pt))
            snapped.append((sp.x, sp.y))
    except Exception as e:
        st.error(f"❌ 지점 처리 중 오류: {str(e)}")
        snapped = []
        for nm in stops:
            try:
                r = gdf[gdf["name"] == nm].iloc[0]
                if not (pd.isna(r.lon) or pd.isna(r.lat)):
                    snapped.append((r.lon, r.lat))
            except Exception as coord_error:
                st.warning(f"⚠️ '{nm}' 좌표를 가져올 수 없습니다: {str(coord_error)}")

    # ───────────────────────────
    # ✅ 경로 생성 처리 (개선본)
    # ───────────────────────────
    if create_clicked and len(snapped) >= 2:
        try:
            segs, total_sec, total_meter = [], 0.0, 0.0
            leg_durs_min = []
            api_mode = "walking" if "도보" in st.session_state.get("mode_key", "") else "driving"

            cong = congestion_factor(st.session_state.get("time_band", "일반"),
                                     st.session_state.get("mode_key", "차량(운행)"))

            for i in range(len(snapped) - 1):
                x1, y1 = snapped[i]
                x2, y2 = snapped[i + 1]
                coord = f"{x1},{y1};{x2},{y2}"

                url = f"https://api.mapbox.com/directions/v5/mapbox/{api_mode}/{coord}"
                params = {
                    "geometries": "geojson",
                    "overview": "full",
                    "access_token": MAPBOX_TOKEN
                }

                try:
                    r = requests.get(url, params=params, timeout=10)
                    if r.status_code == 200:
                        data_resp = r.json()
                        if data_resp.get("routes"):
                            route = data_resp["routes"][0]
                            sec_base = float(route.get("duration", 0.0))
                            dist_m   = float(route.get("distance", 0.0))
                            sec_adj = sec_base * (cong if api_mode == "driving" else 1.0)

                            segs.append(route["geometry"]["coordinates"])
                            total_sec   += sec_adj
                            total_meter += dist_m
                            leg_durs_min.append(sec_adj / 60.0)
                        else:
                            st.warning(f"⚠️ 구간 {i+1}의 경로를 찾을 수 없습니다.")
                    else:
                        st.warning(f"⚠️ API 호출 실패 (상태코드: {r.status_code})")
                except requests.exceptions.Timeout:
                    st.warning("⚠️ API 호출 시간 초과")
                except Exception as api_error:
                    st.warning(f"⚠️ API 호출 오류: {str(api_error)}")

            if segs:
                st.session_state["order"] = stops
                st.session_state["segments"] = segs
                st.session_state["duration"] = total_sec / 60.0
                st.session_state["distance"] = total_meter / 1000.0
                st.session_state["leg_durations"] = leg_durs_min

                st.success(
                    f"✅ 경로가 성공적으로 생성되었습니다! · 총 {st.session_state['distance']:.2f} km · "
                    f"약 {st.session_state['duration']:.1f} 분 (시간대 보정계수 x{cong:.2f})"
                )

                # 경로 생성 직후 1회 ETA 바로 표시 (추가로 중앙 패널에서도 표가 보임)
                try:
                    start_dt = datetime.datetime.combine(
                        datetime.date.today(),
                        st.session_state.get("dep_time", datetime.time(9, 0))
                    )
                    times = [start_dt]
                    for dmin in leg_durs_min:
                        times.append(times[-1] + datetime.timedelta(minutes=float(dmin)))

                    labels = st.session_state["order"][:]
                    if len(labels) < len(times):
                        labels = labels + [f"목적지 {len(times)-len(labels)}"]
                    labels = labels[:len(times)]

                    sched_df = pd.DataFrame({
                        "정류장": labels,
                        "도착 예정 시각": [t.strftime("%H:%M") for t in times]
                    })
                    st.dataframe(sched_df, use_container_width=True)
                except Exception as _e:
                    st.warning(f"ETA 표 생성 중 경고: {str(_e)}")
            else:
                st.error("❌ 모든 구간의 경로 생성에 실패했습니다.")
        except Exception as e:
            st.error(f"❌ 경로 생성 중 오류 발생: {str(e)}")
            st.info("💡 다른 출발지나 경유지를 선택해보세요.")

    # 지도 렌더링
    try:
        m = folium.Map(
            location=[clat, clon],
            zoom_start=12,
            tiles="CartoDB Positron",
            prefer_canvas=True,
            control_scale=True
        )

        if boundary is not None:
            folium.GeoJson(boundary, style_function=lambda f: {
                "color": "#9aa0a6",
                "weight": 2,
                "dashArray": "4,4",
                "fillOpacity": 0.05
            }).add_to(m)

        mc = MarkerCluster().add_to(m)
        for _, row in gdf.iterrows():
            if not (pd.isna(row.lat) or pd.isna(row.lon)):
                folium.Marker(
                    [row.lat, row.lon],
                    popup=folium.Popup(str(row["name"]), max_width=200),
                    tooltip=str(row["name"]),
                    icon=folium.Icon(color="gray")
                ).add_to(mc)

        current_order = st.session_state.get("order", stops)
        for idx, (x, y) in enumerate(snapped, 1):
            place_name = current_order[idx - 1] if idx <= len(current_order) else f"지점 {idx}"
            folium.Marker(
                [y, x],
                icon=folium.Icon(color="red", icon="flag"),
                tooltip=f"{idx}. {place_name}",
                popup=folium.Popup(f"<b>{idx}. {place_name}</b>", max_width=200)
            ).add_to(m)

        if st.session_state.get("segments"):
            palette = ["#4285f4", "#34a853", "#ea4335", "#fbbc04", "#9c27b0", "#ff9800"]
            segments = st.session_state["segments"]
            used_positions = []
            min_distance = 0.001

            for i, seg in enumerate(segments):
                if seg:
                    folium.PolyLine(
                        [(pt[1], pt[0]) for pt in seg],
                        color=palette[i % len(palette)],
                        weight=5,
                        opacity=0.8
                    ).add_to(m)

                    mid = seg[len(seg) // 2]
                    candidate_pos = [mid[1], mid[0]]
                    while any(
                        abs(candidate_pos[0] - used[0]) < min_distance and
                        abs(candidate_pos[1] - used[1]) < min_distance
                        for used in used_positions
                    ):
                        candidate_pos[0] += min_distance * 0.5
                        candidate_pos[1] += min_distance * 0.5

                    folium.map.Marker(
                        candidate_pos,
                        icon=DivIcon(html=f"<div style='background:{palette[i % len(palette)]};"
                                          "color:#fff;border-radius:50%;width:28px;height:28px;"
                                          "line-height:28px;text-align:center;font-weight:600;"
                                          "box-shadow:0 2px 4px rgba(0,0,0,0.3);'>"
                                          f"{i+1}</div>")
                    ).add_to(m)
                    used_positions.append(candidate_pos)

            try:
                pts = [pt for seg in segments for pt in seg if seg]
                if pts:
                    m.fit_bounds([[min(p[1] for p in pts), min(p[0] for p in pts)],
                                  [max(p[1] for p in pts), max(p[0] for p in pts)]])
            except:
                m.location = [clat, clon]
                m.zoom_start = 12
        else:
            m.location = [clat, clon]
            m.zoom_start = 12

        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        _ = st_folium(
            m,
            width="100%",
            height=520,
            returned_objects=[],
            use_container_width=True,
            key="main_map"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as map_error:
        st.error(f"❌ 지도 렌더링 오류: {str(map_error)}")
        st.markdown(
            '<div class="map-container" style="display: flex; align-items: center; justify-content: center; color: #6b7280;">지도를 불러올 수 없습니다.</div>',
            unsafe_allow_html=True
        )
