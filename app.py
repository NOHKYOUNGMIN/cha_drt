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



# ────────────────────────────── 
# ✅ 환경변수 불러오기 (Streamlit Cloud 호환에 저장된 키 사용)
# ──────────────────────────────
MAPBOX_TOKEN = "pk.eyJ1IjoiZ3VyMDUxMDgiLCJhIjoiY21lZ2k1Y291MTdoZjJrb2k3bHc3cTJrbSJ9.DElgSQ0rPoRk1eEacPI8uQ"

# ──────────────────────────────
# ✅ 데이터 로드 (안전한 로드)
# ──────────────────────────────
# ──────────────────────────────
# ✅ 데이터 로드 (DRT 라인 셰이프 자동 병합)
# ──────────────────────────────
@st.cache_data
def load_data():
    import glob
    from pathlib import Path
    try:
        # 1) 대상 shp 자동 탐색 (작업 폴더 기준)
        #    drt_1.shp ~ drt_4.shp, new_drt.shp
        patterns = ["./drt_*.shp", "./new_drt.shp"]
        shp_files = []
        for p in patterns:
            shp_files.extend(glob.glob(p))
        shp_files = sorted(set(shp_files))

        if not shp_files:
            raise FileNotFoundError("drt_*.shp / new_drt.shp 를 찾지 못했습니다.")

        # 2) 모두 읽어서 하나로 병합
        gdfs = []
        for f in shp_files:
            _g = gpd.read_file(f)
            # 출처 파일명 보존(나중에 name 자동 생성에 사용)
            _g["source_file"] = Path(f).stem
            gdfs.append(_g)

        gdf = pd.concat(gdfs, ignore_index=True)
        # GeoDataFrame 보장
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdfs[0].crs)

        # 3) 좌표계 통일 → EPSG:4326
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        # 4) name 컬럼 보정(없으면 생성)
        name_candidates = [c for c in gdf.columns if c.lower() in ["name", "정류장명", "stop_name", "title"]]
        if name_candidates:
            name_col = name_candidates[0]
            gdf["name"] = gdf[name_col].astype(str)
        else:
            # 문자열형 첫 컬럼 시도
            obj_cols = [c for c in gdf.columns if c != "geometry" and gdf[c].dtype == "object"]
            if obj_cols:
                gdf["name"] = gdf[obj_cols[0]].astype(str)
            else:
                # 완전 없으면 파일명+인덱스로 생성
                gdf["name"] = gdf.apply(
                    lambda r: f"{r.get('source_file','drt')}_{int(r.name)+1}", axis=1
                )

        # 5) lon/lat 생성
        #    - Point면 그대로 x/y
        #    - 그 외(Line/Polygon)는 representative_point로 대체
        if gdf.geometry.geom_type.isin(["Point"]).all():
            gdf["lon"] = gdf.geometry.x
            gdf["lat"] = gdf.geometry.y
        else:
            reps = gdf.geometry.representative_point()
            gdf["lon"] = reps.x
            gdf["lat"] = reps.y

        # 6) boundary 생성
        #    - 기존 cb_shp.shp 있으면 사용, 없으면 데이터의 convex hull
        boundary_path = Path("./cb_shp.shp")
        if boundary_path.exists():
            boundary = gpd.read_file(boundary_path).to_crs(epsg=4326)
        else:
            try:
                union = gdf.unary_union
                hull = union.convex_hull
                boundary = gpd.GeoDataFrame(geometry=[hull], crs="EPSG:4326")
            except Exception:
                boundary = None

        return gdf, boundary

    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {str(e)}")
        return None, None

# ↓ 그대로 유지
gdf, boundary = load_data()

# 데이터 로드 실패 시 앱 중단
if gdf is None:
    st.stop()


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
# ✅ 페이지 설정 & 스타일
# ──────────────────────────────
st.set_page_config(
    page_title="청풍로드 - 충청북도 맞춤형 AI기반 스마트 관광 가이드",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* 기본 폰트 시스템 */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* 기본 스타일 */
.main > div {
    padding-top: 1.2rem;
    padding-bottom: 0.5rem;
}

header[data-testid="stHeader"] {
    display: none;
}

.stApp {
    background: #f8f9fa;
}

/* 헤더 컨테이너 (로고 + 제목) */
.header-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
    margin-bottom: 2rem;
    padding: 1rem 0;
}

.logo-image {
    width: 50px;
    height: 50px;
    object-fit: contain;
}

.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: #202124;
    letter-spacing: -1px;
    margin: 0;
}

.title-underline {
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, #4285f4, #34a853);
    margin: 0 auto 2rem auto;
    border-radius: 2px;
}

/* 섹션 헤더 스타일 */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
    padding-bottom: 12px;
    border-bottom: 2px solid #f3f4f6;
}

/* 버튼 스타일 개선 */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-size: 0.9rem;
    font-weight: 600;
    width: 100%;
    height: 48px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
}

/* 방문 순서 리스트 스타일 */
.visit-order-item {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px;
    margin-bottom: 8px;
    font-size: 0.95rem;
    font-weight: 500;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
}

.visit-order-item:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
}

.visit-number {
    background: rgba(255,255,255,0.9);
    color: #667eea;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 12px;
    flex-shrink: 0;
}

/* 메트릭 카드 스타일 */
.stMetric {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    border: none;
    border-radius: 12px;
    padding: 16px 10px;
    text-align: center;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(168, 237, 234, 0.3);
}

.stMetric:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(168, 237, 234, 0.4);
}

/* 빈 상태 메시지 */
.empty-state {
    text-align: center;
    padding: 40px 20px;
    color: #9ca3af;
    font-style: italic;
    font-size: 0.95rem;
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    border-radius: 12px;
    margin: 16px 0;
}

/* 🔧 지도 컨테이너 스타일 - 박스 제거 완전 수정 */
.map-container {
    width: 100% !important;
    height: 520px !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    position: relative !important;
    background: transparent !important;
    border: 2px solid #e5e7eb !important;
    margin: 0 !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}

/* Streamlit iframe 완전 초기화 */
div[data-testid="stIFrame"] {
    width: 100% !important;
    max-width: 100% !important;
    height: 520px !important;
    position: relative !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
    border-radius: 12px !important;
    background: transparent !important;
    border: none !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stIFrame"] > iframe {
    width: 100% !important;
    height: 100% !important;
    border: none !important;
    border-radius: 12px !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    background: transparent !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* 🚨 핵심: Streamlit 내부 빈 div들 완전 제거 */
div[data-testid="stIFrame"] > iframe > html > body > div:empty {
    display: none !important;
}

div[data-testid="stIFrame"] div:empty {
    display: none !important;
}

/* 🚨 Folium 내부 빈 컨테이너 제거 */
.folium-map div:empty {
    display: none !important;
}

/* 🚨 Leaflet 오버레이 박스 제거 */
.leaflet-container .leaflet-control-container div:empty {
    display: none !important;
}

.leaflet-container > div:empty {
    display: none !important;
}

/* 🚨 모든 빈 오버레이 박스 강제 제거 */
div:empty:not(.leaflet-zoom-box):not(.leaflet-marker-icon):not(.leaflet-div-icon) {
    display: none !important;
}

/* 🚨 투명하거나 흰색 배경의 빈 박스들 제거 */
div[style*="background: white"]:empty,
div[style*="background: #fff"]:empty,
div[style*="background: #ffffff"]:empty,
div[style*="background-color: white"]:empty,
div[style*="background-color: #fff"]:empty,
div[style*="background-color: #ffffff"]:empty {
    display: none !important;
}

/* Folium/Leaflet 지도 자체 크기 제한 */
.folium-map {
    width: 100% !important;
    height: 100% !important;
    max-width: 100% !important;
    max-height: 520px !important;
    box-sizing: border-box !important;
    background: transparent !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
}

/* Leaflet 컨테이너 크기 고정 */
.leaflet-container {
    width: 100% !important;
    height: 100% !important;
    max-width: 100% !important;
    max-height: 520px !important;
    box-sizing: border-box !important;
    background: transparent !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
}

/* 폼 스타일 개선 */
.stTextInput > div > div > input,
.stSelectbox > div > div > select {
    border: 2px solid #e5e7eb;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.9rem;
    transition: all 0.2s ease;
    background: #fafafa;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div > select:focus {
    border-color: #667eea;
    background: white;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

/* 간격 조정 */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 1400px;
}

/* 성공/경고 메시지 */
.stSuccess {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border: 1px solid #b8dacd;
    border-radius: 8px;
    color: #155724;
}

.stWarning {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    border: 1px solid #f8d7da;
    border-radius: 8px;
    color: #856404;
}

.stError {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border: 1px solid #f1b0b7;
    border-radius: 8px;
    color: #721c24;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────
# ✅ 헤더 (GitHub Raw URL로 로고 이미지 로드)
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
# ✅ [좌] DRT 노선 추천 설정 (라벨만 개선)
# ------------------------------
with col1:
    st.markdown('<div class="section-header">🚐 DRT 노선 추천 설정</div>', unsafe_allow_html=True)

    # 안내 문구(선택)
    st.caption("출발/경유 정류장을 선택하고 노선을 추천받으세요.")

    # ─ 이동 모드(용어만 변경)
    st.markdown("**운행 모드**")
    mode = st.radio(
        "", 
        ["차량(운행)", "도보(승객 접근)"],   # ← 문구만 변경
        horizontal=True, 
        key="mode_key", 
        label_visibility="collapsed"
    )

    # ─ 출발 정류장
    st.markdown("**출발 정류장**")
    names_list = gdf["name"].dropna().astype(str).unique().tolist()
    start = st.selectbox(
        "",
        names_list,
        key="start_key",
        label_visibility="collapsed",
        help="DRT 운행을 시작할 정류장을 선택하세요."
    )

    # ─ 경유 정류장 (선택)
    st.markdown("**경유 정류장 (선택)**")
    wps = st.multiselect(
        "",
        [n for n in names_list if n != st.session_state.get("start_key", "")],
        key="wps_key",
        label_visibility="collapsed",
        help="중간에 들를 정류장을 선택하세요. (복수 선택 가능)"
    )

    # ─ 버튼
    col_btn1, col_btn2 = st.columns(2, gap="small")
    with col_btn1:
        # 기존 변수명/키 유지 (다른 로직과 호환)
        create_clicked = st.button("노선 추천")   # ← '경로 생성' → '노선 추천'
    with col_btn2:
        clear_clicked = st.button("초기화")

if create_clicked and len(snapped) >= 2:
    try:
        # 시간대 보정계수 (차량만 영향)
        cong = congestion_factor(st.session_state.get("time_band", "일반"),
                                 st.session_state.get("mode_key", "차량(운행)"))

        # 출발 datetime (오늘 날짜 + 사용자가 고른 시각)
        start_dt = datetime.datetime.combine(datetime.date.today(), st.session_state.get("dep_time", datetime.time(9, 0)))

        leg_geoms = []       # 각 구간의 폴리라인 (lon,lat 또는 [lon,lat] 리스트)
        leg_duration_min = []  # 각 구간의 소요시간(분)
        total_distance_km = 0.0

        if G_NET and G_NET.number_of_edges() > 0:
            # ── 노선망 기준(네트워크) 경로
            # 기본 속도 (차량/도보)
            base_speed = 4.5 if "도보" in st.session_state.get("mode_key", "") else 25.0
            eff_speed = base_speed / cong  # 혼잡 시 속도 저하

            for i in range(len(snapped) - 1):
                lon1, lat1 = snapped[i]
                lon2, lat2 = snapped[i + 1]

                u = nearest_node(G_NET, lon1, lat1)
                v = nearest_node(G_NET, lon2, lat2)
                path_nodes = nx.shortest_path(G_NET, u, v, weight="weight")

                # 선분 좌표 [[lon,lat], ...]
                seg_ll = path_to_polyline_lonlat(G_NET, path_nodes)
                leg_geoms.append(seg_ll)

                # 거리 합계 + 구간 시간
                seg_m = 0.0
                for uu, vv in zip(path_nodes[:-1], path_nodes[1:]):
                    data = G_NET.get_edge_data(uu, vv)
                    if data:
                        seg_m += float(data.get("length_m", 0.0))

                total_distance_km += seg_m / 1000.0
                leg_duration_min.append((seg_m / 1000.0) / max(eff_speed, 0.1) * 60.0)

            total_duration_min = sum(leg_duration_min)

        else:
            # ── 폴백: Mapbox Directions (혼잡은 duration에 계수 곱)
            api_mode = "walking" if "도보" in st.session_state.get("mode_key", "") else "driving"
            total_duration_min = 0.0

            for i in range(len(snapped) - 1):
                x1, y1 = snapped[i]
                x2, y2 = snapped[i + 1]
                coord = f"{x1},{y1};{x2},{y2}"

                url = f"https://api.mapbox.com/directions/v5/mapbox/{api_mode}/{coord}"
                params = {"geometries": "geojson", "overview": "full", "access_token": MAPBOX_TOKEN}

                try:
                    r = requests.get(url, params=params, timeout=10)
                    if r.status_code == 200:
                        data_resp = r.json()
                        if data_resp.get("routes"):
                            route = data_resp["routes"][0]
                            coords = route["geometry"]["coordinates"]  # [[lon,lat], ...]
                            base_sec = route.get("duration", 0.0)      # 초
                            dist_m  = route.get("distance", 0.0)

                            # 혼잡 보정(차량만 적용) → 초 * cong
                            sec = base_sec * (cong if "walking" not in api_mode else 1.0)

                            leg_geoms.append(coords)
                            leg_duration_min.append(sec / 60.0)
                            total_duration_min += sec / 60.0
                            total_distance_km += (dist_m / 1000.0)
                        else:
                            st.warning(f"⚠️ 구간 {i+1} 경로 없음")
                    else:
                        st.warning(f"⚠️ Directions API 실패 코드 {r.status_code}")
                except requests.exceptions.Timeout:
                    st.warning("⚠️ Directions API 타임아웃")
                except Exception as api_error:
                    st.warning(f"⚠️ Directions API 오류: {api_error}")

        # ── 세션 업데이트
        st.session_state["order"] = [*([start] + wps)]
        st.session_state["segments"] = leg_geoms
        st.session_state["distance"] = total_distance_km
        st.session_state["duration"] = total_duration_min

        # ✅ 구간별 도착 예정 시간표 계산
        times = [start_dt]
        for dmin in leg_duration_min:
            times.append(times[-1] + datetime.timedelta(minutes=float(dmin)))

        # stops 라벨 (마지막 도착 포함)
        stop_labels = [start] + wps
        if len(stop_labels) < len(times):  # 마지막 목적지 라벨 보강
            stop_labels = stop_labels + [f"목적지 {len(times)-len(stop_labels)}"]
        stop_labels = stop_labels[:len(times)]

        sched_df = pd.DataFrame({
            "정류장": stop_labels,
            "도착 예정 시각": [t.strftime("%H:%M") for t in times]
        })

        st.success(f"✅ 경로 생성 완료 · 총 {total_distance_km:.2f} km · 약 {total_duration_min:.1f} 분 "
                   f"(시간대 보정계수 x{cong:.2f})")
        st.dataframe(sched_df, use_container_width=True)

        st.rerun()

    except nx.NetworkXNoPath:
        st.error("경로가 없습니다. 노선 데이터가 끊겨 있거나 교차점이 연결되지 않았을 수 있어요.")
    except Exception as e:
        st.error(f"❌ 경로 생성 중 오류: {str(e)}")


# ------------------------------
# ✅ 초기화 처리 개선
# ------------------------------
if clear_clicked:
    try:
        keys_to_clear = ["segments", "order", "duration", "distance", "auto_gpt_input"]
        for k in keys_to_clear:
            if k in st.session_state:
                if k in ["segments", "order"]:
                    st.session_state[k] = []
                elif k in ["duration", "distance"]:
                    st.session_state[k] = 0.0
                else:
                    st.session_state[k] = ""
        
        widget_keys = ["mode_key", "start_key", "wps_key"]
        for widget_key in widget_keys:
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
            st.markdown(f'''
            <div class="visit-order-item">
                <div class="visit-number">{i}</div>
                <div>{name}</div>
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state">경로 생성 후 표시됩니다<br>🗺️</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.metric("⏱️ 소요시간", f"{st.session_state.get('duration', 0.0):.1f}분")
    st.metric("📏 이동거리", f"{st.session_state.get('distance', 0.0):.2f}km")

# ------------------------------
# ✅ [우] 지도
# ------------------------------
with col3:
    st.markdown('<div class="section-header">🗺️ 추천경로 지도시각화</div>', unsafe_allow_html=True)
    
    # 지도 설정
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

    stops = [start] + wps
    snapped = []

    # 개선된 스냅핑
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

    # 경로 생성 처리
    if create_clicked and len(snapped) >= 2:
        try:
            segs, td, tl = [], 0.0, 0.0
            api_mode = "walking" if mode == "도보" else "driving"
            
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
                        if data_resp.get("routes") and len(data_resp["routes"]) > 0:
                            route = data_resp["routes"][0]
                            segs.append(route["geometry"]["coordinates"])
                            td += route.get("duration", 0)
                            tl += route.get("distance", 0)
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
                st.session_state["duration"] = td / 60
                st.session_state["distance"] = tl / 1000
                st.session_state["segments"] = segs
                st.success("✅ 경로가 성공적으로 생성되었습니다!")
                st.rerun()
            else:
                st.error("❌ 모든 구간의 경로 생성에 실패했습니다.")
                
        except Exception as e:
            st.error(f"❌ 경로 생성 중 오류 발생: {str(e)}")
            st.info("💡 다른 출발지나 경유지를 선택해보세요.")

    # 🔧 지도 렌더링 - 빈 박스 제거 최적화
    try:
        m = folium.Map(
            location=[clat, clon], 
            zoom_start=12, 
            tiles="CartoDB Positron",
            # 🚨 추가 옵션으로 오버레이 방지
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
                folium.Marker([row.lat, row.lon], 
                            popup=folium.Popup(str(row["name"]), max_width=200),
                            tooltip=str(row["name"]),
                            icon=folium.Icon(color="gray")).add_to(mc)
        
        current_order = st.session_state.get("order", stops)
        for idx, (x, y) in enumerate(snapped, 1):
            if idx <= len(current_order):
                place_name = current_order[idx - 1]
            else:
                place_name = f"지점 {idx}"
            
            folium.Marker([y, x], 
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
                    folium.PolyLine([(pt[1], pt[0]) for pt in seg], 
                                  color=palette[i % len(palette)], 
                                  weight=5, 
                                  opacity=0.8
                    ).add_to(m)
                    
                    mid = seg[len(seg) // 2]
                    candidate_pos = [mid[1], mid[0]]
                    
                    while any(abs(candidate_pos[0] - used[0]) < min_distance and 
                            abs(candidate_pos[1] - used[1]) < min_distance 
                            for used in used_positions):
                        candidate_pos[0] += min_distance * 0.5
                        candidate_pos[1] += min_distance * 0.5
                    
                    folium.map.Marker(candidate_pos,
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
        
        # 🚨 레이어 컨트롤 제거 - 빈 박스 원인 가능성
        # folium.LayerControl().add_to(m)
        
        # 🔧 지도 컨테이너 - 완전 수정된 구조
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        map_data = st_folium(
            m,
            width="100%",
            height=520,
            returned_objects=[],  # 🚨 빈 객체 반환 방지
            use_container_width=True,
            key="main_map"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as map_error:
        st.error(f"❌ 지도 렌더링 오류: {str(map_error)}")
        st.markdown('<div class="map-container" style="display: flex; align-items: center; justify-content: center; color: #6b7280;">지도를 불러올 수 없습니다.</div>', unsafe_allow_html=True)
