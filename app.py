# app.py
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
import math, os, datetime, glob, shutil, tempfile
from pathlib import Path
import fiona

# ── 페이지/스타일 ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="청풍로드 - 충청북도 맞춤형 AI기반 스마트 관광 가이드",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif; }
.main > div { padding-top: 1.2rem; padding-bottom: 0.5rem; }
header[data-testid="stHeader"] { display: none; }
.stApp { background: #f8f9fa; }
.header-container { display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 2rem; padding: 1rem 0; }
.main-title { font-size: 2.8rem; font-weight: 700; color: #202124; letter-spacing: -1px; margin: 0; }
.title-underline { width: 100%; height: 3px; background: linear-gradient(90deg, #4285f4, #34a853); margin: 0 auto 2rem auto; border-radius: 2px; }
.section-header { font-size: 1.3rem; font-weight: 700; color: #1f2937; margin-bottom: 20px; display: flex; align-items: center; gap: 8px; padding-bottom: 12px; border-bottom: 2px solid #f3f4f6; }
.stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 10px; padding: 12px 20px; font-size: 0.9rem; font-weight: 600; width: 100%; height: 48px; transition: all 0.3s ease; box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3); }
.visit-order-item { display: flex; align-items: center; padding: 12px 16px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; margin-bottom: 8px; font-size: 0.95rem; font-weight: 500; }
.empty-state { text-align: center; padding: 40px 20px; color: #9ca3af; font-style: italic; font-size: 0.95rem; background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); border-radius: 12px; margin: 16px 0; }
.map-container { width: 100% !important; height: 520px !important; border-radius: 12px !important; overflow: hidden !important; position: relative !important; background: transparent !important; border: 2px solid #e5e7eb !important; }
div[data-testid="stIFrame"], .folium-map, .leaflet-container { width: 100% !important; height: 520px !important; max-height: 520px !important; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1400px; }
</style>
""", unsafe_allow_html=True)

st.markdown('''
<div class="header-container">
    <img src="https://raw.githubusercontent.com/JeongWon4034/cheongju/main/cheongpung_logo.png" alt='청풍로드 로고' style ="width:125px; height:125px">
    <div class="main-title">청풍로드 - 충청북도 맞춤형 AI기반 스마트 관광 가이드</div>
</div>
<div class="title-underline"></div>
''', unsafe_allow_html=True)

# ── 토큰/드라이버 힌트 ───────────────────────────────────────────────────────────
MAPBOX_TOKEN = "pk.eyJ1IjoiZ3VyMDUxMDgiLCJhIjoiY21lZ2k1Y291MTdoZjJrb2k3bHc3cTJrbSJ9.DElgSQ0rPoRk1eEacPI8uQ"
fiona.drvsupport.supported_drivers["ESRI Shapefile"] = "raw"
os.environ.setdefault("SHAPE_ENCODING", "CP949")  # GDAL 힌트

# ── 셰이프 로딩(함수 없이, 강제 .cpg 교정 포함) ───────────────────────────────────
patterns = ["./drt_*.shp", "./new_new_drt.shp"]
shp_files = []
for p in patterns:
    shp_files.extend(glob.glob(p))
shp_files = sorted(set(shp_files))

if not shp_files:
    st.error("❌ drt_*.shp / new_new_drt.shp 파일을 찾지 못했습니다.")
    st.stop()

gdfs = []
read_logs = []
failed_logs = []
# UTF-8 시도 제거. CP949/EUC-KR만 우선, 최후에 latin1
ENCODING_TRY_ORDER = ["cp949", "euc-kr", "latin1"]

tmp_dir = Path(tempfile.mkdtemp(prefix="cpg_fix_"))

for f in shp_files:
    f = Path(f)
    ok = False
    tried = []
    enc_used = None
    last_err = None

    # 1) 일반 시도(CP949 → EUC-KR → latin1)
    for enc in ENCODING_TRY_ORDER:
        try:
            _g = gpd.read_file(str(f), encoding=enc)
            ok = True; enc_used = enc
            break
        except Exception as e:
            tried.append(enc); last_err = str(e)

    # 2) 실패 시: .cpg 강제 교정해서 재시도
    if not ok:
        try:
            base = f.with_suffix("")  # 파일 베이스명
            siblings = list(f.parent.glob(base.name + ".*"))  # 같은 이름의 모든 확장자
            target_dir = tmp_dir / base.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for s in siblings:
                shutil.copy2(s, target_dir / s.name)  # 원본 -> 임시폴더 복사

            # .cpg를 CP949로 덮어쓰기(없는 경우 생성)
            (target_dir / (base.name + ".cpg")).write_text("CP949", encoding="ascii")

            shp_tmp = target_dir / f.name  # 임시 위치의 shp 경로
            for enc in ["cp949", "euc-kr", "latin1"]:
                try:
                    _g = gpd.read_file(str(shp_tmp), encoding=enc)
                    ok = True; enc_used = f"{enc} (cpg-fixed)"
                    break
                except Exception as e2:
                    tried.append(f"cpgfix:{enc}"); last_err = str(e2)
        except Exception as e_fix:
            last_err = f"cpg-fix failed: {e_fix}"

    # 3) 그래도 실패 → pyogrio가 있으면 마지막 시도
    if not ok:
        try:
            import pyogrio
            try:
                _g = pyogrio.read_dataframe(str(f), encoding="CP949")
                ok = True; enc_used = "pyogrio:CP949"
            except Exception:
                _g = pyogrio.read_dataframe(str(f))  # pyogrio 기본 추정
                ok = True; enc_used = "pyogrio:auto"
        except Exception as e:
            last_err = f"pyogrio not usable: {e}"

    if ok:
        _g["source_file"] = f.stem
        gdfs.append(_g)
        read_logs.append({"file": str(f), "encoding": enc_used, "tried": tried})
    else:
        failed_logs.append({"file": str(f), "tried": tried, "error": last_err})

with st.expander("📄 셰이프 로딩 로그", expanded=False):
    if read_logs:
        st.write("정상 로드:")
        st.dataframe(pd.DataFrame(read_logs))
    if failed_logs:
        st.write("실패 파일:")
        st.dataframe(pd.DataFrame(failed_logs))

if not gdfs:
    st.error("❌ 모든 셰이프 파일 로딩 실패. (일부 파일의 DBF가 손상되었을 수 있습니다. QGIS로 열어 '다른 이름으로 저장' 후 재시도 권장)")
    st.stop()

# ── 병합/CRS ─────────────────────────────────────────────────────────────────────
gdf = pd.concat(gdfs, ignore_index=True)
gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdfs[0].crs if gdfs[0].crs else "EPSG:4326")
try:
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
except Exception as e:
    st.warning(f"CRS 변환 경고: {e}. EPSG:4326으로 강제 지정합니다.")
    gdf.set_crs(epsg=4326, inplace=True)

# ── name/lat/lon 보정 ────────────────────────────────────────────────────────────
name_candidates = [c for c in gdf.columns if str(c).lower() in ["name", "정류장명", "stop_name", "title"]]
if name_candidates:
    gdf["name"] = gdf[name_candidates[0]].astype(str)
else:
    obj_cols = [c for c in gdf.columns if c != "geometry" and gdf[c].dtype == "object"]
    if obj_cols:
        gdf["name"] = gdf[obj_cols[0]].astype(str)
    else:
        gdf["name"] = gdf.apply(lambda r: f"{r.get('source_file','drt')}_{int(r.name)+1}", axis=1)

if gdf.geometry.geom_type.isin(["Point"]).all():
    gdf["lon"] = gdf.geometry.x; gdf["lat"] = gdf.geometry.y
else:
    reps = gdf.geometry.representative_point()
    gdf["lon"] = reps.x; gdf["lat"] = reps.y

# ── 경계(boundary) ───────────────────────────────────────────────────────────────
boundary_path = Path("./cb_shp.shp")
if boundary_path.exists():
    try:
        boundary = gpd.read_file(boundary_path, encoding="cp949").to_crs(epsg=4326)
    except Exception:
        boundary = gpd.read_file(boundary_path).to_crs(epsg=4326)
else:
    try:
        union = gdf.unary_union; hull = union.convex_hull
        boundary = gpd.GeoDataFrame(geometry=[hull], crs="EPSG:4326")
    except Exception:
        boundary = None

# ── 세션 초기값 ──────────────────────────────────────────────────────────────────
for k, v in {
    "order": [], "segments": [], "duration": 0.0, "distance": 0.0,
    "messages": [{"role": "system", "content": "당신은 천안시에서 DRT(수요응답형 교통) 최적 노선을 추천해 주는 전문 교통 어시스턴트입니다."}],
    "auto_gpt_input": ""
}.items():
    if k not in st.session_state: st.session_state[k] = v

# ── 레이아웃/입력 ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1.5, 1.2, 3], gap="large")

with col1:
    st.markdown('<div class="section-header">🚐 DRT 노선 추천 설정</div>', unsafe_allow_html=True)
    st.caption("출발/경유 정류장을 선택하고 노선을 추천받으세요.")

    _ = st.radio("", ["차량(운행)", "도보(승객 접근)"], horizontal=True, key="mode_key", label_visibility="collapsed")
    _ = st.time_input("", value=datetime.time(9, 0), key="dep_time", label_visibility="collapsed")
    _ = st.selectbox("", ["혼잡(출퇴근)", "일반", "심야/한산"], index=1, key="time_band", label_visibility="collapsed")

    names_list = gdf["name"].dropna().astype(str).unique().tolist()
    _ = st.selectbox("", names_list, key="start_key", label_visibility="collapsed")
    _ = st.multiselect("", [n for n in names_list if n != st.session_state.get("start_key", "")],
                       key="wps_key", label_visibility="collapsed")

    c1, c2 = st.columns(2, gap="small")
    with c1: create_clicked = st.button("노선 추천")
    with c2: clear_clicked = st.button("초기화")

if clear_clicked:
    try:
        for k in ["segments", "order", "leg_durations"]: st.session_state[k] = []
        for k in ["duration", "distance"]: st.session_state[k] = 0.0
        st.session_state["auto_gpt_input"] = ""
        for widget_key in ["mode_key", "start_key", "wps_key", "dep_time", "time_band"]:
            if widget_key in st.session_state: del st.session_state[widget_key]
        st.success("✅ 초기화가 완료되었습니다."); st.rerun()
    except Exception as e:
        st.error(f"❌ 초기화 중 오류: {str(e)}")

with col2:
    st.markdown('<div class="section-header">📍 여행 방문 순서</div>', unsafe_allow_html=True)
    if st.session_state.get("order", []):
        for i, name in enumerate(st.session_state["order"], 1):
            st.markdown(f"<div class='visit-order-item'><div class='visit-number'>{i}</div><div>{name}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state">경로 생성 후 표시됩니다<br>🗺️</div>', unsafe_allow_html=True)

    if st.session_state.get("segments") and st.session_state.get("leg_durations"):
        try:
            start_dt = datetime.datetime.combine(datetime.date.today(), st.session_state.get("dep_time", datetime.time(9, 0)))
            times = [start_dt]
            for dmin in st.session_state["leg_durations"]:
                times.append(times[-1] + datetime.timedelta(minutes=float(dmin)))
            labels = st.session_state.get("order", [])[:]
            if len(labels) < len(times): labels += [f"목적지 {len(times)-len(labels)}"]
            labels = labels[:len(times)]
            st.markdown("**🕒 도착 예정 시간표**")
            st.dataframe(pd.DataFrame({"정류장": labels, "도착 예정 시각": [t.strftime("%H:%M") for t in times]}), use_container_width=True)
        except Exception as _e:
            st.warning(f"ETA 계산 중 경고: {str(_e)}")

    st.markdown("---")
    st.metric("⏱️ 소요시간", f"{st.session_state.get('duration', 0.0):.1f}분")
    st.metric("📏 이동거리", f"{st.session_state.get('distance', 0.0):.2f}km")

with col3:
    st.markdown('<div class="section-header">🗺️ 추천경로 지도시각화</div>', unsafe_allow_html=True)

    try:
        if boundary is not None:
            ctr = boundary.geometry.centroid
            clat, clon = float(ctr.y.mean()), float(ctr.x.mean())
        else:
            clat, clon = float(gdf["lat"].mean()), float(gdf["lon"].mean())
        if math.isnan(clat) or math.isnan(clon): clat, clon = 36.64, 127.48
    except Exception as e:
        st.warning(f"중심점 계산 오류: {str(e)}"); clat, clon = 36.64, 127.48

    try:
        G = ox.graph_from_point((clat, clon), dist=3000, network_type="all")
    except Exception as e:
        st.warning(f"도로 네트워크 로드 실패: {str(e)}")
        try:
            G = ox.graph_from_point((36.64, 127.48), dist=3000, network_type="all")
        except Exception:
            G = None

    if G is not None:
        try: edges = ox.graph_to_gdfs(G, nodes=False)
        except Exception as e: st.warning(f"엣지 변환 실패: {str(e)}"); edges = None
    else:
        edges = None

    stops = [st.session_state.get("start_key", "")] + st.session_state.get("wps_key", [])
    snapped = []
    try:
        for nm in stops:
            if not nm: continue
            rows = gdf[gdf["name"] == nm]
            if rows.empty: st.warning(f"⚠️ '{nm}' 정보를 찾을 수 없습니다."); continue
            r = rows.iloc[0]
            if pd.isna(r.lon) or pd.isna(r.lat): st.warning(f"⚠️ '{nm}'의 좌표 정보가 없습니다."); continue
            pt = Point(r.lon, r.lat)

            if edges is None or edges.empty:
                snapped.append((r.lon, r.lat)); continue

            try: edges_tmp = edges.to_crs(epsg=4326)
            except Exception: edges_tmp = edges

            # shapely >=2.0 거리 계산 안전 처리
            try:
                edges_tmp["d"] = edges_tmp.geometry.distance(pt)
            except Exception:
                # 좌표계 불일치나 GEOS 이슈시 원점 좌표 사용
                snapped.append((r.lon, r.lat)); continue

            if edges_tmp["d"].empty or edges_tmp["d"].isna().all():
                snapped.append((r.lon, r.lat)); continue

            ln = edges_tmp.loc[edges_tmp["d"].idxmin()]
            sp = ln.geometry.interpolate(ln.geometry.project(pt))
            snapped.append((sp.x, sp.y))
    except Exception as e:
        st.error(f"❌ 지점 처리 중 오류: {str(e)}")
        snapped = []
        for nm in stops:
            try:
                r = gdf[gdf["name"] == nm].iloc[0]
                if not (pd.isna(r.lon) or pd.isna(r.lat)): snapped.append((r.lon, r.lat))
            except Exception as coord_error:
                st.warning(f"⚠️ '{nm}' 좌표를 가져올 수 없습니다: {str(coord_error)}")

    if "mode_key" not in st.session_state: st.session_state["mode_key"] = "차량(운행)"

    if st.button("노선 추천 실행", key="run_hidden", help="상단 버튼과 동일", disabled=True):
        pass

    if create_clicked and len(snapped) >= 2:
        try:
            segs, total_sec, total_meter = [], 0.0, 0.0
            leg_durs_min = []
            api_mode = "walking" if "도보" in st.session_state.get("mode_key", "") else "driving"
            cong_map = {"혼잡(출퇴근)": 1.40, "일반": 1.15, "심야/한산": 0.90}
            cong = 1.0 if "도보" in st.session_state.get("mode_key","") else cong_map.get(st.session_state.get("time_band","일반"), 1.15)

            for i in range(len(snapped) - 1):
                x1, y1 = snapped[i]; x2, y2 = snapped[i + 1]
                coord = f"{x1},{y1};{x2},{y2}"
                url = f"https://api.mapbox.com/directions/v5/mapbox/{api_mode}/{coord}"
                params = {"geometries": "geojson", "overview": "full", "access_token": MAPBOX_TOKEN}
                try:
                    r = requests.get(url, params=params, timeout=10)
                    if r.status_code == 200:
                        data_resp = r.json()
                        if data_resp.get("routes"):
                            route = data_resp["routes"][0]
                            sec_base = float(route.get("duration", 0.0))
                            dist_m   = float(route.get("distance", 0.0))
                            sec_adj  = sec_base * (cong if api_mode == "driving" else 1.0)
                            segs.append(route["geometry"]["coordinates"])
                            total_sec += sec_adj; total_meter += dist_m; leg_durs_min.append(sec_adj / 60.0)
                        else:
                            st.warning(f"⚠️ 구간 {i+1}의 경로를 찾을 수 없습니다.")
                    else:
                        st.warning(f"⚠️ API 실패 (HTTP {r.status_code})")
                except requests.exceptions.Timeout:
                    st.warning("⚠️ API 호출 시간 초과")
                except Exception as api_error:
                    st.warning(f"⚠️ API 오류: {str(api_error)}")

            if segs:
                st.session_state["order"] = stops
                st.session_state["segments"] = segs
                st.session_state["duration"] = total_sec / 60.0
                st.session_state["distance"] = total_meter / 1000.0
                st.session_state["leg_durations"] = leg_durs_min

                st.success(f"✅ 경로 생성 완료 · 총 {st.session_state['distance']:.2f} km · 약 {st.session_state['duration']:.1f} 분 (보정 x{cong:.2f})")

                try:
                    start_dt = datetime.datetime.combine(datetime.date.today(), st.session_state.get("dep_time", datetime.time(9, 0)))
                    times = [start_dt]
                    for dmin in leg_durs_min: times.append(times[-1] + datetime.timedelta(minutes=float(dmin)))
                    labels = st.session_state["order"][:]
                    if len(labels) < len(times): labels += [f"목적지 {len(times)-len(labels)}"]
                    labels = labels[:len(times)]
                    st.dataframe(pd.DataFrame({"정류장": labels, "도착 예정 시각": [t.strftime("%H:%M") for t in times]}), use_container_width=True)
                except Exception as _e:
                    st.warning(f"ETA 표 생성 중 경고: {str(_e)}")
            else:
                st.error("❌ 모든 구간의 경로 생성에 실패했습니다.")
        except Exception as e:
            st.error(f"❌ 경로 생성 중 오류: {str(e)}")
            st.info("💡 다른 출발지/경유지를 선택해보세요.")

    # 지도 렌더링
    try:
        m = folium.Map(location=[clat, clon], zoom_start=12, tiles="CartoDB Positron",
                       prefer_canvas=True, control_scale=True)

        if boundary is not None:
            folium.GeoJson(boundary, style_function=lambda f: {
                "color": "#9aa0a6", "weight": 2, "dashArray": "4,4", "fillOpacity": 0.05
            }).add_to(m)

        mc = MarkerCluster().add_to(m)
        for _, row in gdf.iterrows():
            if not (pd.isna(row.lat) or pd.isna(row.lon)):
                folium.Marker([row.lat, row.lon],
                              popup=folium.Popup(str(row["name"]), max_width=200),
                              tooltip=str(row["name"]),
                              icon=folium.Icon(color="gray")).add_to(mc)

        current_order = st.session_state.get("order", [st.session_state.get("start_key", "")] + st.session_state.get("wps_key", []))
        for idx, (x, y) in enumerate(snapped, 1):
            place_name = current_order[idx - 1] if idx <= len(current_order) else f"지점 {idx}"
            folium.Marker([y, x], icon=folium.Icon(color="red", icon="flag"),
                          tooltip=f"{idx}. {place_name}",
                          popup=folium.Popup(f"<b>{idx}. {place_name}</b>", max_width=200)).add_to(m)

        if st.session_state.get("segments"):
            palette = ["#4285f4", "#34a853", "#ea4335", "#fbbc04", "#9c27b0", "#ff9800"]
            segments = st.session_state["segments"]
            used_positions, min_distance = [], 0.001
            for i, seg in enumerate(segments):
                if seg:
                    folium.PolyLine([(pt[1], pt[0]) for pt in seg],
                                    color=palette[i % len(palette)], weight=5, opacity=0.8).add_to(m)
                    mid = seg[len(seg)//2]
                    candidate_pos = [mid[1], mid[0]]
                    while any(abs(candidate_pos[0]-u[0]) < min_distance and abs(candidate_pos[1]-u[1]) < min_distance for u in used_positions):
                        candidate_pos[0] += min_distance * 0.5; candidate_pos[1] += min_distance * 0.5
                    folium.map.Marker(candidate_pos,
                        icon=DivIcon(html=f"<div style='background:{palette[i % len(palette)]};color:#fff;border-radius:50%;width:28px;height:28px;line-height:28px;text-align:center;font-weight:600;box-shadow:0 2px 4px rgba(0,0,0,0.3);'>{i+1}</div>")
                    ).add_to(m)
                    used_positions.append(candidate_pos)
            try:
                pts = [pt for seg in segments for pt in seg if seg]
                if pts:
                    m.fit_bounds([[min(p[1] for p in pts), min(p[0] for p in pts)],
                                  [max(p[1] for p in pts), max(p[0] for p in pts)]])
            except:
                m.location = [clat, clon]; m.zoom_start = 12
        else:
            m.location = [clat, clon]; m.zoom_start = 12

        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        _ = st_folium(m, width="100%", height=520, returned_objects=[], use_container_width=True, key="main_map")
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as map_error:
        st.error(f"❌ 지도 렌더링 오류: {str(map_error)}")
        st.markdown('<div class="map-container" style="display:flex;align-items:center;justify-content:center;color:#6b7280;">지도를 불러올 수 없습니다.</div>', unsafe_allow_html=True)
