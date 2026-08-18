"""
build_index.py
--------------
models.json 하나로 배포용 index.html을 생성하는 스크립트.

HTML 템플릿이 이 파일 안에 문자열로 들어 있어서, **이 .py 파일 하나만 있으면**
index.html을 언제든 다시 만들 수 있습니다. 별도의 템플릿 파일이 필요 없습니다.

    Release/
      build_index.py      <- 이 파일
      index.html          <- 생성물 (직접 편집하지 말 것)
    results/
      889_.../models.json <- 입력 (run.export_all_models_json의 결과)

사용:
    python build_index.py
        -> ./models.json 또는 ../results/*/models.json 을 자동으로 찾아 index.html 생성

    python build_index.py --json ../results/889_amyloid_beta_prob_pipeline/models.json
    python build_index.py --out docs/index.html
    python build_index.py --verify ../results/889_.../golden_M4_SCD_MCI.csv
        -> JSON 계수로 확률을 다시 계산해 python_prob과 일치하는지 확인

계수가 index.html에 들어가는 방식 (두 경로를 모두 지원):
    1) 페이지가 같은 폴더의 models.json을 fetch -> 성공하면 그것을 사용 (항상 최신)
    2) 실패하면 HTML에 내장된 사본으로 폴백

    GitHub Pages(https://)에서는 1번이 동작하므로, 모델을 다시 학습했을 때
    docs/models.json만 교체하면 사이트에 즉시 반영됩니다. HTML 재빌드가 필요 없습니다.
    반대로 파일 하나만 받아 더블클릭으로 여는 경우(file://)는 CORS 정책상 fetch가
    막히므로 2번으로 넘어가고, 그래도 계산은 정상 동작합니다.

    그래서 이 스크립트는 index.html을 만들면서 그 옆에 models.json 사본도 함께 둡니다.
    (--no-copy 로 끌 수 있습니다)

화면(HTML/CSS/JS)을 고치고 싶으면 아래 TEMPLATE 문자열을 수정한 뒤 다시 실행하세요.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLACEHOLDER = "__MODEL_JSON__"

TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Amyloid Beta 양성 확률 예측</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js"></script>
<style>
:root{
  color-scheme:light;
  --page:#f9f9f7; --surface:#fcfcfb; --surface-2:#f3f3f0;
  --ink:#0b0b0b; --ink-2:#52514e; --ink-muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --accent:#2a78d6; --accent-soft:#cde2fb;
  --pos:#e34948; --neg:#2a78d6;
  --warn:#fab219; --crit:#d03b3b; --good:#0ca30c;
  --seq-400:#3987e5;
}
@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])){
    color-scheme:dark;
    --page:#0d0d0d; --surface:#1a1a19; --surface-2:#222221;
    --ink:#fff; --ink-2:#c3c2b7; --ink-muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
    --accent:#3987e5; --accent-soft:#184f95;
    --pos:#e66767; --neg:#3987e5;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#0d0d0d; --surface:#1a1a19; --surface-2:#222221;
  --ink:#fff; --ink-2:#c3c2b7; --ink-muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --accent:#3987e5; --accent-soft:#184f95;
  --pos:#e66767; --neg:#3987e5;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--page); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif;
  font-size:14px; line-height:1.55;
}
header{
  position:sticky; top:0; z-index:50; background:var(--surface);
  border-bottom:1px solid var(--border); padding:12px 20px;
  display:flex; align-items:center; gap:16px; flex-wrap:wrap;
}
header h1{font-size:16px; margin:0; font-weight:650; letter-spacing:-.01em}
header .sub{color:var(--ink-muted); font-size:12px}
main{max-width:1500px; margin:0 auto; padding:20px}
.card{
  background:var(--surface); border:1px solid var(--border);
  border-radius:10px; padding:18px 20px; margin-bottom:16px;
}
.card > h2{
  font-size:13px; margin:0 0 4px; font-weight:650;
  text-transform:uppercase; letter-spacing:.06em; color:var(--ink-2);
}
.card > .hint{color:var(--ink-muted); font-size:12px; margin:0 0 14px}
button{
  font:inherit; font-size:13px; padding:7px 14px; border-radius:7px; cursor:pointer;
  border:1px solid var(--border); background:var(--surface-2); color:var(--ink);
}
button:hover{background:var(--grid)}
button.primary{background:var(--accent); border-color:transparent; color:#fff; font-weight:600}
button.primary:hover{filter:brightness(1.08)}
button:disabled{opacity:.45; cursor:not-allowed}
button.icon{padding:6px 9px; font-size:12px}
select{
  font:inherit; font-size:13px; padding:5px 8px; border-radius:6px;
  border:1px solid var(--axis); background:var(--surface); color:var(--ink); max-width:100%;
}
.row{display:flex; gap:10px; align-items:center; flex-wrap:wrap}
.spacer{flex:1}
.muted{color:var(--ink-muted)}
.note{
  border-left:3px solid var(--warn); background:var(--surface-2);
  padding:10px 14px; border-radius:0 8px 8px 0; font-size:12.5px; color:var(--ink-2);
}
.note.bad{border-left-color:var(--crit)}
.note.ok{border-left-color:var(--good)}
.banner{
  display:none; border-left:3px solid var(--crit); background:var(--surface-2);
  padding:10px 14px; border-radius:0 8px 8px 0; font-size:13px; margin-bottom:14px;
}
.banner.on{display:block}
table{border-collapse:collapse; width:100%; font-size:13px}
th,td{padding:7px 10px; text-align:left; border-bottom:1px solid var(--grid); vertical-align:middle; white-space:nowrap}
th{
  font-size:11px; letter-spacing:.02em; color:var(--ink-muted); font-weight:600;
  position:sticky; top:0; background:var(--surface); z-index:2;
}
td.num,th.num{text-align:right; font-variant-numeric:tabular-nums}
tbody tr:hover{background:var(--surface-2)}
.scroll{overflow:auto; max-height:min(64vh,680px); border:1px solid var(--border); border-radius:8px}
.badge{
  display:inline-block; padding:2px 8px; border-radius:999px; font-size:11.5px;
  font-weight:600; background:var(--accent-soft); color:var(--accent);
  border:1px solid transparent; white-space:nowrap;
}
.badge.blocked{background:transparent; color:var(--crit); border-color:var(--crit)}
.tag{font-size:11px; padding:1px 6px; border-radius:4px; background:var(--surface-2);
     color:var(--ink-muted); border:1px solid var(--border); font-weight:500}
.meter{position:relative; height:8px; background:var(--grid); border-radius:4px; overflow:hidden; width:88px}
.meter > i{display:block; height:100%; border-radius:0 4px 4px 0; background:var(--seq-400)}
.pcell{display:flex; align-items:center; gap:9px; justify-content:flex-end}
.pcell b{font-variant-numeric:tabular-nums; font-weight:650; min-width:50px; text-align:right}
.drawer-bg{display:none; position:fixed; inset:0; background:rgba(0,0,0,.34); z-index:100}
.drawer-bg.on{display:block}
.drawer{
  position:fixed; top:0; right:0; height:100%; width:min(760px,96vw); z-index:101;
  background:var(--surface); border-left:1px solid var(--border);
  transform:translateX(100%); transition:transform .18s ease; overflow-y:auto;
}
.drawer.on{transform:none}
.drawer .dhead{
  position:sticky; top:0; background:var(--surface); border-bottom:1px solid var(--border);
  padding:14px 20px; display:flex; align-items:center; gap:12px;
}
.drawer .dbody{padding:18px 20px 40px}
.hero{display:flex; align-items:baseline; gap:12px; margin:6px 0 2px}
.hero .val{font-size:44px; font-weight:680; letter-spacing:-.02em; line-height:1}
.hero .unit{font-size:15px; color:var(--ink-2)}
.bigmeter{position:relative; height:14px; background:var(--grid); border-radius:7px; margin:14px 0 6px; overflow:hidden}
.bigmeter > i{display:block; height:100%; border-radius:0 7px 7px 0; background:var(--seq-400)}
.bigmeter .tick{position:absolute; top:0; bottom:0; width:1px; background:var(--surface); opacity:.85}
.scalerow{display:flex; justify-content:space-between; font-size:11px; color:var(--ink-muted); font-variant-numeric:tabular-nums}
pre.formula{
  background:var(--surface-2); border:1px solid var(--border); border-radius:8px;
  padding:12px 14px; font-size:11.5px; line-height:1.5; overflow-x:auto;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; color:var(--ink-2); max-height:260px;
}
details.fold > summary{cursor:pointer; font-size:12.5px; color:var(--ink-2); margin:16px 0 8px; font-weight:600}
tr.topk{background:color-mix(in srgb,var(--accent) 8%, transparent)}
tr.topk td:first-child{box-shadow:inset 3px 0 0 var(--accent)}
.star{color:var(--accent); font-weight:700}
.forest{position:relative; width:132px; height:14px}
.forest .ref{position:absolute; top:0; bottom:0; width:1px; background:var(--axis)}
.forest .ci{position:absolute; top:6px; height:2px; border-radius:1px}
.forest .pt{position:absolute; top:2.5px; width:9px; height:9px; border-radius:50%; margin-left:-4.5px; box-shadow:0 0 0 2px var(--surface)}
.legend{display:flex; gap:14px; align-items:center; font-size:11.5px; color:var(--ink-2); margin:8px 0 2px}
.legend i{display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:-1px}
.kv{display:grid; grid-template-columns:auto 1fr; gap:4px 14px; font-size:12.5px; color:var(--ink-2)}
.kv b{color:var(--ink); font-weight:600; font-variant-numeric:tabular-nums}
.stepnum{
  display:inline-flex; width:20px; height:20px; border-radius:50%; background:var(--accent); color:#fff;
  align-items:center; justify-content:center; font-size:11px; font-weight:700; margin-right:8px;
}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px;
     background:var(--surface-2); padding:1px 5px; border-radius:4px}
</style>
</head>
<body>

<header>
  <h1>Amyloid Beta 양성 확률 예측</h1>
  <span class="sub">SCST 기반 로지스틱 회귀 · 브라우저 내 계산</span>
  <span class="spacer"></span>
  <span class="tag" id="srcTag">계수 로드 중…</span>
  <button class="icon" id="themeBtn" title="라이트/다크 전환">◐ 테마</button>
</header>

<main>

  <div class="card">
    <div class="note">
      <b>연구용 참고 도구이며 진단 목적이 아닙니다.</b>
      모든 계산은 이 브라우저 안에서만 수행되며, 업로드한 파일과 환자 정보는 어디에도 전송·저장되지 않습니다.
      페이지를 새로고침하면 모든 데이터가 사라집니다.
    </div>
  </div>

  <div class="card" id="libCard" style="display:none">
    <div class="note bad">
      <b>Excel 처리 라이브러리를 불러오지 못했습니다.</b>
      인터넷이 차단된 환경일 수 있습니다. Excel 업로드와 다운로드가 불가능합니다.
    </div>
  </div>

  <div class="card">
    <h2><span class="stepnum">1</span>SCST 결과 파일 업로드</h2>
    <p class="hint">SCST 웹에서 내려받은 Excel(.xlsx)을 그대로 올리세요. <code>COMMON</code> 시트를 읽습니다. 파일은 전송되지 않습니다.</p>
    <div class="row">
      <input type="file" id="file" accept=".xlsx,.xls,.csv">
      <button id="demoBtn">예시 데이터로 시험</button>
      <span id="fileInfo" class="muted"></span>
    </div>
    <div class="row" id="sheetRow" style="display:none; margin-top:10px">
      <label class="muted">시트</label>
      <select id="sheetSel"></select>
    </div>
    <div id="checkBox" style="margin-top:14px"></div>
  </div>

  <div class="card" id="tableCard" style="display:none">
    <h2><span class="stepnum">2</span>환자별 정보 입력 및 결과</h2>
    <p class="hint">
      <code>cognitive_status</code>와 <code>APOE</code>는 수동 입력 항목입니다. 정보가 없으면 <b>Unknown</b>으로 두세요.
      입력할수록 상위 모델이 자동 선택됩니다. 검사 순서상 <b>인지기능검사 → 혈액검사</b>이므로,
      <code>cognitive_status</code> 없이 <code>APOE</code>만 입력할 수는 없습니다.
    </p>
    <div class="banner" id="orderBanner"></div>
    <div class="row" style="margin-bottom:12px">
      <label class="muted">전체 일괄 적용</label>
      <select id="bulkStatus"></select>
      <select id="bulkApoe"></select>
      <span class="spacer"></span>
      <button class="primary" id="dlBtn">Excel 다운로드</button>
    </div>
    <div class="scroll"><table id="tbl"><thead></thead><tbody></tbody></table></div>
    <p class="hint" style="margin-top:12px">행을 클릭하면 적용된 모델의 수식과 계수 표를 볼 수 있습니다.</p>
  </div>

</main>

<div class="drawer-bg" id="drawerBg"></div>
<aside class="drawer" id="drawer">
  <div class="dhead">
    <b id="dTitle">환자</b>
    <span class="badge" id="dModel"></span>
    <span class="spacer"></span>
    <button class="icon" id="closeDrawer">닫기 ✕</button>
  </div>
  <div class="dbody" id="dBody"></div>
</aside>

<script id="model-data" type="application/json">__MODEL_JSON__</script>
<script>
"use strict";

// 계수는 두 경로로 들어옵니다.
//   1) 같은 폴더의 models.json을 fetch  -> 성공하면 이것을 사용 (항상 최신)
//   2) 실패하면 이 파일에 내장된 사본으로 폴백
// GitHub Pages(https://)에서는 1번이 동작하므로 models.json만 갈아끼우면 사이트가 갱신됩니다.
// 파일을 받아 더블클릭으로 여는 경우(file://)는 CORS 정책상 fetch가 막혀 2번으로 넘어갑니다.
let DATA, MODELS, COMPOSITES, SEX_COL, SEX_POS, MODEL_SRC;

// 출력 Excel과 동일한 컬럼 구성. 대시보드도 이 이름을 그대로 씁니다.
const INFO_COLS = ['SCST_DATE','user_ID','NAME','Institution','sex','AGE','education_year'];
const OUT_HEADER = [...INFO_COLS, 'cognitive_status', 'APOE', 'model_used',
                    'P_amyloid_beta_positive_raw', 'P_amyloid_beta_positive_percent'];

// 모델 계산에 반드시 필요한 원본 컬럼 (이름이 정확히 일치해야 함)
const REQUIRED_COLS = () => [...COMPOSITES, 'sex', 'AGE', 'education_year'];

const COG_OPTIONS  = ['SCD','MCI','Dementia'];
const APOE_GENOTYPES = ['E2/E2','E2/E3','E2/E4','E3/E3','E3/E4','E4/E4'];

/* ---------- 상태 ---------- */
let WB = null, FILENAME = '', SHEET = '';
let RAW = [], HEADERS = [], PATIENTS = [], MISSING_COLS = [];

/* ---------- 유틸 ---------- */
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const num = v => { const n = parseFloat(String(v ?? '').replace(/,/g,'')); return Number.isFinite(n) ? n : null; };
const show = v => (v === null || v === undefined || v === '') ? '' : esc(v);

/* ---------- APOE 유전형 -> e4 carrier ---------- */
function apoeCarrier(v){
  if (!v) return null;
  return /E4/i.test(v) ? 1 : 0;
}

/* ---------- 모델 선택 규칙 ---------- */
// 정보가 많을수록 상위 모델. 인지검사 -> 혈액검사 순서를 강제.
function selectModel(status, apoeVal){
  const apoe = apoeCarrier(apoeVal);
  const hasStatus = !!status, hasApoe = (apoe === 0 || apoe === 1);
  if (!hasStatus && hasApoe) return {blocked:true};
  const grp = status === 'Dementia' ? 'Dementia' : 'SCD_MCI';
  if (!hasStatus) return {name:'M2_All'};
  if (!hasApoe)   return {name:`M2_${grp}`};
  return {name: grp === 'Dementia' ? 'M3_Dementia' : 'M4_SCD_MCI'};
}

/* ---------- 확률 계산 (HTML이 하는 일의 전부) ---------- */
function featureValue(name, row, status, apoe){
  if (name === SEX_COL)           return String(row['sex'] ?? '').trim() === SEX_POS ? 1 : 0;
  if (name === 'is_MCI')          return status === 'MCI' ? 1 : 0;
  if (name === 'is_Dementia')     return status === 'Dementia' ? 1 : 0;
  if (name === 'APOE_e4_carrier') return apoeCarrier(apoe) === 1 ? 1 : 0;
  return num(row[name]);
}

function predict(modelName, row, status, apoe){
  const m = MODELS[modelName];
  let logit = m.intercept; const missing = [];
  for (const [k, coef] of Object.entries(m.coefficients)){
    const x = featureValue(k, row, status, apoe);
    if (x === null){ missing.push(k); continue; }
    logit += coef * x;
  }
  if (missing.length) return {missing};
  return {p: 1 / (1 + Math.exp(-logit))};
}

function rowResult(p){
  const sel = selectModel(p.status, p.apoe);
  if (sel.blocked) return {blocked:true};
  return {model:sel.name, ...predict(sel.name, RAW[p.idx], p.status, p.apoe)};
}

/* ---------- 파일 읽기 ---------- */
// 컬럼 매핑 UI는 두지 않습니다. SCST 원본의 컬럼명을 그대로 요구하고,
// 없으면 무엇이 없는지 알려주고 계산하지 않습니다.
const PREFERRED_SHEETS = ['COMMON','common','Sheet1'];

function loadRows(rows, sheetName){
  RAW = rows;
  HEADERS = rows.length ? Object.keys(rows[0]) : [];
  SHEET = sheetName || '';
  PATIENTS = rows.map((_, i) => ({idx:i, status:'', apoe:null}));
  MISSING_COLS = REQUIRED_COLS().filter(c => !HEADERS.includes(c));

  renderCheck();
  $('#tableCard').style.display = MISSING_COLS.length ? 'none' : '';
  if (!MISSING_COLS.length) render();
}

function renderCheck(){
  const box = $('#checkBox');
  if (!RAW.length){ box.innerHTML = '<div class="note bad">데이터 행이 없습니다.</div>'; return; }

  if (MISSING_COLS.length){
    box.innerHTML = `<div class="note bad">
      <b>필수 컬럼 ${MISSING_COLS.length}개가 없어 확률을 산출할 수 없습니다.</b><br>
      <span style="display:inline-block;margin-top:6px">${MISSING_COLS.map(c => `<code>${esc(c)}</code>`).join(' ')}</span>
      <br><span class="muted" style="font-size:11.5px">SCST 원본의 <code>COMMON</code> 시트를 올렸는지, 시트 선택이 맞는지 확인하세요.</span>
    </div>`;
    return;
  }

  const badSex = [...new Set(RAW.map(r => String(r['sex'] ?? '').trim()))]
                   .filter(v => v && v !== '남성' && v !== '여성');
  const noInfo = INFO_COLS.filter(c => !HEADERS.includes(c));

  box.innerHTML = `<div class="note ok">
    <b>필수 컬럼 ${REQUIRED_COLS().length}개 확인 완료</b> · ${RAW.length}명
    ${noInfo.length ? `<br><span class="muted" style="font-size:11.5px">표시용 컬럼 없음(계산에는 영향 없음): ${noInfo.map(c=>`<code>${esc(c)}</code>`).join(' ')}</span>` : ''}
    ${badSex.length ? `<br><span style="color:var(--crit)">sex 값에 <code>${badSex.map(esc).join('</code> <code>')}</code> 가 있습니다 — 남성/여성이 아니면 남성으로 처리됩니다.</span>` : ''}
  </div>`;
}

function readSheet(name){
  const rows = XLSX.utils.sheet_to_json(WB.Sheets[name], {defval:null});
  $('#fileInfo').textContent = `${FILENAME} · [${name}] ${rows.length}명`;
  loadRows(rows, name);
}

$('#file').onchange = e => {
  const f = e.target.files[0]; if (!f) return;
  FILENAME = f.name;
  const rd = new FileReader();
  rd.onload = ev => {
    WB = XLSX.read(new Uint8Array(ev.target.result), {type:'array'});
    const names = WB.SheetNames;
    const pick = PREFERRED_SHEETS.find(n => names.includes(n)) ?? names[0];
    const sel = $('#sheetSel');
    sel.innerHTML = names.map(n => `<option${n===pick?' selected':''}>${esc(n)}</option>`).join('');
    sel.onchange = () => readSheet(sel.value);
    $('#sheetRow').style.display = names.length > 1 ? '' : 'none';
    readSheet(pick);
  };
  rd.readAsArrayBuffer(f);
};

// 예시 데이터: 실제 SCST COMMON 시트와 "동일한 컬럼명"으로 8명을 만들어 냅니다.
// 실데이터 없이 화면 동작을 확인하거나 시연 영상을 찍을 때 사용합니다.
$('#demoBtn').onclick = () => {
  const rows = [];
  for (let i = 0; i < 8; i++){
    const r = {
      SCST_DATE: '2026-07-21',
      user_ID: `DEMO-${String(i+1).padStart(3,'0')}`,
      NAME: `예시환자${i+1}`,
      Institution: '예시기관',
      sex: i % 2 ? '여성' : '남성',
      AGE: 60 + i * 3,
      education_year: [6,9,12,12,14,16,16,18][i],
    };
    COMPOSITES.forEach((c, j) => { r[c] = +((Math.sin((i+1)*(j+2)) * 1.3) - i*0.15).toFixed(2); });
    rows.push(r);
  }
  FILENAME = '예시 데이터';
  $('#fileInfo').textContent = `예시 데이터 · ${rows.length}명`;
  $('#sheetRow').style.display = 'none';
  loadRows(rows, 'DEMO');
};

/* ---------- 결과 표 (출력 Excel과 동일한 컬럼 구성) ---------- */
function render(){
  const cogOpts = v => ['', ...COG_OPTIONS]
    .map(o => `<option value="${o}"${o===v?' selected':''}>${o||'Unknown'}</option>`).join('');
  const apoeOpts = v => ['', ...APOE_GENOTYPES]
    .map(o => `<option value="${o}"${(v ?? '')===o?' selected':''}>${o||'Unknown'}</option>`).join('');

  $('#tbl').querySelector('thead').innerHTML = '<tr><th style="width:34px"></th>'
    + INFO_COLS.map(c => `<th class="${c==='AGE'||c==='education_year'?'num':''}">${c}</th>`).join('')
    + '<th style="width:130px">cognitive_status <span class="tag">수동</span></th>'
    + '<th style="width:140px">APOE <span class="tag">수동</span></th>'
    + '<th>model_used</th>'
    + '<th class="num">P_amyloid_beta_positive_raw</th>'
    + '<th class="num" style="width:170px">P_amyloid_beta_positive_percent</th></tr>';

  let blocked = 0;
  $('#tbl').querySelector('tbody').innerHTML = PATIENTS.map((p, i) => {
    const row = RAW[p.idx], res = rowResult(p);
    let cModel, cRaw, cPct;
    if (res.blocked){
      blocked++;
      cModel = '<span class="badge blocked">⚠ cognitive_status 필요</span>';
      cRaw = ''; cPct = '<span class="muted">인지기능검사 결과를 먼저 입력하세요</span>';
    } else if (res.missing){
      cModel = `<span class="badge">${res.model}</span>`;
      cRaw = ''; cPct = `<span class="muted">값 없음 (${res.missing.length}개)</span>`;
    } else {
      const pct = Math.max(0, Math.min(1, res.p));
      cModel = `<span class="badge">${res.model}</span>`;
      cRaw = res.p.toFixed(6);
      cPct = `<div class="pcell"><div class="meter"><i style="width:${(pct*100).toFixed(1)}%"></i></div>`
           + `<b>${(res.p*100).toFixed(1)}%</b></div>`;
    }
    return `<tr data-i="${i}"><td class="muted num">${i+1}</td>`
      + INFO_COLS.map(c => `<td class="${c==='AGE'||c==='education_year'?'num':''}">${show(row[c])}</td>`).join('')
      + `<td><select class="inp" data-i="${i}" data-k="status">${cogOpts(p.status)}</select></td>`
      + `<td><select class="inp" data-i="${i}" data-k="apoe">${apoeOpts(p.apoe)}</select></td>`
      + `<td>${cModel}</td><td class="num">${cRaw}</td><td class="num">${cPct}</td></tr>`;
  }).join('');

  $('#tbl').querySelectorAll('select.inp').forEach(sel => {
    sel.onclick = e => e.stopPropagation();
    sel.onchange = () => {
      const p = PATIENTS[+sel.dataset.i];
      if (sel.dataset.k === 'status') p.status = sel.value;
      else p.apoe = sel.value || null;
      render();
    };
  });
  $('#tbl').querySelectorAll('tbody tr').forEach(tr => { tr.onclick = () => openDrawer(+tr.dataset.i); });

  const b = $('#orderBanner');
  b.classList.toggle('on', blocked > 0);
  if (blocked) b.innerHTML = `<b>${blocked}명</b>이 APOE만 입력된 상태입니다.
    인지기능검사 → 혈액검사 순서이므로, 해당 환자의 <b>cognitive_status를 먼저 입력</b>해야 확률이 계산됩니다.`;
}

/* ---------- 일괄 적용 ---------- */
$('#bulkStatus').innerHTML = '<option value="">cognitive_status…</option>'
  + COG_OPTIONS.map(o => `<option>${o}</option>`).join('')
  + '<option value="__none__">Unknown</option>';
$('#bulkApoe').innerHTML = '<option value="">APOE…</option>'
  + APOE_GENOTYPES.map(g => `<option>${g}</option>`).join('')
  + '<option value="__none__">Unknown</option>';

$('#bulkStatus').onchange = e => {
  const v = e.target.value; if (!v) return;
  PATIENTS.forEach(p => p.status = (v === '__none__' ? '' : v));
  e.target.value = ''; render();
};
$('#bulkApoe').onchange = e => {
  const v = e.target.value; if (!v) return;
  PATIENTS.forEach(p => p.apoe = (v === '__none__' ? null : v));
  e.target.value = ''; render();
};

/* ---------- 상세 패널 ---------- */
function forestCell(r, lo, hi){
  const L = Math.log, clamp = v => Math.max(lo, Math.min(hi, v));
  const x = v => ((L(clamp(v)) - L(lo)) / (L(hi) - L(lo))) * 100;
  const c = r.OR >= 1 ? 'var(--pos)' : 'var(--neg)';
  const op = r.is_significant ? 1 : .4;
  return `<div class="forest" title="OR ${r.OR.toFixed(3)} (${r.ci_lower.toFixed(3)}–${r.ci_upper.toFixed(3)})">
    <div class="ref" style="left:${x(1)}%"></div>
    <div class="ci" style="left:${x(r.ci_lower)}%; width:${Math.max(1,x(r.ci_upper)-x(r.ci_lower))}%; background:${c}; opacity:${op}"></div>
    <div class="pt" style="left:${x(r.OR)}%; background:${c}; opacity:${op}"></div>
  </div>`;
}

function openDrawer(i){
  const p = PATIENTS[i], row = RAW[p.idx], res = rowResult(p);
  $('#dTitle').textContent = row['user_ID'] ?? `#${i+1}`;

  if (res.blocked || res.missing){
    $('#dModel').textContent = res.model || '—';
    $('#dBody').innerHTML = `<p class="muted">${res.blocked
      ? 'cognitive_status를 먼저 입력해야 모델이 선택됩니다.'
      : '다음 값이 없어 계산할 수 없습니다:<br>' + res.missing.map(esc).join(', ')}</p>`;
    showDrawer(); return;
  }

  const m = MODELS[res.model];
  $('#dModel').textContent = res.model;
  const rows = m.or_table;
  const lo = Math.max(1e-3, Math.min(...rows.map(r => r.ci_lower)) * 0.9);
  const hi = Math.min(1e3,  Math.max(...rows.map(r => r.ci_upper)) * 1.1);
  const topk = rows.filter(r => r.top_k);

  $('#dBody').innerHTML = `
    <div class="hero"><span class="val">${(res.p*100).toFixed(1)}%</span>
      <span class="unit">P_amyloid_beta_positive_percent</span></div>
    <div class="bigmeter"><i style="width:${(Math.max(0,Math.min(1,res.p))*100).toFixed(1)}%"></i>
      ${[25,50,75].map(t=>`<span class="tick" style="left:${t}%"></span>`).join('')}</div>
    <div class="scalerow"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>

    <div class="kv" style="margin-top:18px">
      <span>P_amyloid_beta_positive_raw</span><b>${res.p.toFixed(6)}</b>
      <span>model_used</span><b>${res.model}</b>
      <span>cognitive_status</span><b>${p.status || 'Unknown'}</b>
      <span>APOE</span><b>${p.apoe ? esc(p.apoe) + (apoeCarrier(p.apoe) ? ' (e4 보유)' : ' (e4 비보유)') : 'Unknown'}</b>
      <span>학습 표본</span><b>n = ${m.n} (양성 ${m.n_events}, ${(m.prevalence*100).toFixed(1)}%)</b>
      <span>교차검증 AUC</span><b>${m.cv_auc_mean?.toFixed(3) ?? '—'} ± ${m.cv_auc_std?.toFixed(3) ?? '—'}</b>
    </div>

    ${topk.length ? `<p class="muted" style="font-size:12.5px;margin:18px 0 0">
      주요 변수 Top-${topk.length} — ${esc(m.top_k_rule)}. 아래 표에서 <span class="star">★</span> 표시된 행입니다.</p>` : ''}

    <details class="fold" open><summary>계수별 Odds Ratio</summary>
    <div class="legend">
      <span><i style="background:var(--pos)"></i>OR &gt; 1 · 양성 확률 증가</span>
      <span><i style="background:var(--neg)"></i>OR &lt; 1 · 양성 확률 감소</span>
      <span class="muted">흐린 표시 = p ≥ 0.05</span>
    </div>
    <div class="scroll" style="max-height:420px"><table>
      <thead><tr><th style="width:20px"></th><th>변수</th><th class="num">β</th>
        <th class="num">OR (95% CI)</th><th style="width:140px">　</th><th class="num">p</th></tr></thead>
      <tbody>${rows.map(r => `<tr class="${r.top_k?'topk':''}">
        <td class="star">${r.top_k?'★':''}</td>
        <td>${esc(r.feature.replace(/_z_score$/,''))}</td>
        <td class="num">${r.beta.toFixed(3)}</td>
        <td class="num">${r.OR.toFixed(3)} <span class="muted">(${r.ci_lower.toFixed(2)}–${r.ci_upper.toFixed(2)})</span></td>
        <td>${forestCell(r, lo, hi)}</td>
        <td class="num">${r.p_value < 0.001 ? '&lt;0.001' : r.p_value.toFixed(3)}</td>
      </tr>`).join('')}</tbody>
    </table></div>
    <p class="muted" style="font-size:11.5px;margin-top:8px">
      표시된 신뢰구간과 p-value는 <b>모델 계수</b>에 대한 것이며, 개별 환자 예측확률의 불확실성이 아닙니다.
      다중비교는 보정하지 않았습니다(탐색적 해석).</p>
    </details>

    <details class="fold"><summary>적용된 회귀식</summary>
      <pre class="formula">${esc(m.formula)}</pre></details>`;
  showDrawer();
}
function showDrawer(){ $('#drawer').classList.add('on'); $('#drawerBg').classList.add('on'); }
function hideDrawer(){ $('#drawer').classList.remove('on'); $('#drawerBg').classList.remove('on'); }
$('#closeDrawer').onclick = hideDrawer;
$('#drawerBg').onclick = hideDrawer;
document.addEventListener('keydown', e => { if (e.key === 'Escape') hideDrawer(); });

/* ---------- Excel 다운로드 (화면 표와 동일한 컬럼) ---------- */
$('#dlBtn').onclick = () => {
  const out = PATIENTS.map(p => {
    const row = RAW[p.idx], res = rowResult(p);
    const done = !res.blocked && !res.missing;
    const rec = {};
    INFO_COLS.forEach(c => { rec[c] = (c in row) ? (row[c] ?? null) : null; });
    rec['cognitive_status'] = p.status || null;
    rec['APOE'] = p.apoe || null;
    rec['model_used'] = res.blocked ? null : res.model;
    rec['P_amyloid_beta_positive_raw']     = done ? +res.p.toFixed(6) : null;
    rec['P_amyloid_beta_positive_percent'] = done ? +(res.p * 100).toFixed(1) : null;
    return rec;
  });

  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(out, {header:OUT_HEADER}), 'result');

  const used = [...new Set(out.map(r => r['model_used']).filter(Boolean))];
  const meta = [
    {item:'generated_at', value:new Date().toLocaleString('ko-KR')},
    {item:'source_file', value:FILENAME || '(직접 입력)'},
    {item:'sheet', value:SHEET},
    {item:'n_patients', value:PATIENTS.length},
    ...used.map(nm => ({item:`model ${nm}`,
      value:`train n=${MODELS[nm].n} (positive ${MODELS[nm].n_events}, ${(MODELS[nm].prevalence*100).toFixed(1)}%), `
           + `CV AUC ${MODELS[nm].cv_auc_mean?.toFixed(3) ?? '—'}`})),
    {item:'notice', value:'연구용 참고 도구이며 진단 목적이 아닙니다.'},
  ];
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(meta), 'model_info');

  const d = new Date(), pad = n => String(n).padStart(2,'0');
  XLSX.writeFile(wb, `amyloid_prob_${d.getFullYear()}${pad(d.getMonth()+1)}${pad(d.getDate())}.xlsx`);
};

/* ---------- 계수 로드 ---------- */
$('#file').disabled = true;
$('#demoBtn').disabled = true;

(async function init(){
  const inline = JSON.parse(document.getElementById('model-data').textContent);
  let data = inline, src = 'inline';
  try {
    const r = await fetch('models.json', {cache:'no-store'});
    if (r.ok){
      const j = await r.json();
      if (j && j.models && Object.keys(j.models).length){ data = j; src = 'models.json'; }
    }
  } catch (e) { /* file:// 또는 파일 없음 -> 내장본 사용 */ }

  DATA = data;
  MODELS = data.models;
  COMPOSITES = data.encoding.composite_scores.columns;
  SEX_COL = data.encoding.sex.column;              // 예: sex_여성
  SEX_POS = SEX_COL.replace(/^sex_/, '');          // 예: 여성
  MODEL_SRC = src;

  $('#srcTag').textContent = (src === 'models.json' ? '계수 출처: models.json' : '계수 출처: 내장본')
    + ` · 모델 ${Object.keys(MODELS).length}개`;

  $('#demoBtn').disabled = false;
  if (typeof XLSX !== 'undefined') $('#file').disabled = false;
})();

if (typeof XLSX === 'undefined'){
  $('#libCard').style.display = '';
  $('#dlBtn').disabled = true;
}

$('#themeBtn').onclick = () => {
  const cur = document.documentElement.getAttribute('data-theme');
  document.documentElement.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
};
</script>
</body>
</html>
"""


# ============================================================
# models.json 찾기
# ============================================================

def find_models_json(explicit=None):
    """
    --json으로 명시하면 그 경로를 쓰고, 아니면 아래 순서로 자동 탐색:
        1) 이 스크립트와 같은 폴더의 models.json
        2) ../results/*/models.json
        3) ../../results/*/models.json
    후보가 여러 개면 가장 최근에 수정된 것을 고르고, 무엇을 골랐는지 출력합니다.
    """
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (HERE / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"models.json을 찾을 수 없습니다: {p}")
        return p

    found = []
    same = HERE / "models.json"
    if same.exists():
        found.append(same)
    for up in (HERE.parent, HERE.parent.parent):
        results = up / "results"
        if results.is_dir():
            found.extend(sorted(results.glob("*/models.json")))

    found = list(dict.fromkeys(found))
    if not found:
        raise FileNotFoundError(
            "models.json을 자동으로 찾지 못했습니다.\n"
            f"  탐색한 위치: {HERE}/models.json, {HERE.parent}/results/*/models.json\n"
            "  --json 옵션으로 경로를 직접 지정하세요.\n"
            "  예) python build_index.py --json ../results/889_amyloid_beta_prob_pipeline/models.json"
        )

    if len(found) > 1:
        print("models.json 후보가 여러 개입니다:")
        for f in found:
            print(f"   - {f}")
        print("  -> 가장 최근 파일을 사용합니다. 다른 것을 쓰려면 --json으로 지정하세요.")
    return max(found, key=lambda q: q.stat().st_mtime)


# ============================================================
# 검증
# ============================================================

def validate_payload(payload):
    """깨진 JSON을 그대로 HTML에 박아 넣지 않도록 최소한의 구조를 확인."""
    problems = []
    models = payload.get("models") or {}
    if not models:
        problems.append("models가 비어 있습니다.")
    if "encoding" not in payload:
        problems.append("최상위 encoding 항목이 없습니다. (run_experiments.py가 구버전일 수 있습니다)")

    for name, m in models.items():
        for key in ("intercept", "coefficients", "features", "or_table", "formula"):
            if key not in m:
                problems.append(f"{name}: '{key}' 누락")
        coefs, feats = m.get("coefficients", {}), m.get("features", [])
        if coefs and feats and set(coefs) != set(feats):
            problems.append(f"{name}: coefficients와 features가 불일치")
    return problems


def verify_golden(payload, csv_path, tol=1e-4):
    """
    golden set CSV(python_prob 포함)를 읽어 JSON 계수로 확률을 다시 계산하고 비교.
    HTML이 하게 될 계산(intercept + sum(coef*x) -> sigmoid)을 파이썬으로 그대로 재현합니다.
    """
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    if not rows:
        print("golden set이 비어 있습니다."); return True

    worst, n = 0.0, 0
    for r in rows:
        m = payload["models"][r["model_name"]]
        logit = m["intercept"]
        for k, v in m["coefficients"].items():
            logit += v * float(r[k])
        worst = max(worst, abs(1 / (1 + math.exp(-logit)) - float(r["python_prob"])))
        n += 1

    ok = worst < tol
    print(f"  golden set n={n}, 최대 오차 {worst:.2e} -> {'일치' if ok else '불일치!'}")
    if not ok:
        print("  계수가 어긋났습니다. models.json을 다시 생성하세요.")
    return ok


# ============================================================
# 빌드
# ============================================================

def build(json_path, out_path, copy_json=True):
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))

    problems = validate_payload(payload)
    if problems:
        print("models.json 구조에 문제가 있습니다:")
        for p in problems:
            print(f"   - {p}")
        raise SystemExit(1)

    if PLACEHOLDER not in TEMPLATE:
        raise SystemExit(f"TEMPLATE에 {PLACEHOLDER} 자리표시자가 없습니다.")

    # </ 를 이스케이프: JSON 문자열 안의 </script>가 <script> 블록을 조기 종료시키는 것을 방지
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = (HERE / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(TEMPLATE.replace(PLACEHOLDER, text), encoding="utf-8")

    models = payload["models"]
    print(f"입력 : {json_path}")
    print(f"출력 : {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"모델 {len(models)}개: {', '.join(models)}")

    # index.html 옆에 models.json 사본을 둡니다.
    # 웹(https://)에서는 페이지가 이 파일을 fetch해서 쓰므로, 모델만 갱신하고 싶을 때
    # 이 json만 교체하면 사이트에 즉시 반영됩니다(HTML 재빌드 불필요).
    if copy_json:
        side = out_path.parent / "models.json"
        if Path(json_path).resolve() != side.resolve():
            side.write_text(Path(json_path).read_text(encoding="utf-8"), encoding="utf-8")
            print(f"사본 : {side}  (웹에서 fetch 대상)")
    return out_path, payload


def main():
    ap = argparse.ArgumentParser(description="models.json -> index.html")
    ap.add_argument("--json", default=None, help="models.json 경로 (생략 시 자동 탐색)")
    ap.add_argument("--out", default="index.html", help="출력 파일 (기본 index.html)")
    ap.add_argument("--verify", default=None, help="golden set CSV로 계수 일치 확인")
    ap.add_argument("--no-copy", action="store_true",
                    help="index.html 옆에 models.json 사본을 만들지 않음 (단일 파일만 배포할 때)")
    a = ap.parse_args()

    json_path = find_models_json(a.json)
    _, payload = build(json_path, a.out, copy_json=not a.no_copy)

    if a.verify:
        print("\n계수 검증:")
        if not verify_golden(payload, a.verify):
            sys.exit(1)

    print("\n완료. index.html을 브라우저에서 열어 확인하세요.")


if __name__ == "__main__":
    main()