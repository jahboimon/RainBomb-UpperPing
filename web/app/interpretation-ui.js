(() => {
  "use strict";
  const I = window.RainBombInterpretation;
  const $ = selector => document.querySelector(selector);
  const $$ = selector => [...document.querySelectorAll(selector)];

  function infoButton(term) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "info-trigger";
    button.dataset.term = term;
    button.setAttribute("aria-label", `คำอธิบาย ${I.term(term)?.title || term}`);
    button.textContent = "i";
    return button;
  }

  function decorate(selector, term, title) {
    const node = $(selector);
    if (!node || node.querySelector(".info-trigger")) return;
    if (title) node.textContent = title;
    node.append(" ", infoButton(term));
  }

  function makeCallout(text, className = "interpretation-note") {
    const note = document.createElement("p");
    note.className = className;
    note.textContent = text;
    return note;
  }

  function localizeStatic() {
    document.documentElement.lang = "th";
    const pairs = [
      ["#map-controls .panel-heading h1", "ชั้นข้อมูลแผนที่"],
      [".page-map .control-group legend", "รูปแบบการดู"],
      [".page-map .station-data dt", "ช่วงข้อมูลที่เลือก"],
      ["#events-title", "ตรวจสอบเหตุการณ์ฝนตกหนักเฉพาะแห่ง"],
      ["#model-title", "โมเดลตรวจพบเหตุการณ์ได้อย่างไร"],
      ["#xai-title", "ตัวแปรใดมีบทบาทต่อผลของโมเดล"],
      ["#xai-controls .panel-heading h2", "ตัวแปรและชั้นข้อมูล"],
      ["#method-title", "ข้อมูลและวิธีการวิจัย"]
    ];
    pairs.forEach(([selector, label]) => { const node = $(selector); if (node) node.textContent = label; });

    const mapControls = $("#map-controls .control-scroll");
    mapControls?.prepend(makeCallout("แผนที่นี้แสดงค่าที่สถานีตรวจวัด ไม่ใช่พื้นผิวพยากรณ์ทั่วลุ่มน้ำ", "compact-interpretation-note"));
    $(".events-workspace .event-detail")?.prepend(makeCallout("ผลจริง = พบ Def5 ที่สถานี · ผลทำนาย = โมเดลประเมินจากข้อมูลก่อนหน้า 1 ชั่วโมง", "compact-interpretation-note"));
    $(".model-explanation")?.insertBefore(makeCallout(I.uiText.prediction), $(".model-explanation .metric-lines"));
    $(".confusion-section")?.append(makeCallout("สีใช้ช่วยแยกประเภทเท่านั้น โปรดอ่านคำอธิบาย TP / FP / FN / TN กำกับด้วย", "compact-interpretation-note"));
    $(".model-station-section .model-map-grid")?.before(makeCallout("วงกลมแสดงผลเฉพาะตำแหน่งสถานี: สี = สัดส่วนเหตุการณ์จริงที่ตรวจพบ (Recall), ขนาด = จำนวน Def5 จริง; ไม่ใช่แผนที่ความเสี่ยง", "interpretation-note"));
    $(".xai-page .workspace-titlebar")?.after(makeCallout(I.uiText.shap, "xai-public-note"));
    $("#xai-controls .control-scroll")?.prepend(makeCallout("เลือกตัวแปรเพื่อดูบทบาทของตัวแปรนั้นในโมเดล ค่าเหล่านี้ไม่ใช่ผลเชิงสาเหตุ", "compact-interpretation-note"));
    $(".xai-method-note").textContent = I.uiText.kde;
    $(".method-article")?.prepend(makeCallout("เว็บนี้ช่วยสำรวจผลวิจัยที่คำนวณไว้แล้ว โมเดลพยากรณ์ล่วงหน้า 1 ชั่วโมงที่ตำแหน่งสถานี ไม่ใช่ระบบเตือนภัยหรือแผนที่พยากรณ์ทุกจุด", "method-public-note"));

    const staticText = [
      [".data-badge", "ผลวิจัยที่ตรึงไว้ · ประเมินปี 2025"],
      [".page-map .control-help", "ช่วงปีอื่นยังไม่แสดงในรุ่นนี้"],
      [".page-map .map-legend > .eyebrow", "คำอธิบายสัญลักษณ์"],
      [".page-map .source-chip", "ตำแหน่งจาก GDB · ค่าประเมินปี 2025 ที่ตรึงไว้"],
      [".page-map .method-note", I.uiText.station],
      [".station-data .wide dd", "ประเมิน · พฤษภาคม–ตุลาคม 2025"],
      ["#view-station-events", "ดูเหตุการณ์ที่สถานี →"],
      [".events-workspace .unavailable-block strong", "ยังไม่มีข้อมูลฝนรายชั่วโมง E−6 ถึง E+6"],
      [".events-workspace .unavailable-block p", "ข้อมูลต้นทางมีเพียงค่าสรุปเหตุการณ์ จึงไม่สร้างกราฟฝนรายชั่วโมงขึ้นเอง"],
      [".events-workspace .method-note", I.uiText.time],
      [".workspace-titlebar:has(#events-title) > p", "95 เหตุการณ์ Def5 ที่สถานีตรวจวัดในช่วงประเมินปี 2025"],
      [".events-workspace label[for='event-station-select']", "สถานี"],
      [".events-workspace label[for='event-select']", "เหตุการณ์"],
      [".events-workspace .map-caption span:first-child", "บริบทพื้นที่ลุ่มน้ำปิงตอนบน"],
      [".events-workspace .map-caption span:last-child", "จุดแสดงตำแหน่งสถานีที่ตรวจวัด"],
      [".model-explanation h2", "อ่านผลการประเมิน"],
      [".model-explanation > p:not(.interpretation-note)", "โมเดลจัดลำดับชั่วโมงที่เกิดเหตุการณ์ได้ดีกว่าการสุ่มตาม ROC AUC 0.864 แต่ค่านี้ไม่ใช่ความถูกต้อง 86.4% และกรณีแจ้งว่าเกิดแต่จริงไม่เกิดยังมีมาก เพราะเหตุการณ์จริงพบได้น้อยมาก"],
      ["#confusion-title", "ตารางผลจริงเทียบผลทำนาย"],
      [".confusion-section .section-heading .eyebrow", "แถว = ผลจริง · คอลัมน์ = ผลทำนาย"],
      [".model-station-section .section-heading h2", "ผลการประเมินแยกตามสถานี"],
      ["#model-event-toggle + span", "แสดงจุดเหตุการณ์ TP/FN"],
      [".xai-page .workspace-titlebar > p", "ดูบทบาทตัวแปรที่สถานี ไม่ใช่ปริมาณฝน ความเสี่ยง หรือผลเชิงสาเหตุ"],
      [".method-intro > p", "คำจำกัดความ ข้อมูลนำเข้า ผลประเมิน และข้อจำกัดในการตีความ"],
      ["#xai-kde-legend strong", "การกระจุกตัวของบทบาทตัวแปร (KDE)"],
      ["#xai-kde-legend p", "ช่วงสีปรับตามตัวแปร; เปรียบเทียบความเข้มข้ามตัวแปรโดยตรงไม่ได้"],
      ["#xai-controls-toggle", "ชั้นข้อมูล"],
      ["#xai-inspection-toggle", "สถานี"],
      [".xai-page .map-title-overlay .eyebrow", "ชั้นข้อมูลปัจจุบัน"]
    ];
    staticText.forEach(([selector, value]) => { const node = $(selector); if (node) node.textContent = value; });
    const groupTH = { Rain: "ฝน", Humidity: "ความชื้น", Temperature: "อุณหภูมิ", Time: "เวลา", Static_GIS: "ข้อมูลเชิงพื้นที่", ERA5: "บรรยากาศ ERA5" };
    I.groupLabel = value => groupTH[value] ? groupTH[value] + " (" + value + ")" : value;
    const replaceText = (selector, text) => { const node = $(selector); if (node) node.textContent = text; };
    replaceText(".page-map .segmented label:nth-child(1) span", "สรุปทั้งช่วง");
    replaceText(".page-map .segmented label:nth-child(2) span", "เลือกชั่วโมง");
    replaceText(".page-map .control-scroll > .control-group:nth-child(2) .control-label", "ช่วงข้อมูลวิเคราะห์");
    replaceText(".page-map .control-scroll > fieldset:nth-of-type(2) legend", "แผนที่ฐาน");
    replaceText(".page-map .control-scroll > fieldset:nth-of-type(3) legend", "ขอบเขตและบริบท");
    replaceText(".page-map .control-scroll > fieldset:nth-of-type(4) legend", "ชั้นข้อมูลพื้นหลัง");
    replaceText(".page-map .control-scroll > fieldset:nth-of-type(5) legend", "ข้อมูลตรวจวัด");
    replaceText(".page-map .control-scroll > fieldset:nth-of-type(6) legend", "สัญลักษณ์สถานี");
    replaceText(".page-map .option-row:has(#basin-toggle) span", "ลุ่มน้ำปิงตอนบน");
    replaceText(".page-map .option-row:has(#admin-toggle) span", "ขอบเขตจังหวัด");
    replaceText(".page-map .option-row:has(#gauges-toggle) span", "สถานีตรวจวัดฝน");
    replaceText(".page-map .option-row:has(#events-toggle) span", "จุดเหตุการณ์ Def5");
    replaceText(".page-map .option-row:has([name='station-symbol'][value='count']) span", "จำนวน Def5");
    replaceText(".page-map .option-row:has([name='station-symbol'][value='rate']) span", "อัตรา Def5 ต่อ 100,000 ชั่วโมง");
    replaceText(".page-map .option-row:has([name='station-symbol'][value='location']) span", "ตำแหน่งสถานีเท่านั้น");
    replaceText(".page-map .option-row:has([name='map-thematic'][value='elevation']) span", "ระดับความสูง");
    replaceText(".page-map .control-scroll > fieldset:nth-of-type(4) .control-help", "สีและเงาเป็นการแสดงผล ไม่เปลี่ยนค่าระดับความสูงต้นทาง");
    replaceText("#xai-kde-legend .xai-range span:first-child", "ต่ำกว่า");
    replaceText("#xai-kde-legend .xai-range span:last-child", "สูงกว่า");
    replaceText(".xai-controls label[for='xai-feature-select']", "ตัวแปร");
    replaceText(".xai-controls fieldset:nth-of-type(1) legend", "บทบาทตัวแปรรายสถานี");
    replaceText(".xai-controls fieldset:nth-of-type(2) legend", "ชั้นข้อมูลเชิงพื้นที่");
    replaceText(".xai-controls .option-row:has(#xai-basin-toggle) span", "ลุ่มน้ำปิงตอนบน");
    replaceText(".xai-controls .option-row:has(#xai-admin-toggle) span", "ขอบเขตจังหวัด");
    replaceText(".xai-controls .range-row:has(#xai-kde-opacity) span", "ความทึบ KDE");
    replaceText(".xai-controls .range-row:has(#xai-elevation-opacity) span", "ความทึบระดับความสูง");
    const eventLabels = [["#event-rain", "ฝนในชั่วโมงเกิดเหตุ (Rain(E))"], ["#event-p99", "เกณฑ์ฝนสูงเฉพาะสถานี (P99)"], ["#event-ante", "ฝนสะสม 3 ชั่วโมงก่อนหน้า"], ["#event-post", "ฝน 3 ชั่วโมงถัดไป (Post1–3)"], ["#event-postmax", "ค่าสูงสุดของฝน 3 ชั่วโมงถัดไป"], ["#event-outcome", "ผลจริงเทียบผลทำนาย · เกณฑ์ 0.5"]];
    eventLabels.forEach(([selector, label]) => { const dt = $(selector)?.previousElementSibling; if (dt) dt.textContent = label; });
    const stationLabels = [["#station-count", "จำนวน Def5 ที่ตรวจพบ"], ["#station-hours", "ชั่วโมงที่ประเมินได้"], ["#station-rate", "อัตรา Def5"], ["#station-elevation", "ระดับความสูง"], ["#station-built", "ลักษณะพื้นที่สิ่งปลูกสร้าง"], ["#station-suhi", "ตัวบ่งชี้ความร้อนเมือง (SUHI_5K)"]];
    stationLabels.forEach(([selector, label]) => { const dt = $(selector)?.previousElementSibling; if (dt) dt.textContent = label; });
    const allStations = $("#event-station-select option[value='all']"); if (allStations) allStations.textContent = "ทุกสถานี";
    replaceText(".page-intro:has(#model-title) > p", "ข้อมูลเวลา t → ประเมิน Def5 ที่สถานีเดิมในอีก 1 ชั่วโมง · เกณฑ์แยกผล 0.5");
    replaceText("#model-station-name", "เลือกสถานีบนแผนที่");
    replaceText("#model-station-id", "คลิกจุดสถานีเพื่อตรวจสอบ");
    replaceText(".model-station-section .inline-legend span:nth-child(1)", "● ตรวจพบเหตุการณ์จริงได้มากกว่า");
    replaceText(".model-station-section .inline-legend span:nth-child(2)", "● ตรวจพบได้น้อยกว่า");
    replaceText(".model-station-section .inline-legend span:nth-child(3)", "● ไม่มีเหตุการณ์จริงให้คำนวณ");
    const metricLabels = [["#metric-rows", "จำนวนสถานี-ชั่วโมงที่ประเมิน"], ["#metric-actual", "Def5 ที่เกิดจริง"], ["#metric-prevalence", "สัดส่วนเหตุการณ์จริง"]];
    metricLabels.forEach(([selector, label]) => { const dt = $(selector)?.previousElementSibling; if (dt) dt.textContent = label; });
    const cmAxes = ["ทำนายว่าไม่เกิด", "ทำนายว่าเกิด", "จริงไม่เกิด", "จริงเกิด"];
    $$(".confusion-matrix .cm-axis").forEach((node, index) => { if (cmAxes[index]) node.textContent = cmAxes[index]; });
    const fprLabel = $("#model-station-fpr")?.previousElementSibling; if (fprLabel) fprLabel.textContent = "อัตราแจ้งเกิน (FP rate)";

    const methodSummaries = [
      "ฝนตกหนักเฉพาะแห่ง (Def5) คือฝนรายชั่วโมงที่สูงเมื่อเทียบกับสถานีนั้น มีฝนก่อนหน้าเพียงเล็กน้อย และไม่มีฝนหนักต่อเนื่องอีก 3 ชั่วโมง; เกณฑ์ตัวเลขเดิมดูได้ด้านล่าง",
      "โมเดลใช้ข้อมูล ณ สถานีในเวลา t เพื่อประเมิน Def5 ที่สถานีเดิมในอีก 1 ชั่วโมง ขณะนี้เว็บแสดงเฉพาะผลประเมินปี 2025",
      "ตัวแปรนำเข้าโมเดลมี 20 ตัว ครอบคลุมฝน ความชื้น อุณหภูมิ เวลา ภูมิประเทศ และข้อมูลบรรยากาศ; กดปุ่ม i เพื่ออ่านแต่ละตัว",
      "ตัวอย่างแถวจริงขนาดเล็กแสดงลำดับจากข้อมูลรายชั่วโมง ไปสู่ตัวแปรที่สร้างขึ้น และข้อมูลพร้อมใช้กับโมเดล; ไม่ใช้แทนสัดส่วนคลาสทั้งหมด",
      "แผนที่อ้างอิงตำแหน่งสถานีและขอบเขตลุ่มน้ำ จุดข้อมูลไม่บอกขนาดพายุหรือค่าฝนระหว่างสถานี",
      "ตัวชี้วัดช่วยดูว่าโมเดลแยกกรณีเกิดกับไม่เกิดได้เพียงใด แต่ ROC AUC 0.864 ไม่ได้หมายถึงความถูกต้อง 86.4%",
      "SHAP อธิบายบทบาทของตัวแปรในโมเดล ส่วน KDE แสดงการกระจุกตัวของบทบาทนั้น ไม่ใช่ผลเชิงสาเหตุหรือแผนที่ความเสี่ยง",
      "เว็บแสดงสำเนาข้อมูลเพื่อการสำรวจจากผลวิจัยที่ตรึงไว้ โดยไม่แก้ข้อมูลวิทยาศาสตร์ต้นทาง",
      "ERA5 ถูกแปลงจาก UTC เป็น ICT ก่อนเชื่อมข้อมูลแล้ว ส่วนเขตเวลาของข้อมูลสถานีต้นทางยังไม่ยืนยัน และยังไม่มีกราฟฝนรายชั่วโมง E−6 ถึง E+6"
    ];
    const methodHeadings = ["นิยามฝนตกหนักเฉพาะแห่ง (Def5)", "ช่วงข้อมูลและการพยากรณ์", "ตัวแปรนำเข้าโมเดล 20 ตัว", "ตัวอย่างข้อมูลและลำดับการเตรียม", "ข้อมูลแผนที่และพื้นที่ศึกษา", "ผลการประเมินโมเดล", "SHAP และ KDE", "ที่มาของข้อมูลแสดงผล", "ข้อจำกัดปัจจุบัน"];
    $$(".method-article > section").forEach((section, index) => {
      const body = section.querySelector("div"); const heading = body?.querySelector("h2");
      if (!body || !heading || !methodSummaries[index]) return;
      heading.textContent = methodHeadings[index];
      const details = document.createElement("details"); details.className = "technical-details";
      const summary = document.createElement("summary"); summary.textContent = "ดูรายละเอียดทางเทคนิค"; details.append(summary);
      while (heading.nextSibling) details.append(heading.nextSibling);
      body.append(makeCallout(methodSummaries[index], "method-summary"), details);
    });

    const predictorList = $(".predictor-list");
    if (predictorList) {
      predictorList.replaceChildren();
      Object.values(I.features).forEach(feature => {
        const li = document.createElement("li");
        const name = document.createElement("span"); name.textContent = feature.displayNameTH;
        const code = document.createElement("small"); code.textContent = feature.internalName;
        li.append(name, code, infoButton(feature.internalName));
        predictorList.append(li);
      });
    }

    const metricTerms = [
      ["#metric-ap", "Average Precision"], ["#metric-auc", "ROC AUC"],
      ["#metric-precision", "Precision"], ["#metric-recall", "Recall"]
    ];
    metricTerms.forEach(([valueSelector, term]) => {
      const dt = $(valueSelector)?.previousElementSibling;
      if (dt) { dt.innerHTML = `${I.metrics[term][0]} <small>${term}</small> `; dt.append(infoButton(term)); }
    });
    const scoreDt = $("#event-score")?.previousElementSibling;
    if (scoreDt) { scoreDt.textContent = "คะแนนจากโมเดล "; scoreDt.append(infoButton("Model score")); }
    const modelScoreDt = $("#model-station-score")?.previousElementSibling;
    if (modelScoreDt) { modelScoreDt.textContent = "คะแนนเฉลี่ยจากโมเดล "; modelScoreDt.append(infoButton("Model score")); }
    const recallDt = $("#model-station-recall")?.previousElementSibling;
    if (recallDt) { recallDt.textContent = "สัดส่วนเหตุการณ์จริงที่ตรวจพบ (Recall) "; recallDt.append(infoButton("Recall")); }
    [["#xai-detail-abs", "MeanAbsSHAP"], ["#xai-detail-signed", "MeanSignedSHAP"], ["#xai-detail-recall", "Recall"]].forEach(([selector, term]) => {
      const dt = $(selector)?.previousElementSibling;
      if (dt) { dt.textContent = `${I.term(term).title} `; dt.append(infoButton(term)); }
    });
    [["#xai-abs-toggle", "MeanAbsSHAP"], ["#xai-signed-toggle", "MeanSignedSHAP"], ["#xai-kde-toggle", "KDE"], ["#xai-elevation-toggle", "elevation"]].forEach(([selector, term]) => {
      const label = $(selector)?.closest("label");
      if (label) {
        if (label.querySelector("span")) label.querySelector("span").textContent = I.term(term).title;
        label.after(infoButton(term));
      }
    });
    $(".confusion-matrix")?.querySelectorAll(".cm-cell").forEach(cell => {
      const code = cell.classList.contains("tp") ? "TP" : cell.classList.contains("fp") ? "FP" : cell.classList.contains("fn") ? "FN" : "TN";
      const label = cell.querySelector("span");
      if (label) { label.textContent = `${I.metrics[code][0]} (${code}) `; label.append(infoButton(code)); }
    });
    const threshold = $(".threshold-label");
    if (threshold) { threshold.textContent = "เกณฑ์ 0.5 "; threshold.append(infoButton("Threshold 0.5")); }
  }

  function helpPanel() {
    const button = document.createElement("button");
    button.id = "interpretation-open";
    button.type = "button";
    button.className = "nav-link help-nav";
    button.textContent = "วิธีอ่านผล";
    button.setAttribute("aria-controls", "interpretation-panel");
    button.setAttribute("aria-expanded", "false");
    $(".primary-nav")?.append(button);
    const panel = document.createElement("aside");
    panel.id = "interpretation-panel";
    panel.className = "interpretation-panel";
    panel.hidden = true;
    panel.setAttribute("aria-label", "คำอธิบายและคำศัพท์ในระบบ");
    panel.innerHTML = `<div class="interpretation-head"><div><span class="eyebrow">คำอธิบาย</span><h2>วิธีอ่านผล</h2></div><button type="button" id="interpretation-close" aria-label="ปิดคำอธิบาย">×</button></div><div class="help-overview"><p><strong>เว็บนี้แสดงอะไร</strong> ผลวิจัยที่คำนวณไว้แล้วสำหรับฝนตกหนักเฉพาะแห่ง (Def5) ณ สถานีตรวจวัด</p><p><strong>โมเดลพยากรณ์อะไร</strong> ใช้ข้อมูลเวลา t เพื่อประเมินเหตุการณ์ที่สถานีเดิมในอีก 1 ชั่วโมง ไม่ใช่แผนที่พยากรณ์ทั่วลุ่มน้ำ</p><p><strong>อ่านผลอย่างไร</strong> เทียบผลจริงกับผลทำนายด้วย TP/FP/FN/TN; คะแนนจากโมเดลไม่ใช่เปอร์เซ็นต์โอกาสเกิดโดยตรง</p><p><strong>อ่านแผนที่อย่างไร</strong> ค่าที่จุดเป็นค่ารายสถานี; SHAP อธิบายบทบาทตัวแปรในโมเดล; KDE แสดงการกระจุกตัว ไม่ใช่ฝนหรือความเสี่ยง</p></div><label class="glossary-search-label">ค้นหาคำศัพท์<input id="glossary-search" type="search" placeholder="เช่น q_850, SHAP, Recall, FP, KDE, ERA5"></label><div id="glossary-results" class="glossary-results"></div>`;
    document.body.append(panel);
    const terms = ["Def5", "Model score", "Threshold 0.5", "TP", "FP", "FN", "TN", "SHAP", "MeanAbsSHAP", "MeanSignedSHAP", "KDE", "Average Precision", "ROC AUC", "Precision", "Recall", "Specificity", "F1-score", "ERA5", ...Object.keys(I.features), ...Object.keys(I.layers)];
    function renderResults() {
      const query = $("#glossary-search").value.trim().toLocaleLowerCase();
      const found = terms.filter(term => { const item = I.term(term); return `${term} ${item?.title} ${item?.description}`.toLocaleLowerCase().includes(query); });
      const root = $("#glossary-results"); root.replaceChildren();
      if (!found.length) { root.textContent = "ไม่พบคำศัพท์ที่ตรงกัน"; return; }
      found.forEach(term => {
        const item = I.term(term); const card = document.createElement("article");
        const h = document.createElement("h3"); h.textContent = item.title;
        const code = document.createElement("small"); code.textContent = item.technical;
        const p = document.createElement("p"); p.textContent = item.description;
        card.append(h, code, p);
        if (item.unit) { const unit = document.createElement("p"); unit.textContent = `หน่วย: ${item.unit}`; card.append(unit); }
        if (item.source) { const source = document.createElement("p"); source.textContent = `แหล่งข้อมูล: ${item.source}`; card.append(source); }
        root.append(card);
      });
    }
    function toggle(open) { panel.hidden = !open; button.setAttribute("aria-expanded", String(open)); if (open) { renderResults(); $("#glossary-search").focus(); } else button.focus(); }
    button.addEventListener("click", () => toggle(panel.hidden));
    $("#interpretation-close").addEventListener("click", () => toggle(false));
    $("#glossary-search").addEventListener("input", renderResults);
    document.addEventListener("keydown", event => { if (event.key === "Escape" && !panel.hidden) toggle(false); });
  }

  function popover() {
    const bubble = document.createElement("div"); bubble.id = "term-popover"; bubble.className = "term-popover"; bubble.hidden = true;
    document.body.append(bubble);
    function show(button) {
      const item = I.term(button.dataset.term); if (!item) return;
      bubble.replaceChildren();
      const title = document.createElement("strong"); title.textContent = item.title;
      const code = document.createElement("small"); code.textContent = item.technical;
      const description = document.createElement("p"); description.textContent = item.description;
      bubble.append(title, code, description);
      if (item.unit) { const unit = document.createElement("p"); unit.textContent = `หน่วย: ${item.unit}`; bubble.append(unit); }
      if (item.source) { const source = document.createElement("p"); source.textContent = `แหล่งข้อมูล: ${item.source}`; bubble.append(source); }
      bubble.hidden = false;
      const rect = button.getBoundingClientRect();
      bubble.style.left = `${Math.max(12, Math.min(rect.left, window.innerWidth - 292))}px`;
      bubble.style.top = `${Math.max(12, Math.min(rect.bottom + 8, window.innerHeight - bubble.offsetHeight - 12))}px`;
    }
    document.addEventListener("click", event => {
      const button = event.target.closest(".info-trigger");
      if (!button) { if (!bubble.contains(event.target)) bubble.hidden = true; return; }
      event.preventDefault(); event.stopPropagation();
      if (!bubble.hidden && bubble.dataset.term === button.dataset.term) { bubble.hidden = true; return; }
      bubble.dataset.term = button.dataset.term; show(button);
    });
    document.addEventListener("pointerover", event => { if (event.pointerType === "mouse" && event.target.closest(".info-trigger")) show(event.target.closest(".info-trigger")); });
    document.addEventListener("focusin", event => { if (event.target.matches(".info-trigger")) show(event.target); });
    document.addEventListener("keydown", event => { if (event.key === "Escape") bubble.hidden = true; });
  }

  localizeStatic(); helpPanel(); popover();
})();
