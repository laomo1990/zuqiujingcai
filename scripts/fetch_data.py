#!/usr/bin/env python3
"""足球数据自动抓取"""

import json, os, re, math
from urllib.request import urlopen, Request
from datetime import datetime, timezone, timedelta

HISTORY_FILE = "data/history.json"
PREDICTIONS_FILE = "data/predictions.json"
MATCHES_FILE = "data/matches.json"

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def crs_to_probs(crs):
    probs = {}; total = 0
    for k, odds in crs.items():
        if odds > 0: probs[k] = 1/odds; total += probs[k]
    for k in probs: probs[k] /= total
    return probs

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

def fetch_okooo_results():
    results = []
    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d")
        url = f"https://www.okooo.cn/jingcai/kaijiang/?LotteryType=SportteryWDL&StartDate={start}&EndDate={end}"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=15) as resp:
            html = resp.read().decode("gb2312", errors="ignore")
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
        for row in rows:
            if not re.search(r"\d+-\d+", row): continue
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
            if len(cells) < 8: continue
            clean = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
            if not re.match(r"\d+-\d+", clean[7]): continue
            code = clean[8]
            if code == "3": result = "主胜"
            elif code == "1": result = "平局"
            elif code == "0": result = "客胜"
            else: continue
            results.append({"date": clean[2].split(" ")[0], "league": clean[1], "home": clean[3], "away": clean[4], "ft_score": clean[7], "result": result})
    except Exception as e:
        print(f"澳客网错误: {e}")
    return results

def fetch_sporttery():
    matches = []
    try:
        url = "https://webapi.sporttery.cn/gateway/jc/football/getMatchCalculatorV1.qry"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.lottery.gov.cn/"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        mid = 0
        for day in data.get("value", {}).get("matchInfoList", []):
            for m in day.get("subMatchList", []):
                had = m.get("had", {})
                h = float(had.get("h", "0") or "0")
                if h <= 0: continue
                mid += 1
                crs = {}
                for k, v in m.get("crs", {}).items():
                    if k.startswith("s") and len(k) == 6 and "f" not in k:
                        try: crs[k] = float(v)
                        except: pass
                hhad = m.get("hhad", {})
                hc = {"line": hhad.get("goalLine", ""), "home": float(hhad.get("h", "0") or "0"), "away": float(hhad.get("a", "0") or "0"), "draw": float(hhad.get("d", "0") or "0")}
                hafu = m.get("hafu", {})
                ht = {}
                for k in ["hh","hd","ha","dh","dd","da","ah","ad","aa"]:
                    try: ht[k] = float(hafu.get(k, "0") or "0")
                    except: ht[k] = 0
                matches.append({"id": mid, "league": m.get("leagueAbbName", "") or m.get("groupName", "竞彩"), "time": f"{m.get('matchDate', '')} {m.get('matchTime', '')}", "home": {"name": m.get("homeTeamAbbName", ""), "rank": m.get("homeRank", "").strip("[]")}, "away": {"name": m.get("awayTeamAbbName", ""), "rank": m.get("awayRank", "").strip("[]")}, "odds": {"home": h, "draw": float(had.get("d", "0") or "0"), "away": float(had.get("a", "0") or "0")}, "handicap": hc, "crs": crs, "htft": ht})
    except Exception as e:
        print(f"竞彩API错误: {e}")
    return matches

def make_prediction(m):
    if not m["crs"]: return None
    cp = crs_to_probs(m["crs"])
    hxg = sum(int(k[1:3]) * v for k, v in cp.items())
    axg = sum(int(k[4:6]) * v for k, v in cp.items())
    hL = hxg + 0.35; aL = max(0.3, axg - 0.1)
    ph = pd = pa = 0
    for i in range(7):
        for j in range(7):
            p = poisson_pmf(i, hL) * poisson_pmf(j, aL)
            if i > j: ph += p
            elif i == j: pd += p
            else: pa += p
    mx = max(ph, pd, pa)
    pick = "主胜" if ph == mx else ("客胜" if pa == mx else "平局")
    scores = sorted(cp.items(), key=lambda x: -x[1])
    top_score = f"{int(scores[0][0][1:3])}-{int(scores[0][0][4:6])}"
    o25 = sum(v for k, v in cp.items() if int(k[1:3]) + int(k[4:6]) >= 3)
    return {"home": m["home"]["name"], "away": m["away"]["name"], "pick": pick, "prob": round(mx*100), "score": top_score, "ou": "大2.5" if o25 > 0.5 else "小2.5", "odds": m["odds"]}

def main():
    print(f"[{datetime.now().strftime('%H:%M')}] 开始抓取...")
    history = load_json(HISTORY_FILE, {"updated": "", "summary": {"total":0,"spf_hit":0,"spf_rate":0,"score_hit":0,"score_rate":0,"goal_hit":0,"goal_rate":0,"roi":0}, "matches": []})
    old_preds = load_json(PREDICTIONS_FILE, [])
    results = fetch_okooo_results()
    matches = fetch_sporttery()
    print(f"赛果: {len(results)}场, 在售: {len(matches)}场")
    preds = [p for m in matches if (p := make_prediction(m))]
    if not preds: preds = old_preds
    print(f"预测: {len(preds)}场")
    new_count = 0
    for r in results:
        key = f"{r['date']}_{r['home']}_{r['away']}"
        if any(h.get("key") == key for h in history["matches"]): continue
        pred = None
        for p in preds:
            if (r["home"] in p["home"] or p["home"] in r["home"]) and (r["away"] in p["away"] or p["away"] in r["away"]):
                pred = p; break
        if not pred: continue
        spf_hit = pred["pick"] == r["result"]
        score_hit = pred["score"] == r["ft_score"]
        ft_goals = sum(int(x) for x in r["ft_score"].split("-"))
        ou_hit = (ft_goals > 2) if pred["ou"] == "大2.5" else (ft_goals <= 2)
        history["matches"].insert(0, {"key": key, "date": r["date"], "league": r["league"], "home": r["home"], "away": r["away"], "home_score": int(r["ft_score"].split("-")[0]), "away_score": int(r["ft_score"].split("-")[1]), "prediction": {"pick": pred["pick"], "prob": pred["prob"], "score": pred["score"], "ou": pred["ou"]}, "result": {"pick": "✅" if spf_hit else "❌", "score": "✅" if score_hit else "❌", "ou": "✅" if ou_hit else "❌"}, "odds": pred["odds"]})
        new_count += 1
    total = len(history["matches"])
    if total > 0:
        s = sum(1 for m in history["matches"] if m["result"]["pick"] == "✅")
        c = sum(1 for m in history["matches"] if m["result"]["score"] == "✅")
        g = sum(1 for m in history["matches"] if m["result"]["ou"] == "✅")
        history["summary"] = {"total":total,"spf_hit":s,"spf_rate":round(s/total*100),"score_hit":c,"score_rate":round(c/total*100),"goal_hit":g,"goal_rate":round(g/total*100),"roi":round((s*0.8-(total-s))/total*100,1)}
    history["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    save_json(MATCHES_FILE, {"matches": matches, "updated": history["updated"], "sources": ["中国竞彩网"]})
    save_json(PREDICTIONS_FILE, preds)
    save_json(HISTORY_FILE, history)
    print(f"完成! 新增赛果:{new_count}场, 历史:{total}场, 命中率:{history['summary']['spf_rate']}%")

if __name__ == "__main__":
    main()
