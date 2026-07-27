#!/usr/bin/env python3
"""足球数据抓取脚本"""

import json, os, re, math
from urllib.request import urlopen, Request
from datetime import datetime, timezone, timedelta

HISTORY_FILE = "data/history.json"
PREDICTIONS_FILE = "data/predictions.json"
MATCHES_FILE = "data/matches.json"
HOME_ADV = 0.35

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def crs_to_probs(crs):
    probs = {}
    total = 0
    for k, odds in crs.items():
        if odds > 0:
            probs[k] = 1/odds
            total += probs[k]
    for k in probs:
        probs[k] /= total
    return probs

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def fetch_current_matches():
    matches = []
    try:
        url = "https://webapi.sporttery.cn/gateway/jc/football/getMatchCalculatorV1.qry"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.lottery.gov.cn/"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            match_id = 0
            for day in data.get("value", {}).get("matchInfoList", []):
                for m in day.get("subMatchList", []):
                    had = m.get("had", {})
                    h = float(had.get("h", "0") or "0")
                    if h <= 0: continue
                    match_id += 1
                    crs = {}
                    for k, v in m.get("crs", {}).items():
                        if k.startswith("s") and len(k) == 6 and "f" not in k:
                            try: crs[k] = float(v)
                            except: pass
                    hhad = m.get("hhad", {})
                    handicap = {"line": hhad.get("goalLine", ""), "home": float(hhad.get("h", "0") or "0"), "away": float(hhad.get("a", "0") or "0"), "draw": float(hhad.get("d", "0") or "0")}
                    hafu = m.get("hafu", {})
                    htft = {}
                    for k in ["hh","hd","ha","dh","dd","da","ah","ad","aa"]:
                        try: htft[k] = float(hafu.get(k, "0") or "0")
                        except: htft[k] = 0
                    matches.append({
                        "id": match_id,
                        "league": m.get("leagueAbbName", "") or m.get("groupName", "竞彩"),
                        "time": f"{m.get('matchDate', '')} {m.get('matchTime', '')}",
                        "home": {"name": m.get("homeTeamAbbName", ""), "rank": m.get("homeRank", "").strip("[]")},
                        "away": {"name": m.get("awayTeamAbbName", ""), "rank": m.get("awayRank", "").strip("[]")},
                        "odds": {"home": h, "draw": float(had.get("d", "0") or "0"), "away": float(had.get("a", "0") or "0")},
                        "handicap": handicap, "crs": crs, "htft": htft
                    })
        print(f"竞彩赛事: {len(matches)}场")
    except Exception as e:
        print(f"竞彩API错误: {e}")
    return matches

def fetch_results():
    results = []
    try:
        # 澳客网最多支持5天查询
        end_date = "2026-07-27"
        start_date = "2026-07-23"
        url = f"https://www.okooo.cn/jingcai/kaijiang/?LotteryType=SportteryWDL&StartDate={start_date}&EndDate={end_date}"
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
            
            league = clean[1]
            time_parts = clean[2].split(" ")
            date = time_parts[0] if time_parts else ""
            home = clean[3]
            away = clean[4]
            ft_score = clean[7]
            result_code = clean[8]
            
            if result_code == "3": result = "主胜"
            elif result_code == "1": result = "平局"
            elif result_code == "0": result = "客胜"
            else: continue
            
            results.append({"date": date, "league": league, "home": home, "away": away, "ft_score": ft_score, "result": result})
        print(f"澳客网赛果: {len(results)}场")
    except Exception as e:
        print(f"澳客网错误: {e}")
    return results

def generate_predictions(matches):
    predictions = []
    for m in matches:
        cp = crs_to_probs(m["crs"])
        hxg = sum(int(k[1:3]) * v for k, v in cp.items())
        axg = sum(int(k[4:6]) * v for k, v in cp.items())
        hL = hxg + HOME_ADV
        aL = max(0.3, axg - HOME_ADV * 0.3)
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
        ou = "大2.5" if o25 > 0.5 else "小2.5"
        predictions.append({"home": m["home"]["name"], "away": m["away"]["name"], "pick": pick, "prob": round(mx * 100), "score": top_score, "ou": ou, "odds": m["odds"]})
    print(f"生成预测: {len(predictions)}场")
    return predictions

def update_history(history, results, predictions):
    new_matches = []
    for r in results:
        key = f"{r['date']}_{r['home']}_{r['away']}"
        if any(h.get("key") == key for h in history["matches"]): continue
        pred = None
        for p in predictions:
            if p["home"] == r["home"] and p["away"] == r["away"]:
                pred = p
                break
        if not pred: continue
        spf_hit = pred["pick"] == r["result"]
        ft_score = r["ft_score"]
        score_hit = pred["score"] == ft_score
        ft_goals = sum(int(x) for x in ft_score.split("-"))
        if pred["ou"] == "大2.5": ou_hit = ft_goals > 2
        elif pred["ou"] == "小2.5": ou_hit = ft_goals <= 2
        else: ou_hit = False
        new_matches.append({"key": key, "date": r["date"], "league": r["league"], "home": r["home"], "away": r["away"], "home_score": int(ft_score.split("-")[0]), "away_score": int(ft_score.split("-")[1]), "prediction": {"pick": pred["pick"], "prob": pred["prob"], "score": pred["score"], "ou": pred["ou"]}, "result": {"pick": "✅" if spf_hit else "❌", "score": "✅" if score_hit else "❌", "ou": "✅" if ou_hit else "❌"}, "odds": pred["odds"]})
    history["matches"] = new_matches + history["matches"]
    total = len(history["matches"])
    if total > 0:
        spf_hit = sum(1 for m in history["matches"] if m["result"]["pick"] == "✅")
        score_hit = sum(1 for m in history["matches"] if m["result"]["score"] == "✅")
        goal_hit = sum(1 for m in history["matches"] if m["result"]["ou"] == "✅")
        history["summary"] = {"total": total, "spf_hit": spf_hit, "spf_rate": round(spf_hit / total * 100), "score_hit": score_hit, "score_rate": round(score_hit / total * 100), "goal_hit": goal_hit, "goal_rate": round(goal_hit / total * 100), "roi": round((spf_hit * 0.8 - (total - spf_hit)) / total * 100, 1)}
    history["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return history

def main():
    history = load_json(HISTORY_FILE, {"updated": "", "summary": {"total": 0, "spf_hit": 0, "spf_rate": 0, "score_hit": 0, "score_rate": 0, "goal_hit": 0, "goal_rate": 0, "roi": 0}, "matches": []})
    current_matches = fetch_current_matches()
    results = fetch_results()
    predictions = generate_predictions(current_matches)
    history = update_history(history, results, predictions)
    save_json(MATCHES_FILE, {"matches": current_matches, "updated": history["updated"], "sources": ["中国竞彩网"]})
    save_json(PREDICTIONS_FILE, predictions)
    save_json(HISTORY_FILE, history)
    print(f"完成! 赛事: {len(current_matches)}场, 历史: {history['summary']['total']}场, 命中率: {history['summary']['spf_rate']}%")

if __name__ == "__main__":
    main()
