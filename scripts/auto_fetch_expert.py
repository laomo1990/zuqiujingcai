#!/usr/bin/env python3
"""
自动抓取足球分析数据
===========================
使用多种数据源，自动降级
"""

import json
import os
import re
from urllib.request import urlopen, Request
from urllib.parse import quote
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# 数据源配置
SOURCES = [
    {
        "name": "搜狗搜索",
        "url": "https://www.sogou.com/web?query={query}&ie=utf8",
        "headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        "encoding": "utf-8"
    },
    {
        "name": "百度搜索", 
        "url": "https://www.baidu.com/s?wd={query}",
        "headers": {"User-Agent": "Mozilla/5.0"},
        "encoding": "utf-8"
    }
]

def search_web(query, count=3):
    """使用多个搜索引擎"""
    results = []
    
    for source in SOURCES:
        try:
            url = source["url"].format(query=quote(query))
            req = Request(url, headers=source["headers"])
            with urlopen(req, timeout=10) as resp:
                html = resp.read().decode(source["encoding"], errors="ignore")
            
            # 提取标题和摘要
            titles = re.findall(r'<h3[^>]*>.*?<a[^>]*>(.*?)</a>.*?</h3>', html, re.DOTALL)
            snippets = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL)
            
            for i, title in enumerate(titles[:count]):
                clean_title = re.sub(r'<[^>]+>', '', title).strip()
                clean_snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
                if clean_title and len(clean_title) > 5:
                    results.append({
                        "source": source["name"],
                        "title": clean_title,
                        "snippet": clean_snippet[:200]
                    })
            
            if results:
                break
                
        except Exception as e:
            continue
    
    return results

def fetch_injury_data(team_name):
    """获取伤停数据"""
    # 尝试搜索
    results = search_web(f"{team_name} 伤停 受伤 停赛 2026", 5)
    
    injury_info = {
        "injuries": [],
        "suspensions": [],
        "impact": "low",
        "source": "search",
        "updated": datetime.now().isoformat()
    }
    
    for r in results:
        text = r.get("title", "") + " " + r.get("snippet", "")
        
        # 提取伤停信息
        patterns = [
            r'([^\s,，。]{2,6})(?:受伤|伤停|缺阵)',
            r'(?:受伤|伤停|缺阵)[^:：]*[:：]([^,，\n]{2,10})',
            r'([^\s,，。]{2,6})(?:停赛|红牌)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match and len(match) >= 2:
                    if '停赛' in text or '红牌' in text:
                        injury_info["suspensions"].append(match.strip())
                    else:
                        injury_info["injuries"].append(match.strip())
    
    # 去重
    injury_info["injuries"] = list(set(injury_info["injuries"]))[:5]
    injury_info["suspensions"] = list(set(injury_info["suspensions"]))[:3]
    
    # 评估影响
    total = len(injury_info["injuries"]) + len(injury_info["suspensions"])
    if total >= 3:
        injury_info["impact"] = "high"
    elif total >= 2:
        injury_info["impact"] = "medium"
    
    return injury_info if total > 0 else None

def fetch_expert_opinion(home, away):
    """获取专家意见"""
    results = search_web(f"{home} vs {away} 分析 预测 看好", 5)
    
    expert_info = {
        "consensus": "",
        "confidence": 0.5,
        "reason": "",
        "source": "search",
        "updated": datetime.now().isoformat()
    }
    
    home_votes = 0
    away_votes = 0
    draw_votes = 0
    reasons = []
    
    for r in results:
        text = r.get("title", "") + " " + r.get("snippet", "")
        
        # 分析看好方向
        if re.search(r'主队|主场|主胜|优势', text):
            home_votes += 1
            match = re.search(r'([^,，。]{5,30}(?:优势|看好|胜))', text)
            if match:
                reasons.append(match.group(1))
        
        if re.search(r'客队|客场|客胜|强势', text):
            away_votes += 1
            match = re.search(r'([^,，。]{5,30}(?:看好|胜|强势))', text)
            if match:
                reasons.append(match.group(1))
        
        if re.search(r'平局|握手言和', text):
            draw_votes += 1
    
    total = home_votes + away_votes + draw_votes
    if total > 0:
        if home_votes > away_votes and home_votes > draw_votes:
            expert_info["consensus"] = "主胜"
            expert_info["confidence"] = min(0.8, home_votes / total + 0.3)
        elif away_votes > home_votes and away_votes > draw_votes:
            expert_info["consensus"] = "客胜"
            expert_info["confidence"] = min(0.8, away_votes / total + 0.3)
        elif draw_votes > 0:
            expert_info["consensus"] = "平局"
            expert_info["confidence"] = min(0.7, draw_votes / total + 0.2)
        
        expert_info["reason"] = "; ".join(reasons[:2])
    
    return expert_info if expert_info["consensus"] else None

def auto_update_all():
    """自动更新所有数据"""
    # 加载赛事
    matches_path = os.path.join(DATA_DIR, "matches.json")
    if not os.path.exists(matches_path):
        print("无赛事数据")
        return
    
    with open(matches_path, "r", encoding="utf-8") as f:
        matches = json.load(f).get("matches", [])
    
    # 更新伤停
    injury_path = os.path.join(DATA_DIR, "injuries.json")
    injuries = {}
    if os.path.exists(injury_path):
        with open(injury_path, "r", encoding="utf-8") as f:
            injuries = json.load(f)
    
    injury_updated = 0
    for m in matches[:5]:
        home = m["home"]["name"] if isinstance(m["home"], dict) else m["home"]
        away = m["away"]["name"] if isinstance(m["away"], dict) else m["away"]
        
        for team in [home, away]:
            if team in injuries:
                continue
            print(f"  搜索伤停: {team}")
            info = fetch_injury_data(team)
            if info:
                injuries[team] = info
                injury_updated += 1
    
    with open(injury_path, "w", encoding="utf-8") as f:
        json.dump(injuries, f, ensure_ascii=False, indent=2)
    
    # 更新专家意见
    expert_path = os.path.join(DATA_DIR, "expert_opinions.json")
    experts = {}
    if os.path.exists(expert_path):
        with open(expert_path, "r", encoding="utf-8") as f:
            experts = json.load(f)
    
    expert_updated = 0
    for m in matches[:5]:
        home = m["home"]["name"] if isinstance(m["home"], dict) else m["home"]
        away = m["away"]["name"] if isinstance(m["away"], dict) else m["away"]
        key = f"{home} vs {away}"
        
        if key in experts:
            continue
        print(f"  搜索专家: {key}")
        opinion = fetch_expert_opinion(home, away)
        if opinion:
            experts[key] = opinion
            expert_updated += 1
    
    with open(expert_path, "w", encoding="utf-8") as f:
        json.dump(experts, f, ensure_ascii=False, indent=2)
    
    print(f"伤停更新: {injury_updated}支, 专家意见更新: {expert_updated}场")

if __name__ == "__main__":
    print("=== 自动抓取分析数据 ===")
    auto_update_all()
    print("=== 完成 ===")
