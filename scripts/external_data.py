"""
外部数据源模块
================
- 天气数据: Open-Meteo API (免费)
- 伤情数据: 手动维护/自动抓取
- 专家意见: 手动维护/自动抓取
"""

import json
import os
import re
from urllib.request import urlopen, Request
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# ===== 天气数据 =====

def load_city_coords():
    """加载城市坐标"""
    path = os.path.join(DATA_DIR, 'city_coords.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

CITY_COORDS = load_city_coords()

def get_weather(lat, lon):
    """获取天气数据"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&timezone=auto"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        cw = data.get('current_weather', {})
        return {
            'temp': cw.get('temperature', 20),
            'wind': cw.get('windspeed', 10),
            'code': cw.get('weathercode', 0),
            'desc': weather_desc(cw.get('weathercode', 0))
        }
    except:
        return None

def weather_desc(code):
    """天气代码转描述"""
    descs = {
        0: '晴', 1: '晴间多云', 2: '多云', 3: '阴',
        45: '雾', 48: '雾凇', 51: '小毛毛雨', 53: '毛毛雨',
        55: '大毛毛雨', 61: '小雨', 63: '中雨', 65: '大雨',
        71: '小雪', 73: '中雪', 75: '大雪', 77: '雪粒',
        80: '小阵雨', 81: '阵雨', 82: '大阵雨',
        85: '小阵雪', 86: '大阵雪',
        95: '雷暴', 96: '雷暴+小冰雹', 99: '雷暴+大冰雹'
    }
    return descs.get(code, f'代码{code}')

def find_city_coords(team_name, league):
    """根据球队名/联赛查找城市坐标"""
    # 直接匹配
    for city, coords in CITY_COORDS.items():
        if city in team_name:
            return coords
    
    # 联赛默认城市
    league_defaults = {
        '日职': (35.68, 139.69), '日乙': (35.68, 139.69),
        'K联赛': (37.57, 126.98),
        '巴甲': (-23.55, -46.63), '巴乙': (-23.55, -46.63),
        '英超': (51.51, -0.13), '德甲': (48.14, 11.58),
        '西甲': (40.42, -3.70), '意甲': (41.90, 12.50),
        '法甲': (48.86, 2.35), '荷甲': (52.37, 4.90),
        '瑞超': (59.33, 18.07), '挪超': (59.91, 10.75),
        '芬超': (60.17, 24.94),
        '欧冠': None, '欧联': None,
        '美职': (40.71, -74.01),
    }
    
    for key, coords in league_defaults.items():
        if key in league:
            return coords
    
    return None


# ===== 伤情数据 =====

def load_injuries():
    """加载伤情数据"""
    path = os.path.join(DATA_DIR, 'injuries.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def get_team_injuries(team_name):
    """获取球队伤停信息"""
    injuries = load_injuries()
    
    # 直接匹配
    if team_name in injuries:
        return injuries[team_name]
    
    # 模糊匹配
    for key, val in injuries.items():
        if key in team_name or team_name in key:
            return val
    
    return None


# ===== 专家意见 =====

def load_expert_opinions():
    """加载专家意见"""
    path = os.path.join(DATA_DIR, 'expert_opinions.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def get_expert_opinion(home, away):
    """获取专家意见"""
    experts = load_expert_opinions()
    key = f"{home} vs {away}"
    if key in experts:
        return experts[key]
    # 反向匹配
    key2 = f"{away} vs {home}"
    if key2 in experts:
        return experts[key2]
    return None


def fetch_expert_from_dongqiudi():
    """尝试从懂球帝抓取专家意见（备用）"""
    try:
        url = "https://www.dongqiudi.com/special/47"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        with urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        
        # 简单提取分析内容
        analyses = re.findall(r'分析[^<]{0,300}', html)
        predictions = re.findall(r'预测[^<]{0,200}', html)
        
        return {
            'analyses': analyses[:3],
            'predictions': predictions[:3]
        }
    except:
        return None
