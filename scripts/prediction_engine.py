"""
多维度推演引擎 v3.0
===========================
优化：降低主胜偏向、联赛差异化、平局增强、客队修正
"""

import math
import json
import os
from typing import Dict, List, Optional
from collections import defaultdict
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from external_data import get_weather, find_city_coords, get_team_injuries, get_expert_opinion

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0: return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def crs_to_probs(crs: Dict[str, float]) -> Dict[str, float]:
    probs = {}
    total = 0
    for k, odds in crs.items():
        if odds > 0:
            probs[k] = 1 / odds
            total += probs[k]
    if total > 0:
        for k in probs:
            probs[k] /= total
    return probs

LEAGUE_CFG = {
    '欧冠': {'home_adv': 0.25, 'draw_rate': 0.27, 'avg_goals': 2.8},
    '欧联': {'home_adv': 0.23, 'draw_rate': 0.28, 'avg_goals': 2.7},
    '欧洲超级杯': {'home_adv': 0.20, 'draw_rate': 0.28, 'avg_goals': 2.5},
    '巴甲': {'home_adv': 0.35, 'draw_rate': 0.28, 'avg_goals': 2.3},
    '巴西甲': {'home_adv': 0.35, 'draw_rate': 0.28, 'avg_goals': 2.3},
    '巴西杯': {'home_adv': 0.30, 'draw_rate': 0.30, 'avg_goals': 2.2},
    '解放者杯': {'home_adv': 0.28, 'draw_rate': 0.30, 'avg_goals': 2.3},
    '瑞超': {'home_adv': 0.30, 'draw_rate': 0.25, 'avg_goals': 2.8},
    '瑞典超': {'home_adv': 0.30, 'draw_rate': 0.25, 'avg_goals': 2.8},
    '挪超': {'home_adv': 0.32, 'draw_rate': 0.24, 'avg_goals': 2.9},
    '芬超': {'home_adv': 0.25, 'draw_rate': 0.28, 'avg_goals': 2.4},
    'K联赛': {'home_adv': 0.30, 'draw_rate': 0.27, 'avg_goals': 2.5},
    '美职': {'home_adv': 0.38, 'draw_rate': 0.23, 'avg_goals': 3.0},
    '荷甲': {'home_adv': 0.32, 'draw_rate': 0.24, 'avg_goals': 2.9},
    '德乙': {'home_adv': 0.30, 'draw_rate': 0.26, 'avg_goals': 2.7},
    '葡超': {'home_adv': 0.30, 'draw_rate': 0.26, 'avg_goals': 2.5},
    'J联赛': {'home_adv': 0.28, 'draw_rate': 0.27, 'avg_goals': 2.5},
    'J2联赛': {'home_adv': 0.25, 'draw_rate': 0.28, 'avg_goals': 2.3},
    'default': {'home_adv': 0.28, 'draw_rate': 0.27, 'avg_goals': 2.5}
}

def get_league_cfg(league):
    for k, v in LEAGUE_CFG.items():
        if k in league:
            return v
    return LEAGUE_CFG['default']

def load_team_stats():
    """从历史数据加载球队战绩"""
    stats = defaultdict(lambda: {'w': 0, 'd': 0, 'l': 0, 'gf': 0, 'ga': 0, 'matches': 0, 'recent': [], 'avg_goals': 0})
    history_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'history.json')
    if not os.path.exists(history_path):
        return stats
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            h = json.load(f)
        for m in h.get('matches', []):
            home, away = m['home'], m['away']
            hs, as_ = m['home_score'], m['away_score']
            stats[home]['matches'] += 1
            stats[home]['gf'] += hs
            stats[home]['ga'] += as_
            if hs > as_: stats[home]['w'] += 1; stats[home]['recent'].append('W')
            elif hs == as_: stats[home]['d'] += 1; stats[home]['recent'].append('D')
            else: stats[home]['l'] += 1; stats[home]['recent'].append('L')
            stats[away]['matches'] += 1
            stats[away]['gf'] += as_
            stats[away]['ga'] += hs
            if as_ > hs: stats[away]['w'] += 1; stats[away]['recent'].append('W')
            elif hs == as_: stats[away]['d'] += 1; stats[away]['recent'].append('D')
            else: stats[away]['l'] += 1; stats[away]['recent'].append('L')
    except:
        pass
    return stats

TEAM_STATS = load_team_stats()

WEIGHTS = {
    'crs': 0.20,
    'market': 0.15,
    'handicap': 0.08,
    'htft': 0.05,
    'poisson': 0.06,
    'league': 0.03,
    'draw_signal': 0.06,
    'away_boost': 0.12,
    'recent_form': 0.08,
    'head_to_head': 0.04,
    'schedule': 0.03,
    'weather': 0.03,
    'injury': 0.04,
    'expert': 0.03
}

class PredictionEngine:
    def __init__(self, match):
        self.match = match
        self.odds = match.get('odds', {})
        self.crs = match.get('crs', {})
        self.handicap = match.get('handicap', {})
        self.htft = match.get('htft', {})
        self.league = match.get('league', '')
        self.lc = get_league_cfg(self.league)
        self.dimensions = {}
        self.reasons = []

    def analyze(self):
        crs_hda = self._dim_crs()
        market_hda = self._dim_market()
        hc_hda = self._dim_handicap()
        ht_hda = self._dim_htft()
        poisson_hda = self._dim_poisson()
        league_hda = self._dim_league()
        draw_sig = self._dim_draw_signal()
        away_boost = self._dim_away_boost()
        recent_form = self._dim_recent_form()
        head_to_head = self._dim_head_to_head()
        schedule = self._dim_schedule()
        weather = self._dim_weather()
        injury = self._dim_injury()
        expert = self._dim_expert()

        dims = [crs_hda, market_hda, hc_hda, ht_hda, poisson_hda, league_hda, draw_sig, away_boost, recent_form, head_to_head, schedule, weather, injury, expert]
        keys = ['crs', 'market', 'handicap', 'htft', 'poisson', 'league', 'draw_signal', 'away_boost', 'recent_form', 'head_to_head', 'schedule', 'weather', 'injury', 'expert']

        ph = sum(d['h'] * WEIGHTS[k] for d, k in zip(dims, keys))
        pd = sum(d['d'] * WEIGHTS[k] for d, k in zip(dims, keys))
        pa = sum(d['a'] * WEIGHTS[k] for d, k in zip(dims, keys))

        total = ph + pd + pa
        if total > 0:
            ph /= total; pd /= total; pa /= total

        # v3优化：主胜修正 - 当多个维度显示平局/客胜时，降低主胜
        if pd > 0.30 or pa > 0.30:
            ph_adj = ph * 0.90
            pd_adj = pd * 1.05 if pd > 0.25 else pd
            pa_adj = pa * 1.05 if pa > 0.25 else pa
            t2 = ph_adj + pd_adj + pa_adj
            if t2 > 0:
                ph = ph_adj / t2
                pd = pd_adj / t2
                pa = pa_adj / t2

        conf = self._calc_confidence([crs_hda, market_hda, hc_hda, ht_hda, poisson_hda])

        cp = crs_to_probs(self.crs) if self.crs else {}
        h_xg = sum(int(k[1:3]) * v for k, v in cp.items()) if cp else 1.2
        a_xg = sum(int(k[4:6]) * v for k, v in cp.items()) if cp else 1.0
        hL = h_xg + self.lc['home_adv']
        aL = max(0.3, a_xg - self.lc['home_adv'] * 0.3)

        score_probs = {}
        if cp:
            for k, v in cp.items():
                score_probs[f"{int(k[1:3])}-{int(k[4:6])}"] = v
        else:
            for hi in range(4):
                for ai in range(4):
                    p = poisson_pmf(hi, hL) * poisson_pmf(ai, aL)
                    score_probs[f"{hi}-{ai}"] = p

        sr = sorted(score_probs.items(), key=lambda x: -x[1])[:5]
        o25 = sum(v for k, v in score_probs.items() if int(k.split('-')[0]) + int(k.split('-')[1]) > 2)

        market_total = 1/self.odds.get('home', 99) + 1/self.odds.get('draw', 99) + 1/self.odds.get('away', 99)
        mk = {
            'h': (1/self.odds.get('home', 99)) / market_total if market_total else 0.33,
            'd': (1/self.odds.get('draw', 99)) / market_total if market_total else 0.33,
            'a': (1/self.odds.get('away', 99)) / market_total if market_total else 0.33
        }
        te = {'h': (ph - mk['h']) * 100, 'd': (pd - mk['d']) * 100, 'a': (pa - mk['a']) * 100}
        val = {k: v for k, v in te.items() if v > 3}

        self._build_reasons(crs_hda, market_hda, hc_hda, ht_hda, poisson_hda, league_hda, draw_sig, away_boost, recent_form, head_to_head, schedule, weather, injury, expert)

        mx = max(ph, pd, pa)
        if ph == mx: pick = '主胜'
        elif pa == mx: pick = '客胜'
        else: pick = '平局'

        return {
            'h': ph, 'd': pd, 'a': pa,
            'conf': conf,
            'pick': pick,
            'prob': round(mx * 100),
            'score': sr[0][0] if sr else '1-0',
            'ou': '大2.5' if o25 > 0.5 else '小2.5',
            'sr': [{'s': s, 'p': p} for s, p in sr],
            'o25': o25,
            'hL': hL, 'aL': aL,
            'te': te, 'val': val,
            'reasons': self.reasons,
            'dimensions': self.dimensions
        }

    def _dim_crs(self):
        """维度1: CRS比分赔率"""
        if not self.crs:
            return {'h': 0.40, 'd': 0.25, 'a': 0.35, 'status': 'no_data'}

        cp = crs_to_probs(self.crs)
        h = d = a = 0
        for k, p in cp.items():
            home_goals = int(k[1:3])
            away_goals = int(k[4:6])
            if home_goals > away_goals: h += p
            elif home_goals == away_goals: d += p
            else: a += p

        t = h + d + a
        if t > 0: h /= t; d /= t; a /= t
        self.dimensions['crs'] = {'h': h, 'd': d, 'a': a, 'status': 'ok'}
        return self.dimensions['crs']

    def _dim_market(self):
        """维度2: 市场隐含概率"""
        h_odds = self.odds.get('home', 1)
        d_odds = self.odds.get('draw', 1)
        a_odds = self.odds.get('away', 1)

        total = 1/h_odds + 1/d_odds + 1/a_odds
        h = (1/h_odds) / total
        d = (1/d_odds) / total
        a = (1/a_odds) / total

        self.dimensions['market'] = {'h': h, 'd': d, 'a': a, 'status': 'ok'}
        return self.dimensions['market']

    def _dim_handicap(self):
        """维度3: 让球盘口"""
        hc = self.handicap
        if not hc or 'line' not in hc:
            return {'h': 0.40, 'd': 0.25, 'a': 0.35, 'status': 'no_data'}

        line = float(hc.get('line', 0))
        h = 0.40
        d = 0.25
        a = 0.35

        # 让球修正
        if line <= -1.5: a += 0.20; h -= 0.15
        elif line <= -0.5: a += 0.10; h -= 0.05
        elif line >= 1.5: h += 0.20; a -= 0.15
        elif line >= 0.5: h += 0.10; a -= 0.05
        else: d += 0.05  # 平手盘倾向平局

        t = h + d + a
        if t > 0: h /= t; d /= t; a /= t
        self.dimensions['handicap'] = {'h': h, 'd': d, 'a': a, 'status': 'ok'}
        return self.dimensions['handicap']

    def _dim_htft(self):
        """维度4: 半全场"""
        ht = self.htft
        if not ht:
            return {'h': 0.40, 'd': 0.25, 'a': 0.35, 'status': 'no_data'}

        h = ht.get('HH', 0) + ht.get('DH', 0) * 0.5 + ht.get('AH', 0) * 0.3
        d = ht.get('DD', 0) + ht.get('HD', 0) * 0.5 + ht.get('AD', 0) * 0.5
        a = ht.get('AA', 0) + ht.get('DA', 0) * 0.5 + ht.get('HA', 0) * 0.3

        total = sum(1/v for v in ht.values() if v > 0) if ht else 1
        if total > 0:
            h = (1/h if h > 0 else 0.33) / total if total else 0.33
            d = (1/d if d > 0 else 0.33) / total if total else 0.33
            a = (1/a if a > 0 else 0.33) / total if total else 0.33

        t = h + d + a
        if t > 0: h /= t; d /= t; a /= t
        self.dimensions['htft'] = {'h': h, 'd': d, 'a': a, 'status': 'ok'}
        return self.dimensions['htft']

    def _dim_poisson(self):
        """维度5: 泊松分布"""
        cp = crs_to_probs(self.crs) if self.crs else {}
        h_xg = sum(int(k[1:3]) * v for k, v in cp.items()) if cp else 1.2
        a_xg = sum(int(k[4:6]) * v for k, v in cp.items()) if cp else 1.0
        hL = h_xg + self.lc['home_adv']
        aL = max(0.3, a_xg - self.lc['home_adv'] * 0.3)

        h = d = a = 0
        for hi in range(8):
            for ai in range(8):
                p = poisson_pmf(hi, hL) * poisson_pmf(ai, aL)
                if hi > ai: h += p
                elif hi == ai: d += p
                else: a += p

        t = h + d + a
        if t > 0: h /= t; d /= t; a /= t
        self.dimensions['poisson'] = {'h': h, 'd': d, 'a': a, 'status': 'ok'}
        return self.dimensions['poisson']

    def _dim_league(self):
        """维度6: 联赛特征"""
        h = 0.35 + self.lc['home_adv'] * 0.3
        d = self.lc['draw_rate']
        a = 1 - h - d
        if a < 0.15: a = 0.15; h = 1 - d - a

        t = h + d + a
        if t > 0: h /= t; d /= t; a /= t
        self.dimensions['league'] = {'h': h, 'd': d, 'a': a, 'status': 'ok'}
        return self.dimensions['league']

    def _dim_draw_signal(self):
        """维度7: 平局信号增强"""
        h, d, a = 0.35, 0.30, 0.35
        signals = []

        # CRS平局概率
        if self.crs:
            cp = crs_to_probs(self.crs)
            draw_prob = sum(v for k, v in cp.items() if k[1:3] == k[4:6])
            if draw_prob > 0.25:
                d += 0.10
                signals.append(f"CRS平局{round(draw_prob*100)}%")

        # 市场平局赔率
        d_odds = self.odds.get('draw', 99)
        if d_odds < 3.2:
            d += 0.08
            signals.append(f"平赔偏低({d_odds})")
        elif d_odds < 3.5:
            d += 0.04
            signals.append(f"平赔适中({d_odds})")

        # 让球盘口接近0
        hc = self.handicap
        if hc and 'line' in hc:
            line = abs(float(hc.get('line', 0)))
            if line <= 0.25:
                d += 0.06
                signals.append("盘口接近平手")

        t = h + d + a
        if t > 0: h /= t; d /= t; a /= t
        self.dimensions['draw_signal'] = {'h': h, 'd': d, 'a': a, 'signals': signals, 'status': 'ok'}
        return self.dimensions['draw_signal']

    def _dim_away_boost(self):
        """维度8: 客胜增强"""
        h, d, a = 0.35, 0.25, 0.40
        signals = []

        a_odds = self.odds.get('away', 99)
        if a_odds < 1.8:
            a += 0.18
            signals.append(f"客赔极低({a_odds})")
        elif a_odds < 2.0:
            a += 0.12
            signals.append(f"客赔偏低({a_odds})")
        elif a_odds < 2.5:
            a += 0.06
            signals.append(f"客赔适中({a_odds})")

        hc = self.handicap
        if hc and 'line' in hc:
            line = float(hc.get('line', 0))
            if line >= 1.0:
                a += 0.10
                signals.append(f"客队让{line}球")

        if self.crs:
            cp = crs_to_probs(self.crs)
            away_prob = sum(v for k, v in cp.items() if int(k[4:6]) > int(k[1:3]))
            if away_prob > 0.40:
                a += 0.08
                signals.append(f"CRS客胜{round(away_prob*100)}%")

        if self.htft:
            aa_odds = self.htft.get('AA', 99)
            if aa_odds < 5:
                a += 0.05
                signals.append("半全场客客偏低")

        t = h + d + a
        if t > 0: h /= t; d /= t; a /= t
        self.dimensions['away_boost'] = {'h': h, 'd': d, 'a': a, 'signals': signals, 'status': 'ok'}
        return self.dimensions['away_boost']

    def _dim_recent_form(self):
        """维度9: 近期战绩"""
        home = self.match['home']['name'] if isinstance(self.match['home'], dict) else self.match['home']
        away = self.match['away']['name'] if isinstance(self.match['away'], dict) else self.match['away']

        h_stats = TEAM_STATS.get(home, {'w': 0, 'd': 0, 'l': 0, 'recent': [], 'matches': 0})
        a_stats = TEAM_STATS.get(away, {'w': 0, 'd': 0, 'l': 0, 'recent': [], 'matches': 0})

        h_total = h_stats['matches'] or 1
        a_total = a_stats['matches'] or 1

        h_wr = h_stats['w'] / h_total
        a_wr = a_stats['w'] / a_total

        h = 0.35 + h_wr * 0.20
        a = 0.35 + a_wr * 0.20
        d = max(0.15, 1 - h - a)

        t = h + d + a
        if t > 0: h /= t; d /= t; a /= t
        self.dimensions['recent_form'] = {'h': h, 'd': d, 'a': a, 'status': 'ok'}
        return self.dimensions['recent_form']

    def _dim_head_to_head(self):
        """维度10: 历史交锋"""
        h, d, a = 0.40, 0.25, 0.35
        self.dimensions['head_to_head'] = {'h': h, 'd': d, 'a': a, 'status': 'no_data'}
        return self.dimensions['head_to_head']

    def _dim_schedule(self):
        """维度11: 赛程密度"""
        h, d, a = 0.40, 0.25, 0.35
        self.dimensions['schedule'] = {'h': h, 'd': d, 'a': a, 'status': 'ok'}
        return self.dimensions['schedule']

    def _dim_weather(self):
        """维度12: 天气因素"""
        h, d, a = 0.40, 0.25, 0.35
        signals = []

        home = self.match['home']['name'] if isinstance(self.match['home'], dict) else self.match['home']
        coords = find_city_coords(home, self.league)

        if coords:
            weather = get_weather(coords[0], coords[1])
            if weather:
                temp = weather['temp']
                wind = weather['wind']
                desc = weather['desc']

                if temp > 35:
                    h += 0.04
                    signals.append(f"高温{temp}°C({desc})")
                elif temp < 0:
                    d += 0.02
                    signals.append(f"低温{temp}°C({desc})")

                if wind > 30:
                    d += 0.03
                    signals.append(f"大风{wind}km/h")

                if weather['code'] >= 61:
                    d += 0.02
                    signals.append(f"{desc}天气")

                if not signals:
                    signals.append(f"{temp}°C {desc} 风{wind}km/h")

        t = h + d + a
        if t > 0: h /= t; d /= t; a /= t
        self.dimensions['weather'] = {'h': h, 'd': d, 'a': a, 'signals': signals, 'status': 'ok'}
        return self.dimensions['weather']

    def _dim_injury(self):
        """维度13: 伤情信息"""
        h, d, a = 0.40, 0.25, 0.35
        signals = []

        home = self.match['home']['name'] if isinstance(self.match['home'], dict) else self.match['home']
        away = self.match['away']['name'] if isinstance(self.match['away'], dict) else self.match['away']

        h_inj = get_team_injuries(home)
        a_inj = get_team_injuries(away)

        if h_inj:
            impact = h_inj.get('impact', 'low')
            total = len(h_inj.get('injuries', [])) + len(h_inj.get('suspensions', []))
            if total > 0:
                if impact == 'high':
                    a += 0.08
                    signals.append(f"主队伤停{total}人(影响大)")
                elif impact == 'medium':
                    a += 0.04
                    signals.append(f"主队伤停{total}人")
                else:
                    a += 0.02
                    signals.append(f"主队伤停{total}人(影响小)")

        if a_inj:
            impact = a_inj.get('impact', 'low')
            total = len(a_inj.get('injuries', [])) + len(a_inj.get('suspensions', []))
            if total > 0:
                if impact == 'high':
                    h += 0.08
                    signals.append(f"客队伤停{total}人(影响大)")
                elif impact == 'medium':
                    h += 0.04
                    signals.append(f"客队伤停{total}人")
                else:
                    h += 0.02
                    signals.append(f"客队伤停{total}人(影响小)")

        if not signals:
            signals.append("无伤停信息")

        t = h + d + a
        if t > 0: h /= t; d /= t; a /= t
        self.dimensions['injury'] = {'h': h, 'd': d, 'a': a, 'signals': signals, 'status': 'ok'}
        return self.dimensions['injury']

    def _dim_expert(self):
        """维度14: 专家意见"""
        h, d, a = 0.40, 0.25, 0.35
        signals = []

        home = self.match['home']['name'] if isinstance(self.match['home'], dict) else self.match['home']
        away = self.match['away']['name'] if isinstance(self.match['away'], dict) else self.match['away']

        opinion = get_expert_opinion(home, away)

        if opinion:
            consensus = opinion.get('consensus', '')
            confidence = opinion.get('confidence', 0.5)
            reason = opinion.get('reason', '')

            boost = confidence * 0.15

            if consensus == '主胜':
                h += boost
                signals.append(f"专家看好主队({confidence:.0%})")
            elif consensus == '客胜':
                a += boost
                signals.append(f"专家看好客队({confidence:.0%})")
            elif consensus == '平局':
                d += boost
                signals.append(f"专家看好平局({confidence:.0%})")

            if reason:
                signals.append(f"依据: {reason[:30]}")

        if not signals:
            signals.append("无专家意见")

        t = h + d + a
        if t > 0: h /= t; d /= t; a /= t
        self.dimensions['expert'] = {'h': h, 'd': d, 'a': a, 'signals': signals, 'status': 'ok'}
        return self.dimensions['expert']

    def _calc_confidence(self, dims):
        """计算置信度"""
        picks = []
        for d in dims:
            mx = max(d['h'], d['d'], d['a'])
            if d['h'] == mx: picks.append('h')
            elif d['d'] == mx: picks.append('d')
            else: picks.append('a')

        from collections import Counter
        c = Counter(picks)
        most = c.most_common(1)[0][1]

        if most >= 5: return 5
        elif most >= 4: return 4
        elif most >= 3: return 3
        elif most >= 2: return 2
        else: return 1

    def _build_reasons(self, crs, market, hc, ht, poisson, league, draw_sig, away_boost, recent_form, head_to_head, schedule, weather, injury, expert):
        """构建推演依据"""
        self.reasons = []

        if crs.get('status') == 'ok':
            self.reasons.append(f"CRS赔率：主{round(crs['h']*100)}% 平{round(crs['d']*100)}% 客{round(crs['a']*100)}%")

        if market.get('status') == 'ok':
            self.reasons.append(f"市场概率：主{round(market['h']*100)}% 平{round(market['d']*100)}% 客{round(market['a']*100)}%")

        if hc.get('status') == 'ok':
            line = self.handicap.get('line', '?')
            self.reasons.append(f"让球({line})：主{round(hc['h']*100)}% 平{round(hc['d']*100)}% 客{round(hc['a']*100)}%")

        if poisson.get('status') == 'ok':
            self.reasons.append(f"泊松：进球主{poisson.get('hL', 0):.1f}客{poisson.get('aL', 0):.1f}")

        self.reasons.append(f"{self.league}：主场优势{self.lc['home_adv']}")

        if draw_sig.get('signals'):
            self.reasons.append(f"平局信号：{', '.join(draw_sig['signals'])}")

        if away_boost.get('signals'):
            self.reasons.append(f"客胜信号：{', '.join(away_boost['signals'])}")

        if recent_form.get('status') == 'ok':
            self.reasons.append(f"近期战绩参考")

        if head_to_head.get('status') == 'ok':
            self.reasons.append(f"交锋记录参考")

        if weather.get('signals'):
            self.reasons.append(f"天气因素：{', '.join(weather['signals'])}")

        if injury.get('signals'):
            self.reasons.append(f"伤情信息：{', '.join(injury['signals'])}")

        if expert.get('signals'):
            self.reasons.append(f"专家意见：{', '.join(expert['signals'])}")


def predict_match(match):
    """预测单场比赛"""
    try:
        engine = PredictionEngine(match)
        result = engine.analyze()

        home = match['home']['name'] if isinstance(match['home'], dict) else match['home']
        away = match['away']['name'] if isinstance(match['away'], dict) else match['away']
        league = match.get('league', '')

        return {
            'home': home,
            'away': away,
            'league': league,
            'time': match.get('time', ''),
            'pick': result['pick'],
            'prob': result['prob'],
            'score': result['score'],
            'ou': result['ou'],
            'confidence': result['conf'],
            'h': result['h'],
            'd': result['d'],
            'a': result['a'],
            'sr': result['sr'],
            'o25': result['o25'],
            'hL': result['hL'],
            'aL': result['aL'],
            'te': result['te'],
            'val': result['val'],
            'reasons': result['reasons'],
            'dimensions': result['dimensions']
        }
    except Exception as e:
        return None
