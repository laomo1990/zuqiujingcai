"""
多维度推演引擎 v2.0
===========================
7个维度交叉验证，输出置信度+推演依据
"""

import math
from typing import Dict, List, Optional

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
    '欧冠': {'home_adv': 0.30, 'draw_rate': 0.25, 'avg_goals': 2.8},
    '欧联': {'home_adv': 0.28, 'draw_rate': 0.26, 'avg_goals': 2.7},
    '巴甲': {'home_adv': 0.40, 'draw_rate': 0.28, 'avg_goals': 2.3},
    '瑞超': {'home_adv': 0.35, 'draw_rate': 0.24, 'avg_goals': 2.8},
    '挪超': {'home_adv': 0.38, 'draw_rate': 0.23, 'avg_goals': 2.9},
    '芬超': {'home_adv': 0.32, 'draw_rate': 0.25, 'avg_goals': 2.5},
    'K联赛': {'home_adv': 0.35, 'draw_rate': 0.26, 'avg_goals': 2.6},
    '美职': {'home_adv': 0.42, 'draw_rate': 0.22, 'avg_goals': 3.0},
    'default': {'home_adv': 0.35, 'draw_rate': 0.25, 'avg_goals': 2.6}
}

def get_league_cfg(league):
    for k, v in LEAGUE_CFG.items():
        if k in league:
            return v
    return LEAGUE_CFG['default']

WEIGHTS = {
    'crs': 0.25,
    'market': 0.20,
    'handicap': 0.12,
    'htft': 0.08,
    'poisson': 0.10,
    'league': 0.05,
    'draw_signal': 0.05,
    'away_boost': 0.15
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

        dims = [crs_hda, market_hda, hc_hda, ht_hda, poisson_hda, league_hda, draw_sig, away_boost]
        keys = ['crs', 'market', 'handicap', 'htft', 'poisson', 'league', 'draw_signal', 'away_boost']

        ph = sum(d['h'] * WEIGHTS[k] for d, k in zip(dims, keys))
        pd = sum(d['d'] * WEIGHTS[k] for d, k in zip(dims, keys))
        pa = sum(d['a'] * WEIGHTS[k] for d, k in zip(dims, keys))

        total = ph + pd + pa
        if total > 0:
            ph /= total; pd /= total; pa /= total

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
            for i in range(7):
                for j in range(7):
                    p = poisson_pmf(i, hL) * poisson_pmf(j, aL)
                    if p > 0.01:
                        score_probs[f"{i}-{j}"] = p
        sr = sorted(score_probs.items(), key=lambda x: -x[1])[:5]

        o25_crs = sum(v for k, v in cp.items() if int(k[1:3]) + int(k[4:6]) >= 3) if cp else 0
        o25_p = sum(poisson_pmf(i, hL) * poisson_pmf(j, aL) for i in range(8) for j in range(8) if i + j >= 3)
        o25 = o25_crs * 0.6 + o25_p * 0.4 if cp else o25_p

        mk = market_hda
        mx = max(ph, pd, pa)
        pick = '主胜' if ph == mx else ('客胜' if pa == mx else '平局')

        self._build_reasons(crs_hda, market_hda, hc_hda, ht_hda, poisson_hda, league_hda, draw_sig, away_boost)

        return {
            'h': ph, 'd': pd, 'a': pa, 'hL': hL, 'aL': aL,
            'sr': [{'s': s, 'p': p} for s, p in sr],
            'o25': o25,
            'te': {'h': (ph - mk['h']) * 100, 'd': (pd - mk['d']) * 100, 'a': (pa - mk['a']) * 100},
            'val': {'h': ph - mk['h'], 'd': pd - mk['d'], 'a': pa - mk['a']},
            'conf': conf, 'ci': round(abs(ph - mx) * 100 + abs(pd - mx) * 100 + abs(pa - mx) * 100),
            'models': len([d for d in [crs_hda, market_hda, hc_hda, ht_hda] if d.get('status') == 'ok']),
            'pick': pick, 'pick_prob': round(mx * 100),
            'reasons': self.reasons, 'dimensions': self.dimensions
        }

    def _dim_crs(self):
        if not self.crs or len(self.crs) < 5:
            self.dimensions['crs'] = {'status': 'no_data', 'h': 0.45, 'd': 0.25, 'a': 0.30}
            return self.dimensions['crs']
        cp = crs_to_probs(self.crs)
        h = d = a = 0
        for k, v in cp.items():
            g, e = int(k[1:3]), int(k[4:6])
            if g > e: h += v
            elif g == e: d += v
            else: a += v
        t = h + d + a
        if t > 0: h /= t; d /= t; a /= t
        self.dimensions['crs'] = {'status': 'ok', 'h': h, 'd': d, 'a': a}
        return self.dimensions['crs']

    def _dim_market(self):
        o = self.odds
        if not o or not o.get('home') or not o.get('draw') or not o.get('away'):
            self.dimensions['market'] = {'status': 'no_data', 'h': 0.45, 'd': 0.25, 'a': 0.30}
            return self.dimensions['market']
        h, d, a = 1/o['home'], 1/o['draw'], 1/o['away']
        t = h + d + a
        self.dimensions['market'] = {'status': 'ok', 'h': h/t, 'd': d/t, 'a': a/t}
        return self.dimensions['market']

    def _dim_handicap(self):
        hc = self.handicap
        if not hc or not hc.get('home') or not hc.get('away'):
            self.dimensions['handicap'] = {'status': 'no_data', 'h': 0.40, 'd': 0.25, 'a': 0.35}
            return self.dimensions['handicap']
        d_o = hc.get('draw', 0)
        if d_o > 0:
            s = 1/hc['home'] + 1/d_o + 1/hc['away']
            h, d, a = (1/hc['home'])/s, (1/d_o)/s, (1/hc['away'])/s
        else:
            s = 1/hc['home'] + 1/hc['away']
            h, d, a = (1/hc['home'])/s, 0.15, (1/hc['away'])/s
            t = h + d + a; h /= t; d /= t; a /= t
        self.dimensions['handicap'] = {'status': 'ok', 'h': h, 'd': d, 'a': a, 'line': hc.get('line', '0')}
        return self.dimensions['handicap']

    def _dim_htft(self):
        if not self.htft:
            self.dimensions['htft'] = {'status': 'no_data', 'h': 0.40, 'd': 0.25, 'a': 0.35}
            return self.dimensions['htft']
        hh = hd = ha = 0
        for k, o in self.htft.items():
            if o <= 0: continue
            p = 1 / o
            if k[0] == 'h': hh += p
            elif k[0] == 'd': hd += p
            else: ha += p
        t = hh + hd + ha
        if t > 0:
            self.dimensions['htft'] = {'status': 'ok', 'h': hh/t, 'd': hd/t, 'a': ha/t}
        else:
            self.dimensions['htft'] = {'status': 'no_data', 'h': 0.40, 'd': 0.25, 'a': 0.35}
        return self.dimensions['htft']

    def _dim_poisson(self):
        cp = crs_to_probs(self.crs) if self.crs else {}
        h_xg = sum(int(k[1:3]) * v for k, v in cp.items()) if cp else 1.2
        a_xg = sum(int(k[4:6]) * v for k, v in cp.items()) if cp else 1.0
        hL = h_xg + self.lc['home_adv']
        aL = max(0.3, a_xg - self.lc['home_adv'] * 0.3)
        h = d = a = 0
        for i in range(8):
            for j in range(8):
                p = poisson_pmf(i, hL) * poisson_pmf(j, aL)
                if i > j: h += p
                elif i == j: d += p
                else: a += p
        self.dimensions['poisson'] = {'status': 'ok', 'h': h, 'd': d, 'a': a, 'hL': round(hL, 2), 'aL': round(aL, 2)}
        return self.dimensions['poisson']

    def _dim_league(self):
        lc = self.lc
        h = 0.45 + lc['home_adv'] * 0.2
        d = lc['draw_rate']
        a = max(0.1, 1 - h - d)
        t = h + d + a
        self.dimensions['league'] = {'status': 'ok', 'h': h/t, 'd': d/t, 'a': a/t, 'name': self.league}
        return self.dimensions['league']

    def _dim_away_boost(self):
        """维度8: 客胜增强信号"""
        h, d, a = 0.40, 0.25, 0.35
        signals = []
        
        # 信号1: 市场客胜赔率低（客队被看好）
        if self.odds.get('away'):
            away_odds = self.odds['away']
            if away_odds < 2.0:
                a += 0.15
                signals.append(f"客赔极低({away_odds})")
            elif away_odds < 2.5:
                a += 0.10
                signals.append(f"客赔偏低({away_odds})")
            elif away_odds < 3.0:
                a += 0.05
        
        # 信号2: 让球盘口+1以上（客队让球）
        line = float(self.handicap.get('line', '0') or '0')
        if line >= 1.0:
            a += 0.10
            signals.append(f"客队让{line}球")
        elif line >= 0.5:
            a += 0.05
            signals.append(f"客队让{line}球")
        
        # 信号3: CRS客胜概率高
        if self.crs:
            cp = crs_to_probs(self.crs)
            away_crs = sum(v for k, v in cp.items() if int(k[1:3]) < int(k[4:6]))
            if away_crs > 0.40:
                a += 0.08
                signals.append(f"CRS客胜{away_crs:.0%}")
            elif away_crs > 0.35:
                a += 0.04
        
        # 信号4: 半全场客胜概率高
        if self.htft:
            ah = sum(v for k, v in self.htft.items() if k[0] == 'a' and v > 0)
            total = sum(v for v in self.htft.values() if v > 0)
            if total > 0 and ah/total > 0.35:
                a += 0.05
                signals.append(f"半全场客胜{ah/total:.0%}")
        
        t = h + d + a
        self.dimensions['away_boost'] = {'status': 'ok', 'h': h/t, 'd': d/t, 'a': a/t, 'signals': signals}
        return self.dimensions['away_boost']

    def _dim_draw_signal(self):
        h, d, a = 0.40, 0.25, 0.35
        signals = []
        if self.odds.get('draw'):
            do = self.odds['draw']
            if do < 3.0:
                d += (3.0 - do) * 0.08
                signals.append(f"平赔低({do})")
            elif do < 3.5:
                d += (3.5 - do) * 0.04
        line = abs(float(self.handicap.get('line', '0') or '0'))
        if line <= 0.25:
            d += 0.05
            signals.append(f"盘口平手({line})")
        if self.crs:
            cp = crs_to_probs(self.crs)
            dp = sum(v for k, v in cp.items() if k[1:3] == k[4:6])
            if dp > 0.25:
                d += 0.03
                signals.append(f"CRS平局{dp:.0%}")
        t = h + d + a
        self.dimensions['draw_signal'] = {'status': 'ok', 'h': h/t, 'd': d/t, 'a': a/t, 'signals': signals}
        return self.dimensions['draw_signal']

    def _calc_confidence(self, models):
        valid = [m for m in models if m.get('status') == 'ok']
        if len(valid) < 2: return 1
        max_std = 0
        for key in ['h', 'd', 'a']:
            vals = [m[key] for m in valid]
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            max_std = max(max_std, math.sqrt(variance))
        if max_std < 0.03: return 5
        if max_std < 0.06: return 4
        if max_std < 0.10: return 3
        if max_std < 0.15: return 2
        return 1

    def _build_reasons(self, crs, market, hc, ht, poisson, league, draw_sig, away_boost=None):
        self.reasons = []
        if crs.get('status') == 'ok':
            self.reasons.append(f"CRS赔率：主{crs['h']:.0%} 平{crs['d']:.0%} 客{crs['a']:.0%}")
        if market.get('status') == 'ok':
            self.reasons.append(f"市场概率：主{market['h']:.0%} 平{market['d']:.0%} 客{market['a']:.0%}")
        if hc.get('status') == 'ok':
            self.reasons.append(f"让球({hc.get('line','0')})：主{hc['h']:.0%} 平{hc['d']:.0%} 客{hc['a']:.0%}")
        if poisson.get('status') == 'ok':
            self.reasons.append(f"泊松：进球主{poisson['hL']}客{poisson['aL']}")
        if league.get('status') == 'ok':
            self.reasons.append(f"{league['name']}：主场优势{self.lc['home_adv']:.0%}")
        if draw_sig.get('signals'):
            self.reasons.append(f"平局信号：{', '.join(draw_sig['signals'])}")
        if away_boost and away_boost.get('signals'):
            self.reasons.append(f"客胜信号：{', '.join(away_boost['signals'])}")


def predict_match(match):
    if not match.get('crs'):
        return None
    engine = PredictionEngine(match)
    result = engine.analyze()
    home = match['home']['name'] if isinstance(match['home'], dict) else match['home']
    away = match['away']['name'] if isinstance(match['away'], dict) else match['away']
    return {
        'home': home, 'away': away,
        'date': match.get('date', ''),
        'league': match.get('league', ''),
        'pick': result['pick'],
        'prob': result['pick_prob'],
        'score': result['sr'][0]['s'] if result['sr'] else '1-0',
        'ou': '大2.5' if result['o25'] > 0.5 else '小2.5',
        'odds': match['odds'],
        'reasons': result['reasons'],
        'confidence': result['conf'],
        'dimensions': {k: {'h': round(v.get('h',0)*100), 'd': round(v.get('d',0)*100), 'a': round(v.get('a',0)*100)}
                       for k, v in result.get('dimensions', {}).items() if isinstance(v, dict)}
    }
