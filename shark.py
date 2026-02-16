#!/usr/bin/env python3
import requests, json, os, time, tempfile, numpy as np, collections as c, threading, queue, random
from datetime import datetime
from dataclasses import dataclass, field
from typing import List

#=============== CONFIG ===============
@dataclass
class Config:
    TG_TOKEN:str=os.getenv("TG_TOKEN","")
    CHAT_ID:str=os.getenv("CHAT_ID","")
    INITIAL_CAPITAL:float=float(os.getenv("CAPITAL",4))
    MAX_POSITIONS:int=int(os.getenv("MAX_POSITIONS",3))
    MIN_POSITION:float=float(os.getenv("MIN_TRADE",0.05))
    MAX_POSITION_PERCENT:float=0.25
    BASE_STOP_LOSS:float=0.03
    BASE_TAKE_PROFIT:float=0.08
    TRAILING_STOP:float=0.04
    MAX_STOP_LOSS:float=0.08
    MAX_TAKE_PROFIT:float=0.25
    MIN_LIQUIDITY:float=float(os.getenv("MIN_LIQ",4000))
    MIN_VOLUME:float=float(os.getenv("MIN_VOL",800))
    MIN_SCORE:float=float(os.getenv("SCORE_THRESHOLD",60))
    AI_LEARNING_RATE:float=0.01
    AI_DISCOUNT_FACTOR:float=0.95
    AI_EXPLORATION_RATE:float=0.1
    AI_MEMORY_SIZE:int=10000
    SCAN_INTERVAL:int=30
    API_TIMEOUT:int=10
    MAX_RETRIES:int=3
    LOCK_FILE:str="shark.lock"
    STATE_FILE:str="shark_state.json"
    PROFIT_THRESHOLDS:List[float]=field(default_factory=lambda:[0.05,0.10,0.15,0.20,0.25])
    DEX_API:str="https://api.dexscreener.com/latest/dex/pairs/ton"
    USER_AGENT:str="WhiteSharkX/3.0"
    def __post_init__(s):
        if s.MIN_POSITION<0.01: raise ValueError("MIN_POSITION must be >= 0.01")
        if s.BASE_STOP_LOSS>=s.BASE_TAKE_PROFIT: raise ValueError("STOP_LOSS must be less than TAKE_PROFIT")

C=Config()

#=============== ENUMS ===============
class E:
    SIGNAL={'SB':'🔥 خرید قوی','B':'✅ خرید','W':'👀 تحت نظر','H':'⏳','S':'💰 فروش','SS':'⚡️','E':'❌'}
    RISK={'VL':'🟢 خیلی کم','L':'🟡 کم','M':'🟠 متوسط','H':'🔴 زیاد','EX':'💀'}
    REGIME={'BULL':'🐂','BEAR':'🐻','SIDE':'⚖️','VOL':'🌪️'}

#=============== UTILS ===============
class F:
    @staticmethod
    def save(p,d):
        try:
            with tempfile.NamedTemporaryFile('w',delete=False) as t:
                json.dump(d,t,indent=2)
                os.replace(t.name,p)
        except Exception as e:
            print("Error saving state:", e)
    @staticmethod
    def load(p,d):
        try: return json.load(open(p)) if os.path.exists(p) else d
        except: return d

#=============== AI ENGINE ===============
class NN:
    def __init__(s,i=10,h=20,o=3):
        s.W1=np.random.randn(i,h)*0.1
        s.b1=np.zeros((1,h))
        s.W2=np.random.randn(h,o)*0.1
        s.b2=np.zeros((1,o))
        s.lr=C.AI_LEARNING_RATE
    def f(s,x): return 1/(1+np.exp(-np.clip(x,-500,500)))
    def fd(s,x): return x*(1-x)
    def forward(s,X):
        s.z1=np.dot(X,s.W1)+s.b1
        s.a1=s.f(s.z1)
        s.z2=np.dot(s.a1,s.W2)+s.b2
        s.a2=s.f(s.z2)
        return s.a2
    def train(s,X,y):
        o=s.forward(X)
        m=X.shape[0]
        dZ2=o-y
        dW2=(1/m)*np.dot(s.a1.T,dZ2)
        db2=(1/m)*np.sum(dZ2,0,keepdims=True)
        dA1=np.dot(dZ2,s.W2.T)
        dZ1=dA1*s.fd(s.a1)
        dW1=(1/m)*np.dot(X.T,dZ1)
        db1=(1/m)*np.sum(dZ1,0,keepdims=True)
        s.W2-=s.lr*dW2; s.b2-=s.lr*db2
        s.W1-=s.lr*dW1; s.b1-=s.lr*db1
    def predict(s,X): return s.forward(X)

class QL:
    def __init__(s,ss,as_):
        s.m=c.deque(maxlen=C.AI_MEMORY_SIZE)
        s.ep=C.AI_EXPLORATION_RATE
        s.g=C.AI_DISCOUNT_FACTOR
        s.q=NN(ss,32,as_)
        s.t=NN(ss,32,as_)
    def act(s,st):
        if np.random.rand()<=s.ep: return random.randrange(3)
        return np.argmax(s.q.predict(np.array(st).reshape(1,-1))[0])
    def replay(s):
        if len(s.m)<32: return
        for st,a,r,ns,d in random.sample(s.m,32):
            st=np.array(st).reshape(1,-1)
            ns=np.array(ns).reshape(1,-1)
            t=r if d else r+s.g*np.amax(s.t.predict(ns)[0])
            tf=s.q.predict(st)
            tf[0][a]=t
            s.q.train(st,tf)

#=============== MARKET ANALYZER ===============
class M:
    def __init__(s):
        s.ph=c.defaultdict(lambda:c.deque(maxlen=100))
        s.vh=c.defaultdict(lambda:c.deque(maxlen=100))
    def add(s,a,p,v): s.ph[a].append(p); s.vh[a].append(v)
    def rsi(s,p,per=14):
        if len(p)<per+1: return 50
        d=np.diff(p[-per-1:])
        g=d[d>0].sum()/per
        l=-d[d<0].sum()/per
        return 100-100/(1+g/l) if l else 100
    def macd(s,p):
        if len(p)<26: return 0,0,0
        ema=lambda x,per: np.mean(x[-per:])
        e12=ema(p,12)
        e26=ema(p,26)
        macd_val=e12-e26
        signal=ema([macd_val]*9,9)
        hist=macd_val-signal
        return macd_val, signal, hist
    def bb(s,p,per=20):
        if len(p)<per: return p[-1],p[-1]*1.1,p[-1]*0.9
        sm=np.mean(p[-per:])
        sd=np.std(p[-per:])
        return sm, sm+2*sd, sm-2*sd
    def regime(s,p):
        if len(p)<20: return 'SIDE'
        r=np.diff(np.log(p))
        v=np.std(r)*np.sqrt(24*365)
        t=(p[-1]-p[0])/p[0]
        if v>0.5: return 'VOL'
        return 'BULL' if t>0.1 else 'BEAR' if t<-0.1 else 'SIDE'

#=============== RISK MANAGER ===============
class R:
    def __init__(s,ic):
        s.c=s.pk=ic
        s.dd=s.cl=s.cw=0
        s.daily=[]
        s.res=[]
        s.wr=s.aw=s.al=s.pf=s.sr=0
    def pos(s,sc,vol,reg):
        base=s.c*0.1
        sf=min(sc/100,1)
        vf=max(0.3,1-vol)
        rf={'BULL':1.2,'SIDE':1,'BEAR':0.6,'VOL':0.4}.get(reg,1)
        df=max(0.2,1-s.dd/0.2)
        lf=0.5**s.cl
        wf=1+s.cw*0.1
        tf=0.5 if datetime.now().hour in[0,1,2,3,4,5,23] else 1
        kf=0.1
        p=base*sf*vf*rf*df*lf*wf*tf*(1+max(0,kf))
        p=min(max(p,C.MIN_POSITION),s.c*0.25)
        if sum(s.daily)<-s.c*0.05: p*=0.1
        return p
    def update(s,pnl,pnlp):
        s.res.append(pnlp)
        s.daily.append(pnl)
        s.daily=s.daily[-20:]
        if pnl>0: s.cw+=1; s.cl=0
        else: s.cl+=1; s.cw=0
        s.c+=pnl
        s.pk=max(s.pk,s.c)
        s.dd=(s.pk-s.c)/s.pk
        if len(s.res)>=5:
            w=[r for r in s.res if r>0]
            l=[r for r in s.res if r<0]
            s.wr=len(w)/len(s.res)*100
            s.aw=np.mean(w) if w else 0
            s.al=abs(np.mean(l)) if l else 0
            if s.al: s.pf=(len(w)*s.aw)/(len(l)*s.al) if l else 99
            if len(s.res)>1:
                ret=np.array(s.res)/100
                s.sr=np.mean(ret)/(np.std(ret)+1e-10)*np.sqrt(365)
    def stop(s): return s.dd>0.2 or sum(s.daily)<-s.c*0.05

#=============== SIGNAL GENERATOR ===============
class S:
    def __init__(s,a): s.a=a; s.w={'l':1,'v':1,'m':1.2,'b':1.3,'w':1.5,'p':1.4,'r':0.8,'d':1.2,'s':1.1}
    def gen(s,p,addr):
        r=[]; sc=0
        try:
            l=float(p.get("liquidity",{}).get("usd",0))
            v=float(p.get("volume",{}).get("h24",0))
            pr=float(p.get("priceUsd",0))
            m5=float(p.get("priceChange",{}).get("m5",0))
            m15=float(p.get("priceChange",{}).get("m15",0))
            txns=p.get("txns",{}).get("m5",{})
            bm5=txns.get("buys",0)
            sm5=txns.get("sells",0)
            # Liquidity
            ls=min(l/20000*15,15); 
            if ls>10: sc+=ls*s.w['l']; r.append(f"💰 {ls:.0f}")
            # Volume
            vs=min(v/50000*15,15)
            if vs>10: sc+=vs*s.w['v']; r.append(f"📊 {vs:.0f}")
            # Momentum
            ms=min(m5*0.5+m15*0.3,20)
            if ms>5: sc+=ms*s.w['m']; r.append(f"📈 {ms:.0f}")
            # Buy pressure
            if bm5+sm5>0:
                bs=(bm5/(bm5+sm5))*15
                if bs>10: sc+=bs*s.w['b']; r.append(f"🟢 {bs:.0f}")
            # Whale/Pump
            if bm5>30 and bm5>3*sm5: sc+=20*s.w['w']; r.append("🐋 20")
            if float(p.get("volume",{}).get("m5",0))>float(p.get("volume",{}).get("h1",1))/12*5 and m5>3: sc+=20*s.w['p']; r.append("🚀 20")
            # Technical indicators
            prs=list(s.a.ph[addr])
            if len(prs)>14:
                rsi=s.a.rsi(prs)
                if rsi<30: sc+=15*s.w['r']; r.append(f"📉 {rsi:.0f}")
            vols=list(s.a.vh[addr])
            if vols and len(prs)>20:
                bb=s.a.bb(prs)
                if pr<bb[2]: sc+=15; r.append("📉 BB")
            # Risk
            risk=0
            if l<C.MIN_LIQUIDITY*1.5: risk+=20
            if v<C.MIN_VOLUME*2: risk+=15
            sc*=(1-risk/200)
            # Signal type
            sig='H'; conf=0
            if sc>80 and risk<50: sig='SB'; conf=0.9
            elif sc>65 and risk<60: sig='B'; conf=0.7
            elif sc>50: sig='W'; conf=0.5
            return sig, sc, conf, r[:3]
        except Exception as e: 
            print("Signal gen error:", e)
            return 'E',0,0,['ERR']

#=============== TELEGRAM =================
class T:
    def __init__(s):
        s.s=requests.Session()
        s.q=queue.Queue()
        s.l=0
        s.s.headers.update({"User-Agent":C.USER_AGENT})
    def send(s,msg):
        if not C.TG_TOKEN: print(msg); return
        try:
            s.s.post(f"https://api.telegram.org/bot{C.TG_TOKEN}/sendMessage",
                        json={"chat_id":C.CHAT_ID,"text":msg,"parse_mode":"HTML"})
        except: pass
    def buy(s,sym,sc,rs,sz,pr):
        text=f"""
🦈 <b>SIGNAL — فرصت خرید</b>

<b>توکن:</b> {sym}
<b>امتیاز:</b> {int(sc)}/100
<b>قیمت:</b> ${pr:.8f}
<b>حجم ورود:</b> {sz:.3f} TON

<b>دلایل:</b>
{chr(10).join(['• '+r for r in rs])}
"""
        s.send(text)
    def sell(s,sym,pnlp,pnl,reas,pr):
        emoji="💰" if pnl>0 else "😢"
        text=f"""
{emoji} <b>EXIT — بستن معامله</b>

<b>توکن:</b> {sym}
<b>نتیجه:</b> {pnlp:+.2f}%
<b>سود/ضرر:</b> {pnl:.4f} TON
<b>قیمت خروج:</b> ${pr:.8f}
<b>دلیل:</b> {reas}
"""
        s.send(text)
    def daily(s,st):
        total=st['w']+st['l']
        winrate=(st['w']/total*100) if total>0 else 0
        text=f"""
📊 <b>گزارش ربات</b>

<b>سرمایه:</b> {st['c']:.3f} TON
<b>سود کل:</b> {st['d']:+.3f} TON
<b>برد:</b> {st['w']}
<b>باخت:</b> {st['l']}
<b>درصد موفقیت:</b> {winrate:.0f}%
<b>پوزیشن باز:</b> {st['op']}
"""
        s.send(text)

#=============== TRADING ENGINE ===============
class E:
    def __init__(s):
        s.a=M(); s.sg=S(s.a); s.rm=R(C.INITIAL_CAPITAL); s.t=T()
        s.pos={}; s.bl=set(); s.st={'t':0,'w':0,'l':0,'tp':0,'d':0,'dt':0}
        s.run=False; s.last=datetime.now().date()
    def proc(s,p):
        try:
            a=p.get('pairAddress')
            if a in s.bl or not a: return
            s.a.add(a,float(p.get('priceUsd',0)),float(p.get("volume",{}).get("h24",0)))
            sig,sc,conf,rs=s.sg.gen(p,a)
            if sig in['SB','B']: return {'a':a,'sym':p.get('baseToken',{}).get('symbol','?'),'p':float(p.get('priceUsd',0)),'sc':sc,'rs':rs}
        except Exception as e:
            print("Proc error:", e)
            return
    def entry(s,o):
        if len(s.pos)>=C.MAX_POSITIONS or o['a'] in s.pos: return
        sz=s.rm.pos(o['sc'],0.3,s.a.regime(list(s.a.ph[o['a']])))
        if sz*o['p']>s.rm.c: return
        sl=o['p']*(1-max(C.BASE_STOP_LOSS*0.5,min(C.BASE_STOP_LOSS*(1-o['sc']/200)*1.3,C.MAX_STOP_LOSS)))
        tp=o['p']*(1+min(C.BASE_TAKE_PROFIT*(1+o['sc']/100)*1.15,C.MAX_TAKE_PROFIT))
        s.pos[o['a']]={'sym':o['sym'],'ep':o['p'],'am':sz,'sl':sl,'tp':tp,'pk':o['p'],'sc':o['sc'],'rs':o['rs']}
        s.rm.c-=sz*o['p']
        s.t.buy(o['sym'],o['sc'],o['rs'],sz,o['p'])
    def check(s,a,pr):
        if a not in s.pos: return
        p=s.pos[a]
        if pr>p['pk']: p['pk']=pr
        if pr<=p['sl']: return ('SL',1)
        if pr>=p['tp']: return ('TP',1)
        if pr<=p['pk']*(1-C.TRAILING_STOP): return ('TR',1)

    def exit(s,a,pr,reas,per):
        if a not in s.pos: return
        p = s.pos[a]
        sell = p['am'] if per>=1 else p['am']*per
        if per>=1: del s.pos[a]
        else: p['am'] -= sell
        pnl = (pr - p['ep']) * sell
        pnlp = (pr / p['ep'] - 1) * 100
        s.rm.c += sell * pr
        s.rm.update(pnl, pnlp)
        s.st['t'] += 1
        if pnl>0: s.st['w'] += 1
        else: s.st['l'] += 1
        s.st['tp'] += pnl
        s.st['d'] += pnl
        s.st['dt'] += 1
        s.t.sell(p['sym'], pnlp, pnl, reas, pr)

    def run(s):
        print("🦈 SHARK X ACTIVE")
        s.run = True
        last_daily = datetime.now()
        while s.run:
            try:
                # Stop condition
                if s.rm.stop(): break
                # Fetch market data
                resp = requests.get(C.DEX_API, headers={"User-Agent":C.USER_AGENT}, timeout=C.API_TIMEOUT)
                pairs = resp.json().get('pairs', [])
                if not pairs: time.sleep(C.SCAN_INTERVAL); continue
                # Process opportunities
                opps = [s.proc(p) for p in pairs[:50] if s.proc(p)]
                opps.sort(key=lambda x: x['sc'], reverse=True)
                [s.entry(o) for o in opps[:C.MAX_POSITIONS]]
                # Check exits
                for p in pairs:
                    for a in list(s.pos.keys()):
                        try:
                            chk = s.check(a, float(p.get('priceUsd',0)))
                            if chk: s.exit(a, float(p.get('priceUsd',0)), *chk)
                        except Exception as e:
                            print("Check error:", e)
                # Save state
                F.save(C.STATE_FILE, {'c':s.rm.c,'pos':s.pos,'bl':list(s.bl)})
                # Daily report every hour
                if (datetime.now() - last_daily).seconds > 3600:
                    s.t.daily({
                        'c': s.rm.c,
                        'd': s.st['d'],
                        'w': s.st['w'],
                        'l': s.st['l'],
                        'wr': (s.st['w']/s.st['t']*100) if s.st['t'] else 0,
                        'op': len(s.pos)
                    })
                    last_daily = datetime.now()
                time.sleep(C.SCAN_INTERVAL)
            except Exception as e:
                print("Runtime error:", e)
                time.sleep(60)

#=============== MAIN ===============
if __name__=="__main__":
    if os.path.exists(C.LOCK_FILE):
        print("⚠️ Already running")
        exit()
    open(C.LOCK_FILE,'w').close()
    try:
        E().run()
    finally:
        if os.path.exists(C.LOCK_FILE): os.remove(C.LOCK_FILE)
