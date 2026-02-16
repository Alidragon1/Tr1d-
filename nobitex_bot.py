import os, requests, time, hmac, hashlib, json
from datetime import datetime

# ===========================
# تنظیمات اولیه
# ===========================
TOTAL_CAPITAL = float(os.getenv("TOTAL_CAPITAL", 5_000_000))  # ۵۰۰ هزار تومان
MAX_ACTIVE = 3
STOP_LOSS = 0.08
TRAIL_TRIGGER = 1.4
TRAIL_KEEP = 0.75
CHECK_INTERVAL = 60  # GitHub Actions friendly
SYMBOLS = ["BTCIRT", "ETHIRT", "XRPIRT"]  # چند جفت ارز همزمان
POSITIONS = {}

# ===========================
# API Keys نوبیتکس
# ===========================
API_KEY = os.getenv("NOBITEX_API_KEY")
API_SECRET = os.getenv("NOBITEX_API_SECRET")

# ===========================
# کارمزد معاملات
# ===========================
MAKER_FEE = 0.003
TAKER_FEE = 0.005

# ===========================
# توابع کمکی API
# ===========================
BASE_URL = "https://api.nobitex.ir/v2"

def make_signature(path, body):
    msg = f"{path}{json.dumps(body)}"
    return hmac.new(API_SECRET.encode(), msg.encode(), hashlib.sha512).hexdigest()

def call_private_api(path, body):
    headers = {"APIKEY": API_KEY, "SIGNATURE": make_signature(path, body)}
    r = requests.post(BASE_URL + path, json=body, headers=headers, timeout=10)
    return r.json()

def get_orderbook(symbol):
    try:
        r = requests.get(f"{BASE_URL}/orderbook/{symbol}", timeout=5)
        return r.json()
    except:
        return None

# ===========================
# تحلیل حرفه‌ای بازار
# ===========================
def analyze_market(symbol):
    ob = get_orderbook(symbol)
    if not ob: return False
    bids = [float(x[0]) for x in ob["bids"][:20]]
    asks = [float(x[0]) for x in ob["asks"][:20]]
    momentum = (bids[0]-asks[0])/asks[0]
    vol_change = sum([float(x[1]) for x in ob["bids"][:5]]) / (sum([float(x[1]) for x in ob["asks"][:5]])+0.0001) -1
    trend = sum(bids)/len(bids) - sum(asks)/len(asks)
    atr = (max(bids) - min(asks)) / (sum(asks)/len(asks))
    return momentum>0.002 and vol_change>0.05 and trend>0 and atr>0.01

# ===========================
# مدیریت پوزیشن
# ===========================
def buy(symbol, amount):
    ob = get_orderbook(symbol)
    price = float(ob["bids"][0][0]) if ob else 1
    fee = amount * TAKER_FEE
    invest_amount = amount - fee
    POSITIONS[symbol] = {"buy": price, "high": price, "amount": invest_amount}
    global TOTAL_CAPITAL
    TOTAL_CAPITAL -= amount
    print(f"{datetime.now()} | BUY {symbol} | amount: {invest_amount:.0f} | price: {price:.2f} | Fee: {fee:.0f}")

def sell(symbol, reason):
    if symbol not in POSITIONS: return
    ob = get_orderbook(symbol)
    price = float(ob["asks"][0][0]) if ob else POSITIONS[symbol]["buy"]
    pos = POSITIONS[symbol]
    gross_profit = (price - pos["buy"]) * pos["amount"]/pos["buy"]
    fee = (pos["amount"] + gross_profit) * TAKER_FEE
    net_profit = gross_profit - fee
    global TOTAL_CAPITAL
    TOTAL_CAPITAL += pos["amount"] + net_profit
    print(f"{datetime.now()} | SELL {symbol} | Reason: {reason} | Profit: {net_profit:.0f} | Fee: {fee:.0f}")
    POSITIONS.pop(symbol)

def manage_positions():
    global TOTAL_CAPITAL
    for symbol, pos in list(POSITIONS.items()):
        ob = get_orderbook(symbol)
        price = float(ob["bids"][0][0]) if ob else pos["buy"]
        pos["high"] = max(pos["high"], price)
        if price <= pos["buy"]*(1-STOP_LOSS):
            sell(symbol, "STOP LOSS")
        elif pos["high"] >= pos["buy"]*TRAIL_TRIGGER:
            if price <= pos["high"]*TRAIL_KEEP:
                sell(symbol, "TRAIL EXIT")

# ===========================
# حلقه اصلی
# ===========================
print("🚀 NOBITEX Ultra-Whale Smart Bot Started with 500k Toman")

while True:
    try:
        for symbol in SYMBOLS:
            if len(POSITIONS) >= MAX_ACTIVE: break
            if analyze_market(symbol):
                invest = TOTAL_CAPITAL / MAX_ACTIVE
                buy(symbol, invest)
        manage_positions()
        print(f"{datetime.now()} | Current Capital: {TOTAL_CAPITAL:.0f} ریال | Active Positions: {len(POSITIONS)} | Open Pairs: {list(POSITIONS.keys())}")
    except Exception as e:
        print("Error:", e)
    time.sleep(CHECK_INTERVAL)
