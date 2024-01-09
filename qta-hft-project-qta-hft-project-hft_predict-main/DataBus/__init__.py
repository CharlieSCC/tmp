# ---*** DataProcessing ***---
def unifyz(df):
    """统一字段"""
    import re
    cols_mapper = {
        'open': 'o',
        'high': 'h',
        'low': 'l',
        'close': 'c',
        'last': 'c',
        'last_price': 'c',
        'volume': 'v',
        'amount': 'a',
        'turnover': 'a',
        'trades': 'n',
        'trade_count': 'n',
        'ask_price': 'ap',
        'bid_price': 'bp',
        'ask_volume': 'av',
        'bid_volume': 'bv',
        'long_volume': 'lv',
        'short_volume': 'sv',
        'nv': 'V',
        'net_volume': 'V',
        'net_amount': 'A'
    }
    df.index.name = 't'
    df.columns = df.columns.map(lambda i: cols_mapper.get(j := re.sub(r'\d', '', i), j) + re.sub(r'\D', '', i))
    return df
