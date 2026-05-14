from app.tools import get_info_ticker


def test_get_info_ticker_return_str():
    assert type(get_info_ticker("BTC-USD")) == str 

