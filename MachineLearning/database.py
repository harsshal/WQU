__author__ = 'harsshal'

from urllib.request import urlopen
from bs4 import BeautifulSoup
import datetime
from math import ceil
from lxml import etree
import sqlite3

def parse_wiki(url,path):
    html_parser = etree.HTMLParser()
    page = urlopen(url)
    tree = etree.parse(page,html_parser)
    symbol_list = tree.xpath(path)[1:]
    symbols = []

    for symbol in symbol_list:
        tds = symbol.getchildren()
        sd = {
            'ticker' : tds[0].getchildren()[0].text,
        }
        symbols.append(sd['ticker'])

    return symbols

def populate_database(symbols):
    conn = sqlite3.connect('example.db')
    c = conn.cursor()

    c.execute('''
    create table if not exists stocks (symbol text, qty real, price real)
    ''')

    c.execute("delete from stocks")

    c.execute("insert into stocks values('AGN',100, 200)")
    c.execute("insert into stocks values('MSFT',100, 60)")

    symbol = ('MSFT',)

    c.execute('select * from stocks where symbol = ?',symbol)
    print(c.fetchone())

    positions = [('IBM',200,30),('AAPL',100,40)]
    c.executemany('insert into stocks values (?,?,?)',positions)

    for row in c.execute('select * from stocks order by price'):
        print(row)

    conn.commit()

    conn.close()

def main():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    path = "//table[1]/tr"
    symbols = parse_wiki(url,path)
    populate_database(symbols)

if __name__ == '__main__':
    main()