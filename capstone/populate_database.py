__author__ = 'harsshal'
import sqlite3
import pandas as pd
import datetime
import data

def main():
    conn = sqlite3.connect('alpha.db')
    c = conn.cursor()

    c.execute("drop table if exists secmaster")
    c.execute("drop table if exists close")

    c.execute('''
        create table if not exists secmaster (
            id int auto_increment,
            Symbol char(7),
            Name char(30),
            Sector char (15)
        );
    ''')

    c.execute("delete from secmaster")
    sp500 = pd.read_csv('sp500_constituents_mar2017_okfn.csv', header=2, nrows=5)
    columns = ""
    for i in range(len(sp500)):
    #for i in range(30):
        c.execute("insert into secmaster "
              "(id,symbol,name,sector) "
              "values (%d,\"%s\",\"%s\",\"%s\")"%(i,sp500['Symbol'][i],sp500['Name'][i],sp500['Sector'][i]))

    #rics = sp500.apply(lambda x: ' float ,'.join(x.astype(str).values))['Symbol']
    #rics += ' float'
    rics = ",".join([str(x)+" float" for x in sp500['Symbol'].values])
    query = "create table if not exists close (date date"+rics+");"
    c.execute(query)

    for year in range(2015,2016,1):
        for month in range(1,12):
            start = datetime.date(year,month,1)
            if month != 12:
                end = datetime.date (year, month+1, 1) - datetime.timedelta (days = 1)
            else:
                end = datetime.date (year+1, 1, 1) - datetime.timedelta (days = 1)
            df = data.get_yahoo_data(sp500['Symbol'].values,start,end)
            print(df)
    # c.execute("insert into secmaster (ric,name,sector) values('AGN', 'qweadfdsa', 'home')")
    # c.execute("insert into stocks values('MSFT',100, 60)")
    #
    symbol = ('MSFT',)
    #
    # c.execute('select * from stocks where symbol = ?',symbol)
    # print(c.fetchone())
    #
    # positions = [('IBM',200,30),('AAPL',100,40)]
    # c.executemany('insert into stocks values (?,?,?)',positions)
    #
    # for row in c.execute('select * from stocks order by price'):
    #     print(row)

    conn.commit()

    conn.close()


if __name__ == '__main__':
    main()