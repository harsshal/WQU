__author__ = 'harsshal'
'''
This module uses user built data module
to stored OHLC data into user database
'''

import sqlite3
import pandas as pd
import numpy as np
import datetime
import data

def main():
    '''
    main function in the module to get
    data from yahoo into personal database
    :return: nothing
    '''

    with sqlite3.connect('alpha.db') as conn:
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

        sp500 = pd.read_csv('sp500_constituents_mar2017_okfn.csv', header=2, nrows=50)
        for i in range(len(sp500)):
            c.execute("insert into secmaster "
                  "(id,symbol,name,sector) "
                  "values (%d,\"%s\",\"%s\",\"%s\")"%(i,sp500['Symbol'][i],sp500['Name'][i],sp500['Sector'][i]))

        rics = ",".join(["'"+str(x)+"' float" for x in np.sort(sp500['Symbol'].values)])
        query = "create table if not exists close (date date, "+rics+");"
        c.execute(query)

        for year in range(2015,2016,1):
            for month in range(1,13):
                start = datetime.date(year,month,1)
                if month != 12:
                    end = datetime.date (year, month+1, 1) - datetime.timedelta (days = 1)
                else:
                    end = datetime.date (year+1, 1, 1) - datetime.timedelta (days = 1)
                df = data.get_yahoo_data(np.sort(sp500['Symbol'].values),start,end)

                df.to_sql('close', conn, if_exists='append',index=True)


if __name__ == '__main__':
    main()
