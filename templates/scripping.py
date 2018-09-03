from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import csv
from bs4 import BeautifulSoup
import re
from flask import Flask
from flask import Flask, abort, request, jsonify
import pandas as pd


# app = Flask(__name__)

class scripping():
    # @app.route('/add_time/', methods=['POST'])
    def main(date):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver', chrome_options=chrome_options)
        # times = request.json['times']
        times=date
        # a='2015/3/'
        driver.get("http://www.worldfootball.net/teams/manchester-united/"+times+"/3/")
        html=driver.page_source
        soup = BeautifulSoup(html, 'lxml')#creat soup

        matches = soup.find_all(name='a',attrs={"href": re.compile(r'.\w{13}.\d{4}.\w{3}.\d.')})#match date
        matches1 = soup.find_all(name='a',attrs={"href": re.compile(r'.(teams).\w*(-)\w*(-)?(\w*)(/)'+times+'(/3/)')})#match team
        matches4 = soup.find_all('td')#match place
        #match palce
        tag = []
        for a in matches4:
            tag1 = re.findall(r'(<td)\s(align="center")\s(class="hell">|class="dunkel">)(H|N|A)(</td>)', str(a))
            if len(tag1):
                tag.append(tag1)

        tag2 = []
        tag11 = re.sub(r'(\'|,)', '', str(tag))
        soup2 = BeautifulSoup(tag11, 'lxml')
        matches5 = soup2.find_all('td')
        for f in matches5:
            if len(f):
                tag2.append(f.string)

        team=[]
        date=[]
        #match date
        for i in matches:
            date.append(i.string)
        #match team
        for k in matches1:
            team.append(k.string)
        for j in range(10000):
        #"/Users/ushisensei/Desktop/matchesdata/"
            csvFile = open(times+"footballmatch.csv", "w+")
            del(team[0])
            try:
                writer = csv.writer(csvFile)
                writer.writerow(('date','place', 'team'))
                for j in range(len(team)):

                    writer.writerow((date[j],tag2[j],team[j]))



            finally:
                csvFile.close()
            break




        driver.close()
        rmatches = str(pd.read_csv(times+'footballmatch.csv'))
        return rmatches




    # if __name__ == '__main__':
    #
    #    app.run(host="0.0.0.0", port=8379, debug=True)
    main()

