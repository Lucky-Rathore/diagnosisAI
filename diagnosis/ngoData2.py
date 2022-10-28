import requests
import threading
import json
import psycopg2

conn = psycopg2.connect("change me")
cur = conn.cursor()
# def storeInDB(jsonText):
#     with conn.cursor() as cur:
#         cur.execute("INSERT INTO ngodata(data) VALUES (%s)", (jsonText  ,))
#     conn.commit()
#     conn.close()







idsTotal = [int(i) for i in idsTotal]

print("total ids: " + str(len(idsTotal)))

cur.execute('select distinct  id  from ngodata n ')
rows = cur.fetchall()



doneIDs = [ i[0] for i in rows if i != None and i[0] != None]

print('total doneIDs: ' + str(len(doneIDs)))

ids =  list(set(idsTotal) - set(doneIDs))
n = len(ids)
print('total ids: ' + str(len(ids)))

threadCount = 50

threads = []

cookies = {
    'ci_session': 'a40u581upfedc752tej11dn63a2ud8mj',
    'csrf_cookie_name': 'b6fb02fa9200898df6508207fabe9ff7',
}

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:105.0) Gecko/20100101 Firefox/105.0',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.5',
    # 'Accept-Encoding': 'gzip, deflate, br',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'X-Requested-With': 'XMLHttpRequest',
    'Origin': 'https://ngodarpan.gov.in',
    'Connection': 'keep-alive',
    'Referer': 'https://ngodarpan.gov.in/index.php/home/statewise_ngo/174/35/1?per_page=100',
    # Requests sorts cookies= alphabetically
    # 'Cookie': 'ci_session=a40u581upfedc752tej11dn63a2ud8mj; csrf_cookie_name=b6fb02fa9200898df6508207fabe9ff7',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-GPC': '1',
}

def get(i,j):
    for k in range(i, j):  
        totalCount = 1
        print(str(ids[k]) + 'processing')
        data = { 'id': ids[k], 'csrf_test_name': 'b6fb02fa9200898df6508207fabe9ff7' }
        r = requests.post('https://ngodarpan.gov.in/index.php/ajaxcontroller/show_ngo_info', cookies=cookies, headers=headers, data=data)
        print(r)
        # storeInDB()
        cur.execute("INSERT INTO ngodata(data) VALUES (%s)", (r.text  ,))
        if totalCount % 1000 == 0:
            print('********* commit 1k *********')
            conn.commit()
        print(str(( j - i) ) + " k= " + str(k) + " response: " + str(r))
        doneIDs.append(ids[k])
        totalCount = totalCount + 1

print("threads append!")

j = 0
for i in range(50):
    threads.append( threading.Thread(target=get, args=(j, j + int(n / 50))) )
    j = j + int(n / 50)

print("start!")

for t in threads:
    t.start()

print("join!")

for t in threads:
    t.join()

conn.close()
print("Done!") 




