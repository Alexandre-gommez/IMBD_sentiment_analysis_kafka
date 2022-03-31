import requests
from kafka import KafkaProducer
import json
import time

Alex = "k_kn65a19i"
Narjes = "k_77to52it"
url = "https://imdb-api.com/en/API/BoxOffice/"+Narjes

producer = KafkaProducer(bootstrap_servers='localhost:9092')

while True:
    requested_data = requests.get(url)
    data = json.loads(requested_data.content)
    print(data['errorMessage'])
    if data['errorMessage'] != '':
        print("Too much query for the day")
        break 
    producer.send('imbd-ingest', json.dumps(data).encode('utf-8'))
    print("Sent data to Kafka")
    time.sleep(604800)
    