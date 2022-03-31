from kafka import KafkaProducer, KafkaConsumer
import json
import time
import requests

Alex = "k_kn65a19i"
Narjes = "k_77to52it"
consumer = KafkaConsumer('imbd-selection', bootstrap_servers='localhost:9092')
producer = KafkaProducer(bootstrap_servers='localhost:9092')

while True:
    for movie in consumer:
        movie = json.loads(movie.value.decode('utf-8'))
        url = "https://imdb-api.com/en/API/Reviews/"+Narjes+"/" + movie['Id']
        requested_data = requests.get(url)
        data = json.loads(requested_data.content)
        for review in data['items']:
            review['title'] = movie['Title']
            producer.send('imbd-review', json.dumps(review).encode('utf-8'))
            time.sleep(1)
    time.sleep(5)