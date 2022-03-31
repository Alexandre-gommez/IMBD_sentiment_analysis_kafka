from kafka import KafkaProducer, KafkaConsumer
import json
import time

consumer = KafkaConsumer('imbd-ingest', bootstrap_servers='localhost:9092')
producer = KafkaProducer(bootstrap_servers='localhost:9092')


while True:
    print("Waiting for data")
    for data in consumer:
        movies = {}
        response = json.loads(data.value.decode('utf-8'))
        list_movies  = response['items']
        for movie in list_movies:
            producer.send('imbd-selection', json.dumps({"Id":movie['id'],
                                                        "Title": movie['title']}).encode('utf-8'))