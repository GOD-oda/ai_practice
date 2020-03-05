from janome.tokenizer import Tokenizer
# from gensim.models import word2ved
import re

# binary_data = open('./sample.txt.sjis.txt').read()
# text = binary_data.decode('shift_jis')
# print(text)

import gensim
from gensim import corpora
from pprint import pprint
from collections import defaultdict

docs = []
documents = [
    "JR京葉線／武蔵野線・舞浜駅より徒歩8分。東京ディズニーランド正面に位置したディズニーランド観光の拠点として最適なロケーションのホテルです。イギリス・ヴィクトリア朝様式で統一した壮大で豪華な外装はシンデレラ城と雰囲気が調和しているため、ディズニーランドで見た夢の余韻そのままに過ごすことができます。30mの高さまで吹き抜けになった開放的なロビーは、その天井から巨大な2つのシャンデリアが下がり、まるで舞踏会に来たような気分に浸れます。客室は部屋の様々な場所にディズニーモチーフを散りばめた内装になっています。スタンダード、キャラクタールーム、コンシェルジュ、スイートの4タイプ。テレビ、電話、インターネット接続（LAN形式）、バストイレ、シャワー室、湯沸かしポット、冷蔵庫、ミニバー、ドライヤー、金庫、バスアメニティを完備。アイロン、ズボンプレッサー、加湿器、変圧器の貸出しあり。",
    "東京ディズニーセレブレーションホテルは、リゾート感あふれるディズニーホテルです。舞浜駅で下車し、徒歩5分の東京ディズニーランド・バス、タクシーターミナルより無料シャトルバスにて約20分で到着します。ホテルの庭や館内は、ディズニーキャラクターでいっぱいなので、ホテル内でもディズニーリゾートを楽しむことができ、ファンにおすすめです。東京ディズニーセレブレーションホテルの施設設備は、24時間対応のフロントデスク、荷物預り所、喫煙所、駐車場、コインランドリーに加え、レストラン、コンビニエンスストアなどの娯楽施設も充実しています。サービスとして、無料Wi-Fi、宅配サービス、無料シャトルバスを利用できるので、快適に滞在できます。東京ディズニーセレブレーションホテルの客室は、カラフルなデザインでディズニーキャラクターが描かれています。個別空調、薄型テレビ、クローゼット、加湿器、金庫、冷蔵庫、ティーセット、シーティングエリアに加え、歯ブラシや櫛、髭剃りなど個別包装の無料バスアメニティ、ヘアドライヤー、スリッパ、バスローブ、洗浄機能付きトイレ、専用バスルームが備え付けられています。朝食は、バイキング形式で楽しめます。"
]
# ストップワードの設定
stop_words = set('だ'.split())

texts = [[word for word in document.lower().split() if word not in stop_words] for document in documents]
pprint(texts)

exit()

# TODO: ここからおかしい

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]
# pprint(texts)

dictionary = corpora.Dictionary(texts)
dictionary.save('./sample.dict')
dictionary.save_as_text('./sample.dict.txt')

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('./sample.mm', corpus)

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=3, id2word=dictionary)
print(lda.show_topics())




