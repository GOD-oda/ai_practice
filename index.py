
import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
from sklearn.externals.six import StringIO
from sklearn import metrics

with open('./titanic.csv', 'r') as f:
    titanic_reader = csv.reader(f, quotechar='"')

    # 特徴量の名前が書かれたHeaderを読み取る
    row = next(titanic_reader)
    feature_names = np.array(row)

    # データと正解ラベルを読み取る
    titanic_x, titanic_y = [], []
    for row in titanic_reader:
        titanic_x.append(row)
        titanic_y.append(row[2])

    titanic_x = np.array(titanic_x)
    titanic_y = np.array(titanic_y)

# print(feature_names)
# print(titanic_x[0], titanic_y[0])

index = [1,4,10]
titanic_x = titanic_x[:, index]
feature_names = feature_names[index]

# print(titanic_x[12], titanic_y[12])
# print(feature_names)

# 年齢の欠損値を平均値で埋める
ages = titanic_x[:, 1]
# NA以外のageの平均値を計算する
mean_age = np.mean(titanic_x[ages != 'NA', 1].astype(float))
# ageがNAのものを平均値に置き換える
titanic_x[titanic_x[:, 1] == 'NA', 1] = mean_age

enc = LabelEncoder()
label_encoder = enc.fit(titanic_x[:, 2])
# print('Categorical classes:', label_encoder.classes_)

integer_classes = label_encoder.transform(label_encoder.classes_)
# print('Integer classes:', integer_classes)

t = label_encoder.transform(titanic_x[:, 2])
titanic_x[:, 2] = t
# print(feature_names)
# print(titanic_x[12], titanic_y[12])

label_encoder = enc.fit(titanic_x[:, 0])
# print("Categorical classes:", label_encoder.classes_)

integer_classes = label_encoder.transform(label_encoder.classes_).reshape(3, 1)
# print('Integer classes:', integer_classes)

enc = OneHotEncoder()
one_hot_encoder = enc.fit(integer_classes)

# 最初に、Label Encoderを使ってpclassを0-2に直す
num_of_rows = titanic_x.shape[0]
t = label_encoder.transform(titanic_x[:, 0]).reshape(num_of_rows, 1)
# 次に、OneHotEncoderを使ってデータを1, 0に変換
new_features = one_hot_encoder.transform(t)
# 1,0になおしてデータを統合する
titanic_x = np.concatenate([titanic_x, new_features.toarray()], axis=1)
# OnehotEncoderをする前のpclassのデータを削除する
titanic_x = np.delete(titanic_x, [0], 1)
# 特徴量の名前を更新する
feature_names = ['age', 'sex', 'first class', 'second class', 'third class']

titanic_x = titanic_x.astype(float)
titanic_y = titanic_y.astype(float)

# print(feature_names)
# print(titanic_x[0], titanic_y[0])

train_x, test_x, train_y, test_y = train_test_split(titanic_x, titanic_y,
                                                    test_size=0.25, random_state=33)

# 次にscikit-learnのDecision Tree Classifierを使って学習します。
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
clf = clf.fit(train_x, train_y)

# TODO: pdfに出力されない
# dot_data = StringIO()
# tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_names)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("tree.pdf")

def measure_performance(x, y, clf, show_accuracy=True,
                        show_classification_report=True, show_confusion_matrix=True):
    y_pred = clf.predict(x)

    if show_accuracy:
        print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)), "\n")

    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y, y_pred), "\n")

    if show_confusion_matrix:
        print("Confussion matrix")
        print(metrics.confusion_matrix(y, y_pred), "\n")


measure_performance(train_x, train_y, clf)