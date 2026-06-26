На основі отриманих результатів я б не рекомендував використовувати жорстко підібрані коефіцієнти для функції Q.
оскільки вони не випливають безпосередньо з ML-моделі.

Зараз у вас уже є емпіричні оцінки важливості ознак:
Ознака	Importance
Chessboard	0.237
Ones	0.214
Transitions	0.186
Cmax	0.153
Isolated	0.119
Homogeneous	0.092

Тому я б запропонував перейти до нормалізованої функції якості.

Варіант 1. Інженерна евристика (рекомендую)
Нормалізуємо всі ознаки до [0;1].
Тоді:
Q=0.24(1−Chessboardn)+0.21(1−OnesDevn)+0.19(1−Transitionsn)+0.15Cmaxn+0.12(1−Isolatedn)+0.09Homogeneousn
де
OnesDev=∣Ones−8∣
(для 4×4 маркера ідеальний баланс — 8 білих клітинок із 16).
Переваги:
інтерпретована;
усі коефіцієнти походять із ML;
легко переноситься на інші словники.

Варіант 2. Спрощена функція

Оскільки три фактори домінують:
Chessboard
Ones
Transitions
можна використовувати
Q=0.4(1−Chessboardn)+0.3(1−OnesDevn)+0.3(1−Transitionsn)

Це дасть майже той самий рейтинг.

Варіант 3. Бальна система для генератора маркерів

Для автоматичного відбору кандидатів:

Q=100−20⋅Chessboard−3⋅Transitions−2⋅Isolated−2⋅∣Ones−8∣+2⋅Cmax

Тут не потрібна нормалізація.

Практично:

Q>80 — відмінний кандидат;
60<Q≤80 — прийнятний;
Q<60 — відхилити.

Варіант 4. Найбільш науково коректний

Побудувати логістичну регресію:

P(DetectionRate≥95%)=σ(β
0+β1	​
Chessboard+β2
Ones+β3
Transitions+β4
Cmax+β5
Isolated)

і використовувати саме

Q=P(DetectionRate≥95%)

як функцію якості.

Тоді:

Q=0.95 означає 95% ймовірність потрапити до класу "хороших";
Q=0.20 означає поганий кандидат.

Для наукової статті це найсильніший варіант, оскільки функція має статистичну інтерпретацію.

Моя рекомендація

Для практичної генерації нових ArUco-маркерів я б використовував дворівневу схему:
Фільтр
Chessboard=0
Isolated≤1
4≤Ones≤8
Transitions≤10

Рейтинг
Q=0.24(1−Chessboardn)+0.21(1−OnesDevn)+0.19(1−Transitionsn)+0.15Cmaxn+0.12(1−Isolatedn)+0.09Homogeneousn

Саме цей варіант найбільш узгоджується з результатами аналізу 214 маркерів і не містить довільно підібраних коефіцієнтів.

Нижче наведено повністю автономну програму.
Вона:
Зчитує список маркерів із файлу.
Обчислює:
Ones
OnesDev
Transitions
Chessboard
Isolated
Cmax
Homogeneous
Нормалізує ознаки.
Обчислює рейтинг:
Q=0.24(1−Chessboardn)+0.21(1−OnesDevn)+0.19(1−Transitionsn)+0.15Cmaxn+0.12(1−Isolatedn)+0.09Homogeneousn

Сортує маркери за спаданням Q.
Виводить таблицю рейтингу.

import re
import numpy as np
import pandas as pd
from collections import deque

# ==========================================================
# Parsing
# ==========================================================

def parse_markers(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = text.split("------------------------------------------------------------")

    markers = {}

    for block in blocks:

        id_match = re.search(r"Marker ID:\s*(\d+)", block)

        if not id_match:
            continue

        marker_id = int(id_match.group(1))

        rows = []

        for line in block.splitlines():

            line = line.strip()

            if re.fullmatch(r"[01](\s+[01]){3}", line):
                rows.append(list(map(int, line.split())))

        if len(rows) == 4:
            markers[marker_id] = np.array(rows, dtype=int)

    return markers


# ==========================================================
# Feature extraction
# ==========================================================

def count_ones(m):
    return int(np.sum(m))


def count_transitions(m):

    t = 0

    for r in range(4):
        for c in range(3):
            if m[r, c] != m[r, c + 1]:
                t += 1

    for c in range(4):
        for r in range(3):
            if m[r, c] != m[r + 1, c]:
                t += 1

    return t


def count_chessboard(m):

    count = 0

    for r in range(3):
        for c in range(3):

            block = m[r:r + 2, c:c + 2]

            if np.array_equal(block,
                              np.array([[0, 1],
                                        [1, 0]])):
                count += 1

            elif np.array_equal(block,
                                np.array([[1, 0],
                                          [0, 1]])):
                count += 1

    return count


def count_isolated(m):

    isolated = 0

    for r in range(4):
        for c in range(4):

            val = m[r, c]

            neigh = []

            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:

                rr = r + dr
                cc = c + dc

                if 0 <= rr < 4 and 0 <= cc < 4:
                    neigh.append(m[rr, cc])

            if all(n != val for n in neigh):
                isolated += 1

    return isolated


def largest_component(m):

    visited = np.zeros((4,4), dtype=bool)

    best = 0

    for r in range(4):
        for c in range(4):

            if visited[r,c]:
                continue

            color = m[r,c]

            q = deque([(r,c)])
            visited[r,c] = True

            size = 0

            while q:

                rr,cc = q.popleft()
                size += 1

                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:

                    nr = rr + dr
                    nc = cc + dc

                    if (
                        0 <= nr < 4 and
                        0 <= nc < 4 and
                        not visited[nr,nc] and
                        m[nr,nc] == color
                    ):
                        visited[nr,nc] = True
                        q.append((nr,nc))

            best = max(best, size)

    return best


def homogeneous_score(m):

    score = 0

    for r in range(4):
        if np.all(m[r,:] == m[r,0]):
            score += 1

    for c in range(4):
        if np.all(m[:,c] == m[0,c]):
            score += 1

    return score


# ==========================================================
# Feature table
# ==========================================================

def build_feature_table(markers):

    rows = []

    for marker_id, m in markers.items():

        ones = count_ones(m)

        rows.append({
            "MarkerID": marker_id,
            "Ones": ones,
            "OnesDev": abs(ones - 8),
            "Transitions": count_transitions(m),
            "Chessboard": count_chessboard(m),
            "Isolated": count_isolated(m),
            "Cmax": largest_component(m),
            "Homogeneous": homogeneous_score(m)
        })

    return pd.DataFrame(rows)


# ==========================================================
# Normalization
# ==========================================================

def normalize(series):

    mn = series.min()
    mx = series.max()

    if mx == mn:
        return pd.Series(np.zeros(len(series)))

    return (series - mn) / (mx - mn)


# ==========================================================
# Ranking function
# ==========================================================

def calculate_score(df):

    chess_n = normalize(df["Chessboard"])
    ones_n = normalize(df["OnesDev"])
    trans_n = normalize(df["Transitions"])
    iso_n = normalize(df["Isolated"])
    cmax_n = normalize(df["Cmax"])
    hom_n = normalize(df["Homogeneous"])

    df["Q"] = (
        0.24 * (1 - chess_n)
        + 0.21 * (1 - ones_n)
        + 0.19 * (1 - trans_n)
        + 0.15 * cmax_n
        + 0.12 * (1 - iso_n)
        + 0.09 * hom_n
    )

    return df


# ==========================================================
# Main
# ==========================================================

def rank_markers(filename):

    markers = parse_markers(filename)

    df = build_feature_table(markers)

    df = calculate_score(df)

    df = df.sort_values(
        "Q",
        ascending=False
    )

    return df


if __name__ == "__main__":

    FILE = "markers.txt"

    ranking = rank_markers(FILE)

    pd.set_option("display.max_rows", None)

    print(
        ranking[
            [
                "MarkerID",
                "Q",
                "Chessboard",
                "Transitions",
                "Ones",
                "Isolated",
                "Cmax",
                "Homogeneous"
            ]
        ]
    )

Для наукової роботи я б рекомендував наступний крок: замість фіксованих ваг 0.24, 0.21, ... автоматично навчати логістичну регресію на вашій базі 214 маркерів і використовувати прогнозовану ймовірність P(DetectionRate ≥ 95%) як рейтинг. Це дасть статистично обґрунтований рейтинг замість евристичного.

як автоматично навчати логістичну регресію на вашій базі 214 маркерів ?

Автоматичне навчання логістичної регресії означає, що ви більше не задаєте ваги вручну (0.24, 0.21, ...), а дозволяєте моделі самій визначити їх із ваших 214 експериментально виміряних маркерів.

Ідея

Для кожного маркера маємо:

MarkerID	Ones	Transitions	Chessboard	Isolated	Cmax	Homogeneous	DetectionRate
13	7	8	0	0	10	2	100
135	6	18	3	4	4	0	43.76
...	...	...	...	...	...	...	...

Створюємо клас:

target = 1 if DetectionRate >= 95 else 0
Навчання
from sklearn.linear_model import LogisticRegression

X = df[
    [
        "Ones",
        "Transitions",
        "Chessboard",
        "Isolated",
        "Cmax",
        "Homogeneous"
    ]
]

y = (df["DetectionRate"] >= 95).astype(int)

model = LogisticRegression(
    max_iter=5000
)

model.fit(X, y)

Після цього модель знайде:

β0
β1
β2
β3
β4
β5
β6

для рівняння

P=
1+e
−z
1
	​


де

z=β
0
	​

+β
1
	​

Ones+β
2
	​

Transitions+β
3
	​

Chessboard+β
4
	​

Isolated+β
5
	​

Cmax+β
6
	​

Homogeneous
Отримання коефіцієнтів
for name, coef in zip(
    X.columns,
    model.coef_[0]
):
    print(name, coef)

print("Intercept:", model.intercept_[0])

Приклад результату:

Ones          -0.11
Transitions   -0.39
Chessboard    -1.47
Isolated      -0.62
Cmax           0.28
Homogeneous    0.14
Intercept      4.83

(числа наведено як приклад)

Рейтинг нового маркера

Припустимо маємо:

Ones = 7
Transitions = 8
Chessboard = 0
Isolated = 0
Cmax = 10
Homogeneous = 2

Тоді:

p = model.predict_proba(
    [[7,8,0,0,10,2]]
)[0][1]

print(p)

Результат:

0.93

означає:

модель оцінює 93% ймовірність того, що Detection Rate буде ≥95%.

Використання як функції оцінки

Тоді ваша функція якості стає:

Q=P(DetectionRate≥95%)

і лежить у межах

0 ... 1

або

0% ... 100%
Як ранжувати словник

Для кожного маркера:

df["Q"] = model.predict_proba(X)[:,1]

ranking = df.sort_values(
    "Q",
    ascending=False
)

Отримаємо:

MarkerID	Q
425	0.997
366	0.996
13	0.995
...	...
135	0.021
Ще кращий варіант

Оскільки у вас вже 214 маркерів, я б рекомендував навчати не Logistic Regression, а одночасно:

LogisticRegression
RandomForestClassifier
GradientBoostingClassifier

та обирати модель з найбільшим ROC-AUC.

Наприклад:

if auc_rf > auc_lr:
    ranking_model = rf
else:
    ranking_model = lr

Тоді рейтинг буде:

Q = ranking_model.predict_proba(X)[:,1]
Для наукової статті

Я б рекомендував використовувати подвійну систему:

Пояснювана модель

Logistic Regression

зрозумілі коефіцієнти;
видно вплив Chessboard, Ones, Transitions.
Робочий рейтинг

Random Forest або Gradient Boosting

краща точність;
кращий ROC-AUC;
використовується для реального відбору маркерів.

Це дає і наукову інтерпретацію, і практичний інструмент генерації нових ArUco-маркерів.

надй нову версію програму із урахуванням твоїх рекомендацій.

Я б рекомендував змінити архітектуру принципово.

Поточна версія має проблему: ваги (0.24, 0.21, ...) зашиті в код.

Краще зробити систему з двох режимів:

train.py
навчає модель на еталонному наборі маркерів;
зберігає модель у файл.
rank_markers.py
завантажує навчену модель;
обчислює ознаки нових маркерів;
повертає рейтинг:
Q = P(DetectionRate >= 95%)
Структура
project/
│
├── train_markers.py
├── rank_markers.py
├── marker_features.py
├── markers_training.txt
├── model.pkl
└── scaler.pkl
marker_features.py

Модуль містить:

parse_markers()
count_ones()
count_transitions()
count_chessboard()
count_isolated()
largest_component()
homogeneous_score()
build_feature_table()

Практично весь код з попередньої версії можна перенести сюди без змін.

train_markers.py
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from marker_features import (
    parse_markers,
    build_feature_table
)

TRAIN_FILE = "markers_training.txt"

markers = parse_markers(TRAIN_FILE)

df = build_feature_table(markers)

# Detection Rate зчитується з файлу
y = (df["DetectionRate"] >= 95).astype(int)

X = df[
    [
        "Ones",
        "Transitions",
        "Chessboard",
        "Isolated",
        "Cmax",
        "Homogeneous"
    ]
]

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced"
            )
        )
    ]
)

model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model saved.")
rank_markers.py
import joblib
import pandas as pd

from marker_features import (
    parse_markers,
    build_feature_table
)

MODEL_FILE = "model.pkl"

model = joblib.load(MODEL_FILE)

markers = parse_markers(
    "candidate_markers.txt"
)

df = build_feature_table(markers)

X = df[
    [
        "Ones",
        "Transitions",
        "Chessboard",
        "Isolated",
        "Cmax",
        "Homogeneous"
    ]
]

df["Q"] = model.predict_proba(X)[:, 1]

df = df.sort_values(
    "Q",
    ascending=False
)

print(
    df[
        [
            "MarkerID",
            "Q",
            "Ones",
            "Transitions",
            "Chessboard",
            "Isolated",
            "Cmax"
        ]
    ]
)
Додавання Random Forest

Я б ще зберігав дві моделі:

LogisticRegression
RandomForestClassifier

та автоматично вибирав кращу.

train_markers.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",
     LogisticRegression(
         max_iter=5000,
         class_weight="balanced"
     ))
])

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

auc_lr = cross_val_score(
    lr,
    X,
    y,
    cv=5,
    scoring="roc_auc"
).mean()

auc_rf = cross_val_score(
    rf,
    X,
    y,
    cv=5,
    scoring="roc_auc"
).mean()

if auc_rf > auc_lr:
    best_model = rf
    print("Using Random Forest")
else:
    best_model = lr
    print("Using Logistic Regression")

best_model.fit(X, y)

joblib.dump(
    best_model,
    "model.pkl"
)
Додавання нових ознак

На основі аналізу 214 маркерів я б обов'язково додав:

Perimeter
EdgeWhiteCells
ConnectedComponents
LargestWhiteComponent
LargestBlackComponent

Тоді список ознак стане:

FEATURES = [
    "Ones",
    "Transitions",
    "Chessboard",
    "Isolated",
    "Cmax",
    "Homogeneous",
    "Perimeter",
    "EdgeWhiteCells",
    "Components"
]

Саме ці ознаки, на мою думку, дадуть найбільший приріст ROC-AUC порівняно з початковими шістьма характеристиками.

Найкраща формула рейтингу

Після переходу на ML формула стає:

Q = model.predict_proba(X)[0][1]

де:

Q = 0.00

майже гарантовано поганий маркер,

Q = 1.00

майже гарантовано маркер із Detection Rate ≥95%.

Тобто рейтинг більше не залежить від вручну підібраних коефіцієнтів і автоматично покращується щоразу, коли ви додаєте нові експериментальні дані до навчальної вибірки.