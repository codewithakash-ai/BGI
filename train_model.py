import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = [
    # Road
    "road broken",
    "road damaged",
    "potholes on road",
    "street not good",
    "road not good",
    "main road has cracks",
    "bad condition of street",
    "bridge road is broken",
    # Water
    "no water",
    "water supply problem",
    "water leakage",
    "dirty water",
    "no water supply",
    "pipeline leakage near home",
    "tap water not coming",
    "drinking water is dirty",
    # Electricity
    "no electricity",
    "power cut",
    "light not working",
    "electric issue",
    "transformer problem",
    "frequent power outage",
    "voltage fluctuation issue",
    "electricity supply failed",
    # Garbage
    "garbage not collected",
    "waste problem",
    "dirty area",
    "trash issue",
    "garbage issue",
    "dustbin overflowing",
    "waste not picked",
    "unclean surroundings",
    # Health
    "hospital issue",
    "no doctor available",
    "health problem",
    "medical help needed",
    "need doctor",
    "clinic has no medicine",
    "patient not getting treatment",
    "ambulance not available",
    # Fire
    "fire emergency",
    "fire in building",
    "fire brigade needed",
    "smoke problem",
    "short circuit fire",
    "house is burning",
    "need fire service quickly",
    "flames seen in market",
]

labels = [
    # Road
    "road",
    "road",
    "road",
    "road",
    "road",
    "road",
    "road",
    "road",
    # Water
    "water",
    "water",
    "water",
    "water",
    "water",
    "water",
    "water",
    "water",
    # Electricity
    "electricity",
    "electricity",
    "electricity",
    "electricity",
    "electricity",
    "electricity",
    "electricity",
    "electricity",
    # Garbage
    "garbage",
    "garbage",
    "garbage",
    "garbage",
    "garbage",
    "garbage",
    "garbage",
    "garbage",
    # Health
    "health",
    "health",
    "health",
    "health",
    "health",
    "health",
    "health",
    "health",
    # Fire
    "fire",
    "fire",
    "fire",
    "fire",
    "fire",
    "fire",
    "fire",
    "fire",
]

# Better text understanding: include unigram + bigram features.
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved ✅")

# Validation checks
tests = [
    ("road not good", "road"),
    ("no water supply", "water"),
    ("power cut", "electricity"),
    ("garbage issue", "garbage"),
    ("need doctor", "health"),
    ("fire emergency", "fire"),
]

all_ok = True
for text, expected in tests:
    pred = model.predict(vectorizer.transform([text]))[0]
    ok = pred == expected
    all_ok = all_ok and ok
    print(f"{text!r} -> {pred} (expected: {expected})")

if not all_ok:
    raise SystemExit("Validation failed: one or more test predictions did not match expected labels.")

print("Validation passed ✅")
