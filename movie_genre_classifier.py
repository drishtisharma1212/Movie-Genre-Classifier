import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## DATASET ##
data = {
    "description": [
        "A police officer fights terrorists in a skyscraper",
        "A spy saves the country from a dangerous villian",
        "Explosions and car chases across the city",
        "A retired soldier takes revenge on criminals",
        "A ninja fights an army of assassins",
        "A ghost haunts a family in an old mansion",
        "A demon possesses a young child",
        "A cursed doll kills people at night",
        "People get trapped in a hauted hospital",
        "A monster attacks campers in the forest",
        "Two best friends ruin a wedding accidentally",
        "A man keeps getting into funny misunderstandings",
        "Friends go on a hilarious road trip",
        "A comedian pretends to be a millionaire",
        "Three roommates create chaos in college"
    ],
    "genre": [
        "Action",
        "Action",
        "Action",
        "Action",
        "Action",
        "Horror",
        "Horror",
        "Horror",
        "Horror",
        "Horror",
        "Comedy",
        "Comedy",
        "Comedy",
        "Comedy",
        "Comedy"
    ],
}
# convert dataset into DataFrame
df = pd.DataFrame(data)

## FEATURES AND LABLES ##
X = df["description"]
Y = df["genre"]

## TEXT VECTORIOZATION ##
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2)
)
X_vectors = vectorizer.fit_transform(X)

## TRAIN TEST SPLIT ##
X_train, X_test, Y_train, Y_test = train_test_split(
    X_vectors,
    Y,
    test_size=0.2,
    random_state=42
)

## MODEL TRAINING ##
model = LogisticRegression()
model.fit(X_train, Y_train)

## MODEL TESTING ##
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("\n==================================")
print("MOVIE GENRE CLASSIFIER TRAINED ")
print("================================")
print(f"\nModel Accuracy: {accuracy: .2f}")

## SAVE MODEL ##
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print(f"\nModel and vectorizer saved successfully!")

## PREDICTION LOOP ##
print("\n=================================")
print(" ENTER MOVIE DESCRIPTIONS ")
print(" Type 'exit' to close program ")
print("===================================")
while True:
    user_input = input("\nEnter movie description: ")
    
    # Exit condition
    if user_input.lower() == "exit":
        print("\nProgram closed.")
        break

    # Convert text into vector
    user_vector = vectorizer.transform([user_input])

    # Predict genre
    prediction = model.predict(user_vector)[0]

    # Get probabilities
    probabilities = model.predict_proba(user_vector)

    # Confidence score
    confidence = max(probabilities[0])*100

    # Output
    print(f"\nPredicted Genre: {prediction}")
    print(f"Confidence Score: {confidence: .2f}%")
