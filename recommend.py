import joblib
from preprocess import preprocess_text

# Load the vectorizer and model
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("journal_recommendation_model.pkl")

def recommend_journal(title, abstract):
    """Recommend a journal based on title and abstract."""
    combined_text = title + " " + abstract
    combined_text_cleaned = preprocess_text(combined_text)  # Reuse your preprocess_text function
    X_input = tfidf_vectorizer.transform([combined_text_cleaned])
    return model.predict(X_input)[0]

# Example usage
title = "Design, control and evaluation of a treadmill-based Pelvic Exoskeleton (PeXo) with self-paced walking mode"
abstract = "Most gait rehabilitation exoskeletons focus only on assisting lower limb motions. However, the pelvis plays an essential role in overground ambulation. This calls for the need of gait training devices that allow full control and assistance of user’s pelvis to improve gait rehabilitation therapies of stroke survivors. This paper presents a new pelvic assistance treadmill-based exoskeleton (PeXo) for gait rehabilitation. The system includes a total of 5 actively controlled Degrees of Freedom (DOFs), plus a passive one, in order to provide all DOFs of the human pelvis. The paper describes the mechatronic design and haptic control of PeXo, and introduces a newly developed self-paced walking mode where the user can control walking speed at his/her will. The system was evaluated by means of a pilot study with a stroke survivor. Results showed PeXo can be used safely and reliably during walking activities, allowing for the natural motion of the pelvis with reduced interaction forces. However, the use of the platform reduced user’s Range of Motion at the pelvis and lower limb joints, whereas lower limb’s muscle activity increased to compensate the disturbances introduced by the platform. Nevertheless, the user reported a positive feedback when using the system, suggesting potential and promising advantages of PeXo that should be explored in a larger study in the future."
print("Recommended Journal:", recommend_journal(title, abstract))
