# Sample code for model training (updated)
from sklearn.ensemble import RandomForestClassifier
import pickle

# Define and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open('matching_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
