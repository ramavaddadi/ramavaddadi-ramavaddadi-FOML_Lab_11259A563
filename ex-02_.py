from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load Iris dataset
X, y = load_iris(return_X_y=True)

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# Create KNN model
model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)

# Calculate accuracy
accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)