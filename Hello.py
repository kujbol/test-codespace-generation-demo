
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Create the Streamlit app
def main():
    st.title("scikit-learn Demo")
    st.write("This app demonstrates the usage of scikit-learn for machine learning tasks.")
    
    # Display the Iris dataset
    st.subheader("Iris Dataset")
    st.write(iris.data)
    
    # Display the target variable
    st.subheader("Target Variable")
    st.write(iris.target)
    
    # Display the training and testing sets
    st.subheader("Training and Testing Sets")
    st.write("X_train:", X_train)
    st.write("X_test:", X_test)
    st.write("y_train:", y_train)
    st.write("y_test:", y_test)
    
    # Display the trained model coefficients
    st.subheader("Model Coefficients")
    st.write(model.coef_)
    
    # Display the predictions and accuracy
    st.subheader("Predictions")
    st.write("Predicted values:", y_pred)
    st.write("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
