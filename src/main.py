from model import train_model

def main():
    model = train_model()

    age = int(input("Age: "))
    sex = input("Sex (M/F): ").upper()
    studytime = int(input("Study time (1-4): "))
    failures = int(input("Failures: "))
    absences = int(input("Absences: "))
    g1 = int(input("G1 grade: "))
    g2 = int(input("G2 grade: "))

    # Encode sex
    sex = 1 if sex == "M" else 0

    # Predict
    prediction = model.predict([[
        age, sex, studytime, failures, absences, g1, g2
    ]])

    print(f"Predicted Final Grade (G3): {prediction[0]:.2f}")

if __name__ == "__main__":
    main()
