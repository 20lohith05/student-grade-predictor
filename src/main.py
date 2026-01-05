from model import train_model
import pandas as pd

def main():
    model,scalar = train_model()

    #USER INPUT
    age = int(input("Age: "))
    sex = input("Sex (M/F): ").upper()
    studytime = int(input("Study time (1-4): "))
    failures = int(input("Failures: "))
    absences = int(input("Absences: "))
    g1 = int(input("G1 grade: "))
    g2 = int(input("G2 grade: "))

    sex = 1 if sex == "M" else 0

    if not (0 <= g1 <= 20 and 0 <= g2 <= 20):
        print(" G1 and G2 must be between 0 and 20")
        return
    
    avg_marks = (g1 + g2) / 2
    study_efficiency = studytime / (absences + 1)
    parent_edu=4

    input_df = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "studytime": studytime,
    "failures": failures,
    "absences": absences,
    "G1": g1,
    "G2": g2,
    "avg_marks": avg_marks,
    "study_efficiency": study_efficiency,
    "parent_edu": parent_edu
    }])

    input_scaled= scalar.transform(input_df)
    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction = max(0, min(20, prediction))


    print(f"Predicted Final Grade (G3): {prediction:.2f}")

if __name__ == "__main__":
    main()
