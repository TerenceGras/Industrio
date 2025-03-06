import sys
from Industrio_Augmenter import PromptAugment
from Industrio_Model_Trainer import PromptTrain

if __name__ == '__main__':

    message = "🚀 WELCOME TO INDUSTRIO CNN 🚀"
    border = "*" * 50
    padding = (50 - len(message) - 2) // 2  # Calculate space padding

    print("\n" + border)
    print("*" + " " * padding + message + " " * (50 - len(message) - padding - 2) + "*")
    print(border + "\n")

    print("\n" + "=" * 50)
    print("🤖 What should we do now? 🤖")
    print("=" * 50)
    print(" 1️⃣  I want to augment my dataset. 💪")
    print(" 2️⃣  I want to train my model using an existing dataset. 🧠")
    print(" 3️⃣  I want to test my model using an existing model file. 🧪")
    print("=" * 50 + "\n")

    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        PromptAugment()
    elif choice == "2":
        PromptTrain()
    elif choice == "3":
        print("WIP")
    else:
        print("💀 Input error! Shutting down... 💀")
        sys.exit()




