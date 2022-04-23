from src.app import App

learning_rate = 0.01
number_of_epochs = 1000

train_directory = '../data/train/'
test_directory = '../data/test/'


def main():
    app = App(learning_rate, number_of_epochs, train_directory, test_directory)
    app.evaluate_model()
    app.predict_user_input()


if __name__ == '__main__':
    main()
