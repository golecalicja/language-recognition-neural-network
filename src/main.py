from src.app import App

alpha = 0.01
number_of_epochs = 1000

train_directory = '../data/train/'
test_directory = '../data/test/'


def main():
    app = App(alpha, number_of_epochs, train_directory, test_directory)
    app.evaluate_model()
    app.predict_user_input()


if __name__ == '__main__':
    main()
