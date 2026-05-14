from src.app.agent import run
import os

if __name__ == "__main__":

    os.system('clear')

    while True:
        question = input("Sua Mensagem: ")

        if question == "sair":
            break

        result = run(question)
        print(result)

