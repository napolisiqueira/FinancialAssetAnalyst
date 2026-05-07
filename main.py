from src import agent
import os

if __name__ == "__main__":

    os.system('clear')

    while True:
        question = input("Sua Mensagem: ")

        if question == "sair":
            break

        result = agent.run(question)
        print(result)

