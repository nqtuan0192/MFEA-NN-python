from InputHandler import InputHandler
def main():
    inputHandler = InputHandler()
    #inputHandler.TicTacToe('./dataset/tic-tac-toe/tic-tac-toe.data')
    #inputHandler.ionosphere('./dataset/ionosphere/ionosphere.data')
    inputHandler.creditScreening('./dataset/credit-screening/crx.data')
    
    pass

if __name__ == '__main__':
    main()