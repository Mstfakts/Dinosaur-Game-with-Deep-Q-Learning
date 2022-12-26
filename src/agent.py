class Dinosaur:

    def __init__(self, game):
        """
        This class is for determining the capability of the Dinosaur.

        :param game: 'ChromeDinoGame' in order to play the game.
        """
        self._game = game
        self.jump()

    def is_running(self):
        """
        Check if the Dinosaur still plays.
        """
        return self._game.get_playing()

    def is_crashed(self):
        """
        Check if the Dinosaur hits the cactus or not.
        """
        return self._game.get_crashed()

    def jump(self):
        """
        Make the Dinosaur jump.
        """
        self._game.press_up()
