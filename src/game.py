from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from src.configs import DriverVariables


class ChromeDinoGame:

    def __init__(self, custom_config=True):
        """
        Selenium interface between the python and browser.
        Since this is about the game driver, details are not critical for DQL project.
        """
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(executable_path=DriverVariables.CHROME_DRIVER_PATH,
                                        chrome_options=chrome_options)
        self._driver.set_window_position(x=-10, y=0)
        try:
            self._driver.get(DriverVariables.GAME_URL)
        except WebDriverException:
            pass
        self._driver.execute_script("Runner.config.ACCELERATION=0")
        # Create id for canvas for faster selection from DOM
        self._driver.execute_script("document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'")

    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")

    def press_up(self):
        self._driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ARROW_UP)

    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        score = ''.join(score_array)
        return int(score)

    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")

    def end(self):
        self._driver.close()
