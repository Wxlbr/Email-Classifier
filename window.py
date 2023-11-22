import sys
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QLabel,
    QMainWindow
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the title
        self.setWindowTitle("My Awesome App")

        # Set the size of the window
        self.setFixedSize(QSize(400, 300))

        # Set menu bar
        menu = self.menuBar()

        # Set a custom height for the menu bar
        menu.setFixedHeight(40)  # Adjust the height as needed

        # Action for opening the home page
        home_action = QAction("Home", self)
        home_action.triggered.connect(self.openHomePage)

        # Custom widget for the action
        home_widget = QLabel("Home")
        home_widget.setAlignment(Qt.AlignCenter)

        menu.addAction(home_action)

        # Action for opening the settings page
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.openSettingsPage)

        # Custom widget for the action
        settings_widget = QLabel("Settings")
        settings_widget.setAlignment(Qt.AlignCenter)

        menu.addAction(settings_action)

    def openHomePage(self):
        print("Opening Home Page")

    def openSettingsPage(self):
        print("Opening Settings Page")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
