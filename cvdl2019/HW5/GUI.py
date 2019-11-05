# -*- coding: utf-8 -*-
import sys
from PyQt5 import QtWidgets

import UI as main_menu
Ui_MainWindow = main_menu.Ui_Dialog

class CoperQt(QtWidgets.QDialog,Ui_MainWindow):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        #self.retranslateUi(self)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CoperQt()
    window.show()
    sys.exit(app.exec_())