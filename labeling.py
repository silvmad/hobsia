#! /usr/bin/env python3

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import os
import sys

def main(args):
    app = QApplication(args)
    w = MainWindow()
    w.show()
    app.exec_()
    
class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fileButton = QPushButton("Choisir un fichier",self)
        self.startButton = QPushButton("Commencer l'étiquetage", self)
        self.quitButton = QPushButton("Quitter", self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.fileButton)
        layout.addWidget(self.startButton)
        layout.addWidget(self.quitButton)
        self.resize(300, 150)

        # Mettre la fenêtre au centre de l'écran.
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
        
        self.choose_file()
        self.fileButton.clicked.connect(self.choose_file)
        self.startButton.clicked.connect(self.label)
        self.quitButton.clicked.connect(self.close)

    @pyqtSlot()
    def choose_file(self):
        home = "/home/{}".format(os.getlogin())
        ret = QFileDialog.getOpenFileName(self,
                                          "Sélection du fichier à étiqueter",
                                          home,
                                          "Text (*.csv, *.txt);;All Files (*)")
        self.filename = ret[0]

    @pyqtSlot()
    def label(self):
        if(len(self.filename) == 0):
            QMessageBox.warning(self,
                                "Pas de fichier choisi",
                                "Vous devez d'abord choisir un fichier à étiqueter.")
            return
        with open(self.filename, "r") as f:
            data = [m.split("\t;\t") for m in f.read().splitlines()]

        for entry in data:
            # Ignorer les messages déjà étiquetés.
            if (len(entry) > 1):
                continue
            
            mbox = QMessageBox()
            mbox.setText("Ce message est-il haineux ?")
            mbox.setInformativeText(entry[0])
            yes_but = mbox.addButton(QMessageBox.Yes)
            no_but = mbox.addButton(QMessageBox.No)
            stop_button = mbox.addButton("Arrêter l'étiquetage", QMessageBox.RejectRole)
            mbox.setDefaultButton(stop_button)
            
            mbox.exec()
            if (mbox.clickedButton() == yes_but):
                entry.append('1')
            elif (mbox.clickedButton() == no_but):
                entry.append('0')
            elif (mbox.clickedButton() == stop_button):
                break
                
        if (mbox.clickedButton() == stop_button):
            title = "Travail interrompu"
            text = "Vous avez interrompu votre travail, souhaitez-vous le sauvegarder ?"
        else:
            title = "Fichier entièrement étiqueté"
            text = "Le fichier est entièrement étiqueté, souhaitez-vous sauvegarder votre travail ?"
            
        ret = QMessageBox.question(self,
                                   title,
                                   text,
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.Yes)
        if (ret == QMessageBox.Yes):
            self.save(data)
        else :
            ret = QMessageBox.warning(self,
                                      "Quitter sans sauvegarder ?",
                                      "Attention, tout votre travail sera perdu, êtes-vous sûr de vouloir continuer ?",
                                      QMessageBox.Discard | QMessageBox.Save,
                                      QMessageBox.Save)
            if (ret == QMessageBox.Save):
                self.save(data)

    def save(self, data):
        target = QFileDialog.getSaveFileName(self,
                                             "Enregistrer sous",
                                             self.filename,
                                             "Text (*.csv, *.txt);;All Files (*)")
        while (target[0] == ''):
            ret = QMessageBox.warning(self,
                                      "Aucun fichier choisi",
                                      "Attention, votre travail ne sera pas enregistré, souhaitez-vous continuer ?",
                                      QMessageBox.Discard | QMessageBox.Save,
                                      QMessageBox.Save)
            if (ret == QMessageBox.Discard):
                return
            target = QFileDialog.getSaveFileName(self,
                                                 "Enregistrer sous",
                                                 self.filename,
                                                 "Text (*.csv, *.txt);;All Files (*)")
        with open(target[0], "w") as f:
            for entry in data:
                f.write("\t;\t".join(entry) + '\n')
            
if __name__ == "__main__":
    main(sys.argv)
