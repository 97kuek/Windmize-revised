#Windmize – TR-797 最適循環分布 + 構造弾性計算

#必要なライブラリをインポート
import sys, os, csv, copy, json
import numpy as np
import numpy

from PyQt5 import QtCore, QtGui, QtWidgets
Qt = QtCore.Qt           
QApplication      = QtWidgets.QApplication
QMainWindow       = QtWidgets.QMainWindow
QWidget           = QtWidgets.QWidget
QTabWidget        = QtWidgets.QTabWidget
QGroupBox         = QtWidgets.QGroupBox
QFrame            = QtWidgets.QFrame
QLabel            = QtWidgets.QLabel
QLineEdit         = QtWidgets.QLineEdit
QPushButton       = QtWidgets.QPushButton
QCheckBox         = QtWidgets.QCheckBox
QProgressBar      = QtWidgets.QProgressBar
QTableWidget      = QtWidgets.QTableWidget
QTableWidgetItem  = QtWidgets.QTableWidgetItem
QHeaderView       = QtWidgets.QHeaderView
QSizePolicy       = QtWidgets.QSizePolicy
QHBoxLayout       = QtWidgets.QHBoxLayout
QVBoxLayout       = QtWidgets.QVBoxLayout
QFileDialog       = QtWidgets.QFileDialog
QMessageBox       = QtWidgets.QMessageBox
QSplashScreen     = QtWidgets.QSplashScreen
QPixmap           = QtGui.QPixmap
QIcon             = QtGui.QIcon

import matplotlib
matplotlib.use("Qt5Agg")   # ensure Qt 5 backend
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

#表示用のクラスを定義
class Dataplot(FigureCanvas):
    """Reusable matplotlib canvas with helper ‘drawplot’ method."""
    def __init__(self, parent=None, width=8, height=3, dpi=80):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.tick_params(axis='both', which='major', labelsize=20)
        super().__init__(self.fig)                         # FigureCanvas init
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
    def drawplot(self, x, y, x2=None, y2=None, *,
                 xlabel=None, ylabel=None, legend=None, aspect="equal"):

        self.axes.plot(x, y)
        if x2 is not None and y2 is not None:
            self.axes.plot(x2, y2, "--")
        if xlabel:
            self.axes.set_xlabel(xlabel, fontsize=20)
        if ylabel:
            self.axes.set_ylabel(ylabel, fontsize=20)
        if legend:
            self.axes.legend(legend, fontsize=15)
        if aspect == "auto":
            self.axes.set_aspect("auto")
        self.draw()

#タブウィジェットを定義
class ResultTabWidget(QTabWidget):
    """タブ切替でグラフ結果を表示するウィジェット"""
    def __init__(self, parent=None):
        super().__init__(parent)

        self.circulation_graph  = Dataplot()
        self.bending_graph      = Dataplot()
        self.bendingangle_graph = Dataplot()
        self.moment_graph       = Dataplot()
        self.shforce_graph      = Dataplot()
        self.ind_graph          = Dataplot()

        self.addTab(self.circulation_graph,  "循環分布")
        self.addTab(self.ind_graph,          "誘導角度[deg]")
        self.addTab(self.bending_graph,      "たわみ(軸:等倍)")
        self.addTab(self.bendingangle_graph, "たわみ角[deg]")
        self.addTab(self.moment_graph,       "曲げモーメント[N·m]")
        self.addTab(self.shforce_graph,      "せん断力[N]")

# ボタンと進捗バーを定義
class ExeExportButton(QWidget):
    """計算実行・CSV出力・進捗バー"""
    def __init__(self, parent=None):
        super().__init__(parent)

        self.exebutton      = QPushButton("計算",       self)
        self.exportbutton   = QPushButton("CSV出力",    self)
        self.do_stracutual  = QCheckBox("構造考慮",     self)
        self.do_stracutual.setChecked(True)

        self.progressbar = QProgressBar(self)
        self.progressbar.setTextVisible(True)
        self.progressbar.setStyleSheet("""
            QProgressBar{
                border: 2px solid grey; border-radius: 5px; text-align: center;
            }
            QProgressBar::chunk{
                background-color: lightblue; width: 10px; margin: 1px;
            }
        """)

        lay = QHBoxLayout(self)
        lay.addStretch()
        lay.addWidget(self.progressbar)
        lay.addWidget(self.do_stracutual)
        lay.addWidget(self.exebutton)
        lay.addWidget(self.exportbutton)

#設計パラメータ入力パネルを定義
class SettingWidget(QGroupBox):
    """設計パラメータ入力パネル（スパン分割テーブル等）"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("設計変数　※各翼終端位置は計算精度確保のため dy の整数倍とするとよい")
        font = QtGui.QFont()
        font.setPointSize(12)
        self.setFont(font)
        #上段：揚力・最大たわみ・ワイヤー位置等の入力
        self.lift_maxbending_input = QWidget(self)
        lbl = self.lift_maxbending_input
        lbl.liftlabel      = QLabel("揚力(kgf) :",        lbl)
        lbl.bendinglabel   = QLabel("  最大たわみ(mm) :", lbl)
        lbl.wireposlabel   = QLabel("  ワイヤー取付位置(mm) :", lbl)
        lbl.forcewirelabel = QLabel("  ワイヤー下向引張(N) :",  lbl)
        lbl.velocitylabel  = QLabel("  速度(m/s) :",      lbl)
        lbl.dylabel        = QLabel("  dy(mm) :",         lbl)
        def _le(width, text):
            le = QLineEdit(lbl)
            le.setFixedWidth(width)
            le.setText(text)
            return le

        lbl.liftinput     = _le(25,  "103")
        lbl.velocityinput = _le(33,  "7.21")
        lbl.bendinginput  = _le(33,  "2401")
        lbl.wireposinput  = _le(33,  "7500")
        lbl.forcewireinput= _le(25,  "130")
        lbl.dyinput       = _le(25,  "50")

        h1 = QHBoxLayout(lbl)
        h1.addStretch()
        for w in (lbl.liftlabel, lbl.liftinput,
                  lbl.velocitylabel, lbl.velocityinput,
                  lbl.bendinglabel, lbl.bendinginput,
                  lbl.wireposlabel, lbl.wireposinput,
                  lbl.forcewirelabel, lbl.forcewireinput,
                  lbl.dylabel, lbl.dyinput):
            h1.addWidget(w)
        lbl.setLayout(h1)

        # 下段：桁剛性・スパン分割
        self.EIinput = QFrame(self)
        self.EIinput.EIinputbutton = QPushButton("桁詳細設定", self.EIinput)
        self.EIinput.EIinputbutton.setFixedWidth(100)
        h2 = QHBoxLayout(self.EIinput)
        h2.addStretch()
        h2.addWidget(self.EIinput.EIinputbutton)
        self.strechwid = QFrame(self)
        self.tablewidget = QTableWidget(self.strechwid)
        self.tablewidget.setMaximumSize(1000, 100)
        self.tablewidget.setMinimumSize(600, 100)
        self.tablewidget.setColumnCount(7)
        self.tablewidget.setRowCount(2)

        headers = ["", "第1翼", "第2翼", "第3翼", "第4翼", "第5翼", "第6翼"]
        for i, h in enumerate(headers):
            self.tablewidget.setHorizontalHeaderItem(i, QTableWidgetItem(h))

        self.tablewidget.setItem(0, 0, QTableWidgetItem("終端(mm)"))
        self.tablewidget.setItem(1, 0, QTableWidgetItem("調整係数"))
        for r in (0, 1):
            self.tablewidget.item(r, 0).setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

        span_list = [1100, 4300, 7500, 10200, 13150, 16250]
        for i in range(6):
            self.tablewidget.setItem(0, i + 1, QTableWidgetItem(str(span_list[i])))
            self.tablewidget.setItem(1, i + 1, QTableWidgetItem("1"))
        self.tablewidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tablewidget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tablewidget.buttons = QWidget(self)
        self.tablewidget.insertcolumn = QPushButton("列追加", self.tablewidget.buttons)
        self.tablewidget.deletecolumn = QPushButton("列削除", self.tablewidget.buttons)
        hb = QHBoxLayout(self.tablewidget.buttons)
        hb.addStretch()
        hb.addWidget(self.tablewidget.insertcolumn)
        hb.addWidget(self.tablewidget.deletecolumn)
        hf = QHBoxLayout(self.strechwid)
        hf.addStretch()
        hf.addWidget(self.tablewidget)
        vlay = QVBoxLayout(self)
        vlay.addWidget(self.lift_maxbending_input)
        vlay.addWidget(self.tablewidget.buttons)
        vlay.addWidget(self.strechwid)
        vlay.addWidget(self.EIinput)
        self.setLayout(vlay)

#計算結果ラベルを定義
class ResultValWidget(QGroupBox):
    """計算結果ラベルの束"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("計算結果")
        self.liftresultlabel  = QLabel("計算揚力[kgf] : --", self)
        self.Diresultlabel    = QLabel("   抗力[N] : --",      self)
        self.swresultlabel    = QLabel("   桁重量概算[kg] : --",self)
        self.lambda1label     = QLabel("   構造制約係数λ1[-] : --", self)
        self.lambda2label     = QLabel("   揚力制約係数λ2[-] : --", self)
        h = QHBoxLayout(self)
        h.addStretch()
        for w in (self.liftresultlabel, self.Diresultlabel,
                  self.swresultlabel, self.lambda1label, self.lambda2label):
            h.addWidget(w)

# EI設定ダイアログを定義
class EIsettingWidget(QtWidgets.QDialog):
    """桁剛性 / 線密度 入力用ダイアログ (複数タブ)"""
    def __init__(self, tablewidget, parent=None):
        super().__init__(parent)
        self.setFixedSize(600, 170)
        self.setModal(True)

        self.tabwidget = QTabWidget(self)
        v = QVBoxLayout(self)
        v.addWidget(self.tabwidget)
    def EIsetting(self, tablewidget):
        section_num = tablewidget.columnCount() - 1
        self.tabwidget.clear()
        self.EIinputWidget = []
        for i in range(section_num):
            gb = QGroupBox(f"第{i+1}翼の剛性と線密度を入力してください", self)
            table = QTableWidget(gb)
            table.setColumnCount(5)
            table.setRowCount(3)
            table.setFixedSize(570, 100)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

            for r, title in enumerate(("翼区切終端[mm]", "EI", "線密度[kg/m]")):
                table.setItem(r, 0, QTableWidgetItem(title))
                table.item(r, 0).setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            glay = QVBoxLayout(gb)
            glay.addWidget(table)
            self.tabwidget.addTab(gb, f"第{i+1}翼")

            # keep reference
            gb.EIinputtable = table
            self.EIinputWidget.append(gb)


#ここから計算パート！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
class TR797_modified():
    def __init__(self):
        #リスト変数の定義
        self.dy = 0.05

        self.y_div = []
        self.z_div = []
        self.y_section = []
        self.Ndiv_sec = []
        self.y = []
        self.z = []
        self.phi = []

        self.dS = []

        self.sigma = []
        self.spar_weight = 0
        #sigma_wireは線密度＋ワイヤー引っ張りを考慮した下向きの[N/m]
        self.sigma_wire = []

        #多角形化行列
        self.polize_mat = [[]]

        #誘導速度行列
        self.Q_ij = [[]]

        #せん断力を積分によって求める行列
        self.sh_mat = [[]]

        #モーメントを積分によって求める行列
        self.mo_mat = [[]]

        #たわみ角を積分によって求める行列
        #剛性値ベクトル
        self.EI = []
        self.vd_mat = []

        #たわみを求める行列
        self.v_mat = []

        #構造制約行列B
        self.B = [[]]

        #揚力制約行列C
        self.C = [[]]

        #最適化行列A
        self.A = [[]]

        #最適循環値
        self.gamma = []
        #誘導速度見積もり
        self.ind_vel = []

        #計算中スイッチ
        self.run = 1
        #出力可能な値があるか
        self.comp = 1

    def prepare(self,settingwidget,eisettingwidget):
        self.dy = float(settingwidget.lift_maxbending_input.dyinput.text()) / 1000
        self.b = round(float(settingwidget.tablewidget.item(0,settingwidget.tablewidget.columnCount() - 1).text()) * 2 / 1000,4)
        self.n_section = int(settingwidget.tablewidget.columnCount()) - 1
        self.max_tawami = float(settingwidget.lift_maxbending_input.bendinginput.text()) / 1000
        self.y_wire = float(settingwidget.lift_maxbending_input.wireposinput.text()) / 1000
        self.rho = 1.184
        self.U = float(settingwidget.lift_maxbending_input.velocityinput.text())
        self.M = float(settingwidget.lift_maxbending_input.liftinput.text())
        #セクションの区切りの位置
        for n in range(self.n_section):
            self.y_section.append(float(settingwidget.tablewidget.item(0,n + 1).text()) / 1000)

        i = 0
        j = 0
        while True:
            self.y_div.append(round(self.dy * (i + 1),4))
            if round(self.y_div[i],4) > round(self.y_section[j]  - self.dy / 2,4) and round(self.y_div[i],4) <= round(self.y_section[j] + self.dy / 2,4):
                self.Ndiv_sec.append(i)
                j = j + 1
            if round(self.y_div[i],4) > round(self.y_wire  - self.dy / 2,4) and round(self.y_div[i],4) <= round(self.y_wire + self.dy / 2,4):
                self.Ndiv_wire = i

            if j == self.n_section:
                break
            i = i + 1

        #パネル幅dSの作成
        coe_tawami = self.max_tawami / (self.b / 2) ** 2
        for n in range(len(self.y_div)):
            self.z_div.append(coe_tawami * self.y_div[n] ** 2)
            if n != 0:
                self.dS.append(numpy.sqrt((self.y_div[n]-self.y_div[n-1])**2+(self.z_div[n]-self.z_div[n-1])**2) / 2)
                self.y.append((self.y_div[n]+self.y_div[n-1]) / 2)
                self.z.append((self.z_div[n]+self.z_div[n-1]) / 2)
                self.phi.append(numpy.arctan((self.z_div[n]-self.z_div[n-1]) / (self.y_div[n]-self.y_div[n-1])))
            else:
                self.dS.append(numpy.sqrt(self.y_div[n]**2+self.z_div[n]**2) / 2)
                self.y.append(self.y_div[n] / 2)
                self.z.append(self.z_div[n] / 2)
                self.phi.append(numpy.arctan(self.z_div[n] / self.y_div[n]))
        #control pointにおける桁剛性、線密度
        n = 0
        for i_wings in range(self.n_section) :
            j = 1
            coe_EI = float(settingwidget.tablewidget.item(1,i_wings + 1).text())
            while True:
                if i_wings == 0:
                    if round(self.y[n],4) < round(float(eisettingwidget.EIinputWidget[i_wings].EIinputtable.item(0,j).text()) / 1000 ,4):
                        self.EI.append(float(eisettingwidget.EIinputWidget[i_wings].EIinputtable.item(1,j).text()) * coe_EI)
                        self.sigma.append(float(eisettingwidget.EIinputWidget[i_wings].EIinputtable.item(2,j).text()) *coe_EI)
                    else:
                        self.EI.append(float(eisettingwidget.EIinputWidget[i_wings].EIinputtable.item(1,j).text()) *coe_EI)
                        self.sigma.append(float(eisettingwidget.EIinputWidget[i_wings].EIinputtable.item(2,j).text()) *coe_EI)
                        j = j + 1

                elif i_wings != 0:
                    if round(self.y[n],4) < round(float(eisettingwidget.EIinputWidget[i_wings].EIinputtable.item(0,j).text()) / 1000 + self.y_section[i_wings-1],4):
                        self.EI.append(float(eisettingwidget.EIinputWidget[i_wings].EIinputtable.item(1,j).text())* coe_EI)
                        self.sigma.append(float(eisettingwidget.EIinputWidget[i_wings].EIinputtable.item(2,j).text())* coe_EI)
                    else:
                        self.EI.append(float(eisettingwidget.EIinputWidget[i_wings].EIinputtable.item(1,j).text())* coe_EI)
                        self.sigma.append(float(eisettingwidget.EIinputWidget[i_wings].EIinputtable.item(2,j).text())* coe_EI)
                        j = j + 1
                n = n + 1
                if n == len(self.y):
                    break
                if not round(self.y[n],4) < round(self.y_section[i_wings],4):
                    break



        self.spar_weight = numpy.sum(numpy.array(self.sigma) * numpy.array(self.dS) * 2) * 2

        self.sigma_wire = copy.deepcopy(self.sigma)
        for i in range(len(self.y)):
            self.sigma_wire[i] *= 9.8
        self.sigma_wire[self.Ndiv_wire] += float(settingwidget.lift_maxbending_input.forcewireinput.text()) / self.dS[self.Ndiv_wire] / 2
    def matrix(self,progressbar,qApp):
        def calc_Q(y,z,phi,dS,progressbar):
            Q_ij = numpy.zeros([len(y),len(y)])
            yd_ij = numpy.zeros([len(y),len(y)])
            zd_ij = numpy.zeros([len(y),len(y)])
            ydd_ij = numpy.zeros([len(y),len(y)])
            zdd_ij = numpy.zeros([len(y),len(y)])

            R_2_Pij = numpy.zeros([len(y),len(y)])
            R_2_Mij = numpy.zeros([len(y),len(y)])
            Rd_2_Pij = numpy.zeros([len(y),len(y)])
            Rd_2_Mij = numpy.zeros([len(y),len(y)])

            Q_ij_1 = numpy.zeros([len(y),len(y)])
            Q_ij_2 = numpy.zeros([len(y),len(y)])
            Q_ij_3 = numpy.zeros([len(y),len(y)])
            Q_ij_4 = numpy.zeros([len(y),len(y)])

            for i in range (len(y)):
                #中止フラグを検知
                if self.run == 1:
                    break

                for j in range(len(y)):
                    qApp.processEvents()
                    progressbar.setValue(int((i*len(y)+(j+1))/len(y)**2*100))
                    yd_ij[i,j] =  (y[i] - y[j]) * numpy.cos(phi[j]) + (z[i]-z[j]) * numpy.sin(phi[j])
                    zd_ij[i,j] = -(y[i] - y[j]) * numpy.sin(phi[j]) + (z[i]-z[j]) * numpy.cos(phi[j])
                    ydd_ij[i,j] = (y[i] + y[j]) * numpy.cos(phi[j]) - (z[i]-z[j]) * numpy.sin(phi[j])
                    zdd_ij[i,j] = (y[i] + y[j]) * numpy.sin(phi[j]) + (z[i]-z[j]) * numpy.cos(phi[j])

                    R_2_Pij[i,j] = (yd_ij[i,j] - dS[j]) ** 2 + zd_ij[i,j] ** 2
                    R_2_Mij[i,j] = (yd_ij[i,j] + dS[j]) ** 2 + zd_ij[i,j] ** 2
                    Rd_2_Pij[i,j] = (ydd_ij[i,j] + dS[j]) ** 2 + zdd_ij[i,j] ** 2
                    Rd_2_Mij[i,j] = (ydd_ij[i,j] - dS[j])**2 + zdd_ij[i,j] ** 2

                    Q_ij_1[i,j] = ((yd_ij[i,j] - dS[j]) / R_2_Pij[i,j] - (yd_ij[i,j] + dS[j]) / R_2_Mij[i,j]) * numpy.cos(phi[i]-phi[j])
                    Q_ij_2[i,j] = ((zd_ij[i,j]) / R_2_Pij[i,j] - (zd_ij[i,j]) / R_2_Mij[i,j]) * numpy.sin(phi[i] - phi[j])
                    Q_ij_3[i,j] = ((ydd_ij[i,j] - dS[j]) / Rd_2_Mij[i,j] - (ydd_ij[i,j] + dS[j]) / Rd_2_Pij[i,j]) * numpy.cos(phi[i]+phi[j]);
                    Q_ij_4[i,j] = ((zdd_ij[i,j]) / Rd_2_Mij[i,j] - (zdd_ij[i,j]) / Rd_2_Pij[i,j]) * numpy.sin(phi[i]+phi[j])

                    Q_ij[i,j] = -1 / 2 / numpy.pi * (Q_ij_1[i,j] + Q_ij_2[i,j] + Q_ij_3[i,j] + Q_ij_4[i,j])
            return Q_ij


        self.Q_ij = calc_Q(self.y,self.z,self.phi,self.dS,progressbar)
        #-----多角形化行列
        self.polize_mat = numpy.zeros([len(self.y),self.n_section])
        for i in range(self.Ndiv_sec[1]):
            self.polize_mat[i,0]           = 1
            self.polize_mat[i,self.n_section-1] = 0

        for j in range(1,self.n_section):
            for i in range(self.Ndiv_sec[j-1] + 1, self.Ndiv_sec[j] + 1):
                self.polize_mat[i,j-1]   =  -(self.y[i]-self.y_section[j])   / (self.y_section[j]-self.y_section[j-1])
                self.polize_mat[i,j]     =   (self.y[i]-self.y_section[j-1]) / (self.y_section[j]-self.y_section[j-1])

        #積分によりせん断力Qを求める
        self.sh_mat = numpy.zeros([len(self.y),len(self.y)])
        for j in range(len(self.y)-1,-1,-1):
            for i in range(j,-1,-1):
                if j == i:
                    self.sh_mat[i,j] = self.dS[j]
                else:
                    self.sh_mat[i,j] = self.dS[j] * 2
        self.sh_mat = self.sh_mat * self.U * self.rho
        #積分によりモーメントを求める
        self.mo_mat = numpy.zeros([len(self.y),len(self.y)])
        for j in range(len(self.y)-1,-1,-1):
            for i in range(j,-1,-1):
                if j == i:
                    self.mo_mat[i,j] = self.dS[j]
                else:
                    self.mo_mat[i,j] = self.dS[j] * 2

        #積分によりたわみ角を求める行列
        self.vd_mat = numpy.zeros([len(self.y),len(self.y)])
        for i in range(len(self.y)-1,-1,-1):
            for j in range(i,-1,-1):
                if j == i:
                    self.vd_mat[i,j] = self.dS[j] / self.EI[j]* 10 ** 6
                else:
                    self.vd_mat[i,j] = self.dS[j] / self.EI[j] * 10 ** 6 * 2

        self.v_mat = numpy.zeros([len(self.y),len(self.y)])
        for i in range(len(self.y)-1,-1,-1):
            for j in range(i,-1,-1):
                if j == i:
                    self.v_mat[i,j] = self.dS[j]
                else:
                    self.v_mat[i,j] = self.dS[j] * 2

        #制約となる撓みとなる位置（固定)
        B_want = numpy.zeros([1,len(self.y)])
        B_want[0,len(self.y)-1] = 1

        #構造制約行列
        self.B = numpy.dot(B_want,numpy.dot(self.v_mat,numpy.dot(self.vd_mat,numpy.dot(self.mo_mat,numpy.dot(self.sh_mat,self.polize_mat)))))
        self.B_val = self.max_tawami + numpy.dot(numpy.dot(B_want,numpy.dot(self.v_mat,numpy.dot(self.vd_mat,numpy.dot(self.mo_mat,self.sh_mat)))),numpy.array(self.sigma_wire).T / self.rho / self.U)

        #-----揚力制約条件行列
        self.C = 4 * self.rho * self.U * numpy.dot(self.dS,self.polize_mat)
        #-----揚力制約条件の値
        self.C_val = self.M * 9.8


    def optimize(self,checkbox):
        def calc_ellipticallift(self):
            root_gamma = self.M * 9.8 * 4 / numpy.pi / self.b / self.U / self.rho
            self.gamma_el = numpy.zeros(len(self.y))
            for i in range(len(self.y)):
                self.gamma_el[i] = (numpy.sqrt(root_gamma ** 2 -(self.y[i] * root_gamma / self.b * 2)**2 ))


        if checkbox.checkState() == 2:
            #構造考慮
            A = copy.deepcopy(self.Q_ij)
            for i in range(A.shape[1]):
                for j in range(A.shape[0]):
                    A[j,i] = A[j,i] * self.dS[j] * 2
            Mat_indD = A * self.rho
            A = (A + A.T)
            A = self.rho * numpy.dot(self.polize_mat.T,numpy.dot(A,self.polize_mat))
            A = numpy.vstack((A,-self.B))
            A = numpy.vstack((A,-self.C))
            A = numpy.column_stack((A,numpy.append(-self.B,[0,0]).T))
            A = numpy.column_stack((A,numpy.append(-self.C,[0,0]).T))
            A_val = numpy.zeros([A.shape[0],1])
            A_val[A.shape[0]-2,0] = -self.B_val
            A_val[A.shape[0]-1,0] = -self.C_val

            self.Optim_Answer = numpy.linalg.solve(A,A_val)
            self.lambda1 = self.lambda2 = self.Optim_Answer[self.n_section-1,0]
            self.lambda2 = self.Optim_Answer[self.n_section,0]
            self.gamma_opt = self.Optim_Answer[0:self.n_section,:]

        else:
            #構造無視
            A = copy.deepcopy(self.Q_ij)
            for i in range(A.shape[1]):
                for j in range(A.shape[0]):
                    A[j,i] = A[j,i] * self.dS[j] * 2
            A = (A + A.T)
            A = self.rho * numpy.dot(self.polize_mat.T,numpy.dot(A,self.polize_mat))
            A = numpy.vstack((A,-self.C))
            A = numpy.column_stack((A,numpy.append(-self.C,[0]).T))
            A_val = numpy.zeros([A.shape[0],1])
            A_val[A.shape[0]-1,0] = -self.C_val

            self.Optim_Answer = numpy.linalg.solve(A,A_val)
            self.lambda1 = 0
            self.lambda2 = self.Optim_Answer[self.n_section,0]
            self.gamma_opt = self.Optim_Answer[0:self.n_section,:]

        self.bending_mat = numpy.dot(self.v_mat,numpy.dot(self.vd_mat,numpy.dot(self.mo_mat,self.sh_mat)))
        self.shearForce = numpy.dot(self.sh_mat,(numpy.dot(self.polize_mat,self.gamma_opt) - numpy.array([self.sigma_wire]).T / self.U / self.rho))
        self.moment = numpy.dot(numpy.dot(self.mo_mat,self.sh_mat),(numpy.dot(self.polize_mat,self.gamma_opt) - numpy.array([self.sigma_wire]).T / self.U / self.rho))
        self.bending_angle =numpy.dot(numpy.dot(self.vd_mat,numpy.dot(self.mo_mat,self.sh_mat)),(numpy.dot(self.polize_mat,self.gamma_opt) - numpy.array([self.sigma_wire]).T / self.U / self.rho))
        self.bending = numpy.dot(self.bending_mat,(numpy.dot(self.polize_mat,self.gamma_opt) - numpy.array([self.sigma_wire]).T / self.U / self.rho))

        calc_ellipticallift(self)
        self.gamma = numpy.dot(self.polize_mat,self.gamma_opt)
        self.ind_vel = numpy.dot(self.Q_ij / 2 ,self.gamma)
        self.Di = 0
        self.Lift = numpy.dot(self.C,self.gamma_opt)[0]
        for i in range(len(self.y)):
            self.Di += self.rho * self.ind_vel[i] * self.gamma[i] * self.dy * 2
        self.Di = self.Di[0]


#GUI のメイン関数
def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("WM.ico"))

    # Splash
    splash = QSplashScreen(QPixmap("WM_splash.png"))
    splash.show()

    # Widgets
    mainwindow        = QMainWindow()
    mainwindow.setWindowTitle("Windmize")
    resulttabwidget   = ResultTabWidget()
    exeexportbutton   = ExeExportButton()
    settingwidget     = SettingWidget()
    resultvalwidget   = ResultValWidget()
    eisettingwidget   = EIsettingWidget(settingwidget.tablewidget)
    eisettingwidget.EIsetting(settingwidget.tablewidget)

    #EIサンプル
    EI_samples     = [3.4375e10, 3.6671e10, 1.6774e10, 8.3058e9, 1.8648e9, 7.094e7]
    sigma_samples  = [0.377, 0.357, 0.284, 0.245, 0.0929, 0.0440]
    for i_wing in range(settingwidget.tablewidget.columnCount()-1):
        for s in range(4):
            tbl = eisettingwidget.EIinputWidget[i_wing].EIinputtable
            tbl.setItem(1, s+1, QTableWidgetItem(str(EI_samples[i_wing])))
            tbl.setItem(2, s+1, QTableWidgetItem(str(sigma_samples[i_wing])))

    TR797_opt = TR797_modified()
    # レイアウト
    def insertcolumn():
        tw = settingwidget.tablewidget
        col = tw.columnCount()
        tw.setColumnCount(col+1)
        tw.setHorizontalHeaderItem(col, QTableWidgetItem(f"第{col}翼"))
        tw.setItem(0, col, QTableWidgetItem(str(
            float(tw.item(0, col-1).text()) + 2000)))
        tw.setItem(1, col, QTableWidgetItem("1"))
        tw.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # EIタブ追加
        i = col-1
        eisettingwidget.EIinputWidget.append(QGroupBox(f"第{i+1}翼の剛性と線密度を入力してください"))
        gb = eisettingwidget.EIinputWidget[i]
        table = QTableWidget(3,5, gb);  table.setFixedSize(570,100)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for r, t in enumerate(("翼区切終端[mm]","EI[N·mm^2]","線密度[kg/m]")):
            table.setItem(r,0, QTableWidgetItem(t))
            table.item(r,0).setFlags(Qt.ItemIsSelectable|Qt.ItemIsEnabled)
        gb.EIinputtable = table
        gl = QVBoxLayout(gb); gl.addWidget(table)
        eisettingwidget.tabwidget.addTab(gb, f"第{i+1}翼")

    def deletecolumn():
        tw = settingwidget.tablewidget
        col = tw.columnCount()
        if col >= 3:
            tw.setColumnCount(col-1)
            tw.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            eisettingwidget.tabwidget.removeTab(col-2)
            eisettingwidget.EIinputWidget.pop(col-2)

    def EIsettingshow():
        init_EI_widget()
        eisettingwidget.show()

    def init_EI_widget():
        """列数・終端位置に合わせて EI ダイアログをリフレッシュ"""
        y_div = [float(settingwidget.tablewidget.item(0,i+1).text())
                 for i in range(settingwidget.tablewidget.columnCount()-1)]
        for i, gb in enumerate(eisettingwidget.EIinputWidget):
            if i == 0:
                base = y_div[i]
            else:
                base = y_div[i] - y_div[i-1]
            pos = [base/4, base/2, base*3/4, base]
            for s in range(4):
                itm = QTableWidgetItem(str(pos[s]))
                itm.setFlags(Qt.ItemIsSelectable|Qt.ItemIsEnabled)
                gb.EIinputtable.setItem(0, s+1, itm)

    # グラフの初期化
    def show_results():
        resulttabwidget.circulation_graph.axes.clear()
        resulttabwidget.circulation_graph.drawplot(
            np.array(TR797_opt.y).T,
            TR797_opt.gamma,
            np.array(TR797_opt.y),
            TR797_opt.gamma_el,
            xlabel="y[m]", ylabel="gamma[m^2/s]",
            legend=("optimized","elliptical"), aspect="auto")

        resulttabwidget.bending_graph.axes.clear()
        resulttabwidget.bending_graph.drawplot(
            np.array(TR797_opt.y), TR797_opt.bending,
            xlabel="y[m]", ylabel="bending[m]")

        resulttabwidget.ind_graph.axes.clear()
        resulttabwidget.ind_graph.drawplot(
            np.array(TR797_opt.y),
            np.degrees(np.arctan(-TR797_opt.ind_vel / TR797_opt.U)),
            xlabel="y[m]", ylabel="induced angle[deg]", aspect="auto")

        resulttabwidget.bendingangle_graph.axes.clear()
        resulttabwidget.bendingangle_graph.drawplot(
            np.array(TR797_opt.y),
            np.degrees(TR797_opt.bending_angle),
            xlabel="y[m]", ylabel="bending angle[deg]")

        resulttabwidget.moment_graph.axes.clear()
        resulttabwidget.moment_graph.drawplot(
            np.array(TR797_opt.y),
            TR797_opt.moment,
            xlabel="y[m]", ylabel="moment[Nm]")

        resulttabwidget.shforce_graph.axes.clear()
        resulttabwidget.shforce_graph.drawplot(
            np.array(TR797_opt.y),
            TR797_opt.shearForce,
            xlabel="y[m]", ylabel="shearforce[N]")

        # 数値ラベル
        resultvalwidget.liftresultlabel.setText(f"計算揚力[kgf] : {TR797_opt.Lift/9.8:.3f}")
        resultvalwidget.Diresultlabel.setText(  f"   抗力[N] : {TR797_opt.Di:.3f}")
        resultvalwidget.swresultlabel.setText(  f"   桁重量概算[kg] : {TR797_opt.spar_weight:.3f}")
        if exeexportbutton.do_stracutual.isChecked():
            resultvalwidget.lambda1label.setText(f"   構造制約係数λ1[-] : {TR797_opt.lambda1:.3f}")
        else:
            resultvalwidget.lambda1label.setText("   構造制約係数λ1[-] : --")
        resultvalwidget.lambda2label.setText(  f"   揚力制約係数λ2[-] : {TR797_opt.lambda2:.3f}")

    def calculation():
        if TR797_opt.run:                                
            exeexportbutton.exebutton.setText("計算中止")
            init_EI_widget()
            TR797_opt.__init__()                         
            TR797_opt.prepare(settingwidget, eisettingwidget)
            TR797_opt.run = 0
            TR797_opt.matrix(exeexportbutton.progressbar, app)
            if not TR797_opt.run:
                TR797_opt.optimize(exeexportbutton.do_stracutual)
                show_results()
                TR797_opt.comp = 0   # ← ここを追加
            TR797_opt.run = 1
            exeexportbutton.exebutton.setText("計算")
        else:                                           
            TR797_opt.run = 1
            exeexportbutton.exebutton.setText("計算")

    def exportCSV():
        if TR797_opt.comp:
            QMessageBox.warning(mainwindow,"Error","計算が完了していません")
            return
        fname, _ = QFileDialog.getSaveFileName(mainwindow, "計算結果出力", "", "CSV (*.csv)")
        if not fname: return
        try:
            with open(fname, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["循環分布最化結果"])
                w.writerow(["揚力[kgf]",      round(TR797_opt.Lift/9.8,3)])
                w.writerow(["抗力[N]",        round(TR797_opt.Di,3)])
                w.writerow(["桁重量概算[kg]", round(TR797_opt.spar_weight,3)])
                w.writerow(["構造制約係数[-]",
                            "--" if not exeexportbutton.do_stracutual.isChecked()
                            else round(TR797_opt.lambda1,3)])
                w.writerow(["揚力制約係数[-]", round(TR797_opt.lambda2,3)])
                w.writerow([])
                w.writerow(["以下 翼セクションでの値"])
                w.writerow(["y[m]", "gamma[m^2/s]"])
                buf = np.hstack([np.array([TR797_opt.y_section]).T, TR797_opt.gamma_opt])
                for row in buf: w.writerow(row)
                w.writerow([])
                w.writerow(["以下 Control Point での値"])
                w.writerow(["y[m]","gamma","ind[deg]","bending[m]","bendAng[deg]",
                            "moment","shear","EI","sigma"])
                for i, y in enumerate(TR797_opt.y):
                    w.writerow([y,
                                TR797_opt.gamma[i,0],
                                np.degrees(np.arctan(TR797_opt.ind_vel[i,0]/TR797_opt.U)),
                                TR797_opt.bending[i,0],
                                np.degrees(TR797_opt.bending_angle[i,0]),
                                TR797_opt.moment[i,0],
                                TR797_opt.shearForce[i,0],
                                TR797_opt.EI[i],
                                TR797_opt.sigma[i]])
        except Exception as e:
            QMessageBox.warning(mainwindow, "Error", f"CSV出力に失敗しました\n{e}")

    def save_settings():
        """現在の設定をJSONファイルに保存"""
        fname, _ = QFileDialog.getSaveFileName(mainwindow, "設定を保存", "", "JSON (*.json)")
        if not fname:
            return
        try:
            data = {
                "lift": settingwidget.lift_maxbending_input.liftinput.text(),
                "velocity": settingwidget.lift_maxbending_input.velocityinput.text(),
                "bending": settingwidget.lift_maxbending_input.bendinginput.text(),
                "wirepos": settingwidget.lift_maxbending_input.wireposinput.text(),
                "forcewire": settingwidget.lift_maxbending_input.forcewireinput.text(),
                "dy": settingwidget.lift_maxbending_input.dyinput.text(),
                "span_list": [settingwidget.tablewidget.item(0, i+1).text() for i in range(settingwidget.tablewidget.columnCount()-1)],
                "coef_list": [settingwidget.tablewidget.item(1, i+1).text() for i in range(settingwidget.tablewidget.columnCount()-1)],
                "EI_sigma": [
                    [
                        [gb.EIinputtable.item(r, c).text() if gb.EIinputtable.item(r, c) else ""
                        for c in range(gb.EIinputtable.columnCount())]
                        for r in range(gb.EIinputtable.rowCount())
                    ]
                    for gb in eisettingwidget.EIinputWidget
                ]
            }
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.warning(mainwindow, "Error", f"設定保存に失敗しました\n{e}")

    def load_settings():
        """JSONファイルから設定を読み込む"""
        fname, _ = QFileDialog.getOpenFileName(mainwindow, "設定を読み込む", "", "JSON (*.json)")
        if not fname:
            return
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 上段入力
            settingwidget.lift_maxbending_input.liftinput.setText(data["lift"])
            settingwidget.lift_maxbending_input.velocityinput.setText(data["velocity"])
            settingwidget.lift_maxbending_input.bendinginput.setText(data["bending"])
            settingwidget.lift_maxbending_input.wireposinput.setText(data["wirepos"])
            settingwidget.lift_maxbending_input.forcewireinput.setText(data["forcewire"])
            settingwidget.lift_maxbending_input.dyinput.setText(data["dy"])
            # スパン分割
            col = len(data["span_list"])
            settingwidget.tablewidget.setColumnCount(col+1)
            for i, val in enumerate(data["span_list"]):
                settingwidget.tablewidget.setItem(0, i+1, QTableWidgetItem(val))
            for i, val in enumerate(data["coef_list"]):
                settingwidget.tablewidget.setItem(1, i+1, QTableWidgetItem(val))
            settingwidget.tablewidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            # EI/線密度
            eisettingwidget.EIinputWidget.clear()
            eisettingwidget.tabwidget.clear()
            for i, tabdata in enumerate(data["EI_sigma"]):
                gb = QGroupBox(f"第{i+1}翼の剛性と線密度を入力してください")
                table = QTableWidget(len(tabdata), len(tabdata[0]), gb)
                table.setFixedSize(570, 100)
                table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
                for r, row in enumerate(tabdata):
                    for c, val in enumerate(row):
                        item = QTableWidgetItem(val)
                        if c == 0:
                            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                        table.setItem(r, c, item)
                gb.EIinputtable = table
                gl = QVBoxLayout(gb)
                gl.addWidget(table)
                eisettingwidget.tabwidget.addTab(gb, f"第{i+1}翼")
                eisettingwidget.EIinputWidget.append(gb)
        except Exception as e:
            QMessageBox.warning(mainwindow, "Error", f"設定読込に失敗しました\n{e}")

    # メインウィンドウのメニューバー
    menubar   = mainwindow.menuBar()
    filemenu  = menubar.addMenu("File")
    filemenu.addAction("ファイルI/O未実装")

    filemenu.addSeparator()
    act_save = filemenu.addAction("設定保存")
    act_load = filemenu.addAction("設定読込")
    act_save.triggered.connect(save_settings)
    act_load.triggered.connect(load_settings)

    aboutmenu = menubar.addMenu("About")
    act_about = aboutmenu.addAction("About Windmize")
    act_about.triggered.connect(lambda: QMessageBox.about(
        mainwindow, "About Windmize",
        "<h2>Windmize 1.00</h2>"
        "<p>Copyright (C) 2014 Naoto Morita</p>"
        "<p>Windmize is without any warranty. "
        "This program has been developed excusively for the design of aerowings."
        "Any other usage is strongly disapproved.</p>"
        "<p>Distributed under the GNU GPL.</p>"))

    settingwidget.tablewidget.insertcolumn.clicked.connect(insertcolumn)
    settingwidget.tablewidget.deletecolumn.clicked.connect(deletecolumn)
    settingwidget.EIinput.EIinputbutton.clicked.connect(EIsettingshow)
    exeexportbutton.exebutton.clicked.connect(calculation)
    exeexportbutton.exportbutton.clicked.connect(exportCSV)

    mainpanel = QWidget()
    vlay = QVBoxLayout(mainpanel)
    vlay.addWidget(resulttabwidget)
    vlay.addWidget(resultvalwidget)
    vlay.addWidget(exeexportbutton)
    vlay.addWidget(settingwidget)
    mainwindow.setCentralWidget(mainpanel)

    mainwindow.resize(1200, 900)
    mainwindow.show()
    splash.finish(mainwindow)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()