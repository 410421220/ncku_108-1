
// cv2019_hw1_p76081514Dlg.cpp : 實作檔
//

#include "stdafx.h"
#include "cv2019_hw1_p76081514.h"
#include "cv2019_hw1_p76081514Dlg.h"
#include "afxdialogex.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 對 App About 使用 CAboutDlg 對話方塊

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 對話方塊資料
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支援

// 程式碼實作
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// Ccv2019_hw1_p76081514Dlg 對話方塊



Ccv2019_hw1_p76081514Dlg::Ccv2019_hw1_p76081514Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_CV2019_HW1_P76081514_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void Ccv2019_hw1_p76081514Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(Ccv2019_hw1_p76081514Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &Ccv2019_hw1_p76081514Dlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON4, &Ccv2019_hw1_p76081514Dlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &Ccv2019_hw1_p76081514Dlg::OnBnClickedButton5)
END_MESSAGE_MAP()


// Ccv2019_hw1_p76081514Dlg 訊息處理常式

BOOL Ccv2019_hw1_p76081514Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 將 [關於...] 功能表加入系統功能表。

	// IDM_ABOUTBOX 必須在系統命令範圍之中。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 設定此對話方塊的圖示。當應用程式的主視窗不是對話方塊時，
	// 框架會自動從事此作業
	SetIcon(m_hIcon, TRUE);			// 設定大圖示
	SetIcon(m_hIcon, FALSE);		// 設定小圖示

	// TODO: 在此加入額外的初始設定
	AllocConsole();
	freopen("CONOUT$", "w", stdout);

	return TRUE;  // 傳回 TRUE，除非您對控制項設定焦點
}

void Ccv2019_hw1_p76081514Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果將最小化按鈕加入您的對話方塊，您需要下列的程式碼，
// 以便繪製圖示。對於使用文件/檢視模式的 MFC 應用程式，
// 框架會自動完成此作業。

void Ccv2019_hw1_p76081514Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 繪製的裝置內容

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 將圖示置中於用戶端矩形
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 描繪圖示
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 當使用者拖曳最小化視窗時，
// 系統呼叫這個功能取得游標顯示。
HCURSOR Ccv2019_hw1_p76081514Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}
///////////////////////////////////////////////////////////////////////

//1.1
vector<Point2f> corners;
Mat img1[15];
string dir = "./res/images/CameraCalibration/";
Size bsize1 = Size(11, 8);
//1.2
vector<vector<Point3f>> object_points;// 棋盤上
vector<vector<Point2f>> image_points;// 圖片上，在1.1先準備了
Mat intrinsic = Mat(3, 3, CV_32FC1);
Mat distCoeffs;
vector<Mat> rvecs;//rotation，[3x1] 需使用rodrigues 改成選轉矩陣
vector<Mat> tvecs;//translation
				  //1.3
int which_img = 0;
//2
//Mat img2[15];

void find_corners() {
	cout << "find corners" << endl;
	for (int i = 0; i < 15; i++) {
		// open file
		string fpath = dir + to_string(i+1) + ".bmp";
		img1[i] = imread(fpath, CV_LOAD_IMAGE_COLOR);
		// find corner
		int found = findChessboardCorners(img1[i], bsize1, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
		// 1.2
		image_points.push_back(corners);
		//show corner
		drawChessboardCorners(img1[i], bsize1, corners, found);
	}
	cout << "end find corners" << endl;
}

void find_intrinsic() {
	int numCors = bsize1.width*bsize1.height;
	vector<Point3f> chessborad_pts;
	for (int i = 0; i<numCors; i++) {
		chessborad_pts.push_back(Point3f(i / bsize1.width, i % bsize1.width, 0.0f));
	}
	for (int i = 0; i<15; i++) {
		object_points.push_back(chessborad_pts);
	}
	calibrateCamera(object_points, image_points, bsize1, intrinsic, distCoeffs, rvecs, tvecs);
	//cout << intrinsic << endl;
}


void Ccv2019_hw1_p76081514Dlg::OnBnClickedButton1()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	// 2.1
	cout << "2.1 AR" << endl;
	if (corners.size() == 0) {
		find_corners();
	}
	if (rvecs.size() == 0) {
		find_intrinsic();
	}

	Mat img[15];
	for (int i = 0; i < 15; i++) {
		// open file
		string fpath = dir + to_string(i + 1) + ".bmp";
		img[i] = imread(fpath, CV_LOAD_IMAGE_COLOR);
	}
	vector<Point3f> cube = {
		Point3f(1,1,0),Point3f(1,-1,0),Point3f(-1,-1,0),Point3f(-1,1,0),Point3f(0,0,2)
	};
	vector<Point2f> projectedPoints;
	// rotation
	//cout << cube << endl;
	for (int i = 0; i<5; i++) {
		// 3D -> 2D
		projectPoints(cube, rvecs[i], tvecs[i], intrinsic, distCoeffs, projectedPoints);
		// draw line
		line(img[i], projectedPoints[0], projectedPoints[1], Scalar(0, 0, 255), 5);
		line(img[i], projectedPoints[1], projectedPoints[2], Scalar(0, 0, 255), 5);
		line(img[i], projectedPoints[2], projectedPoints[3], Scalar(0, 0, 255), 5);
		line(img[i], projectedPoints[3], projectedPoints[0], Scalar(0, 0, 255), 5);
		line(img[i], projectedPoints[0], projectedPoints[4], Scalar(0, 0, 255), 5);
		line(img[i], projectedPoints[1], projectedPoints[4], Scalar(0, 0, 255), 5);
		line(img[i], projectedPoints[2], projectedPoints[4], Scalar(0, 0, 255), 5);
		line(img[i], projectedPoints[3], projectedPoints[4], Scalar(0, 0, 255), 5);
	}
	namedWindow("2.1 AR", WINDOW_NORMAL);
	//while (1) {
		for (int i = 0; i<5; i++) {
			//show
			Mat tmp;
			cv::flip(img[i], tmp, -1);
			resizeWindow("2.1 AR", tmp.size[0] / 2, tmp.size[1] / 2);
			imshow("2.1 AR", tmp);
			waitKey(500);
		}
	//}
}


void Ccv2019_hw1_p76081514Dlg::OnBnClickedButton4()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	// 1.1
	if (corners.size() == 0) {
		// 尚未找出corner
		find_corners();
	}
	for (int i = 0; i < 15; i++) {
		string fpath = dir + to_string(i + 1) + ".bmp";
		//show result
		namedWindow(fpath, WINDOW_NORMAL);
		resizeWindow(fpath, img1[i].size[0] / 2, img1[i].size[1] / 2);
		imshow(fpath, img1[i]);
	}
}


void Ccv2019_hw1_p76081514Dlg::OnBnClickedButton5()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	if (corners.size() == 0) {
		// 尚未找出corner
		find_corners();
	}
	if (tvecs.size() == 0) {
		find_intrinsic();
	}
	cout << intrinsic << endl;
}


void Ccv2019_hw1_p76081514Dlg::OnBnClickedSplit1()
{
	// TODO: 在此加入控制項告知處理常式程式碼
}


void Ccv2019_hw1_p76081514Dlg::OnCbnSelchangeCombo1()
{
	// TODO: 在此加入控制項告知處理常式程式碼
}
