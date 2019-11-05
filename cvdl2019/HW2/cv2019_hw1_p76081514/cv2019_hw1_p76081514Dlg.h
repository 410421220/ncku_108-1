
// cv2019_hw1_p76081514Dlg.h : 標頭檔
//

#pragma once


// Ccv2019_hw1_p76081514Dlg 對話方塊
class Ccv2019_hw1_p76081514Dlg : public CDialogEx
{
// 建構
public:
	Ccv2019_hw1_p76081514Dlg(CWnd* pParent = NULL);	// 標準建構函式

// 對話方塊資料
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_CV2019_HW1_P76081514_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支援


// 程式碼實作
protected:
	HICON m_hIcon;

	// 產生的訊息對應函式
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedButton4();
	afx_msg void OnBnClickedButton5();
	afx_msg void OnBnClickedSplit1();
	afx_msg void OnCbnSelchangeCombo1();
};
