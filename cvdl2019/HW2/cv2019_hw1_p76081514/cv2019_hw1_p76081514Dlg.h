
// cv2019_hw1_p76081514Dlg.h : ���Y��
//

#pragma once


// Ccv2019_hw1_p76081514Dlg ��ܤ��
class Ccv2019_hw1_p76081514Dlg : public CDialogEx
{
// �غc
public:
	Ccv2019_hw1_p76081514Dlg(CWnd* pParent = NULL);	// �зǫغc�禡

// ��ܤ�����
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_CV2019_HW1_P76081514_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV �䴩


// �{���X��@
protected:
	HICON m_hIcon;

	// ���ͪ��T�������禡
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
