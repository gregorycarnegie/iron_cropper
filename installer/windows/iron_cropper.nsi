Unicode True

!ifndef APP_NAME
  !define APP_NAME "Iron Cropper"
!endif

!ifndef APP_PUBLISHER
  !define APP_PUBLISHER "Iron Cropper"
!endif

!ifndef APP_VERSION
  !define APP_VERSION "0.0.0"
!endif

!ifndef DIST_DIR
  !error "DIST_DIR define is required"
!endif

!ifndef OUT_FILE
  !error "OUT_FILE define is required"
!endif

Name "${APP_NAME} ${APP_VERSION}"
OutFile "${OUT_FILE}"
InstallDir "$PROGRAMFILES64\Iron Cropper"
InstallDirRegKey HKLM "Software\Iron Cropper" "Install_Dir"
RequestExecutionLevel admin

!include "MUI2.nsh"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "Install"
  SetOutPath "$INSTDIR"
  File /r "${DIST_DIR}\*.*"

  WriteUninstaller "$INSTDIR\Uninstall.exe"

  CreateDirectory "$SMPROGRAMS\Iron Cropper"
  CreateShortCut "$SMPROGRAMS\Iron Cropper\Iron Cropper GUI.lnk" "$INSTDIR\yunet-gui.exe"
  CreateShortCut "$SMPROGRAMS\Iron Cropper\Uninstall Iron Cropper.lnk" "$INSTDIR\Uninstall.exe"

  WriteRegStr HKLM "Software\Iron Cropper" "Install_Dir" "$INSTDIR"

  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\IronCropper" "DisplayName" "${APP_NAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\IronCropper" "DisplayVersion" "${APP_VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\IronCropper" "Publisher" "${APP_PUBLISHER}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\IronCropper" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\IronCropper" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\IronCropper" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\IronCropper" "NoRepair" 1
SectionEnd

Section "Uninstall"
  Delete "$SMPROGRAMS\Iron Cropper\Iron Cropper GUI.lnk"
  Delete "$SMPROGRAMS\Iron Cropper\Uninstall Iron Cropper.lnk"
  RMDir "$SMPROGRAMS\Iron Cropper"

  Delete "$INSTDIR\Uninstall.exe"
  RMDir /r "$INSTDIR"

  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\IronCropper"
  DeleteRegKey HKLM "Software\Iron Cropper"
SectionEnd

