Unicode True

!ifndef APP_NAME
  !define APP_NAME "Face Crop Studio"
!endif

!ifndef APP_PUBLISHER
  !define APP_PUBLISHER "Face Crop Studio"
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
InstallDir "$PROGRAMFILES64\Face Crop Studio"
InstallDirRegKey HKLM "Software\Face Crop Studio" "Install_Dir"
RequestExecutionLevel admin

!include "MUI2.nsh"
!define MUI_ICON "${DIST_DIR}\app_icon.ico"
!define MUI_UNICON "${DIST_DIR}\app_icon.ico"

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

  CreateDirectory "$SMPROGRAMS\Face Crop Studio"
  CreateShortCut "$SMPROGRAMS\Face Crop Studio\Face Crop Studio.lnk" "$INSTDIR\yunet-gui.exe" "" "$INSTDIR\yunet-gui.exe" 0
  CreateShortCut "$SMPROGRAMS\Face Crop Studio\Uninstall Face Crop Studio.lnk" "$INSTDIR\Uninstall.exe" "" "$INSTDIR\yunet-gui.exe" 0

  WriteRegStr HKLM "Software\Face Crop Studio" "Install_Dir" "$INSTDIR"

  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "DisplayName" "${APP_NAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "DisplayVersion" "${APP_VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "Publisher" "${APP_PUBLISHER}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "DisplayIcon" "$INSTDIR\yunet-gui.exe"
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "NoRepair" 1
SectionEnd

Section "Uninstall"
  Delete "$SMPROGRAMS\Face Crop Studio\Face Crop Studio.lnk"
  Delete "$SMPROGRAMS\Face Crop Studio\Uninstall Face Crop Studio.lnk"
  RMDir "$SMPROGRAMS\Face Crop Studio"

  Delete "$INSTDIR\Uninstall.exe"
  RMDir /r "$INSTDIR"

  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio"
  DeleteRegKey HKLM "Software\Face Crop Studio"
SectionEnd
