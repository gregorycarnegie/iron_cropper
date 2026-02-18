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
!include "LogicLib.nsh"
!include "WinMessages.nsh"
!include "Sections.nsh"
!define MUI_ICON "${DIST_DIR}\app_icon.ico"
!define MUI_UNICON "${DIST_DIR}\app_icon.ico"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "Application Files (required)" SecMain
  SectionIn RO
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

Section "Desktop Shortcut" SecDesktop
  CreateShortCut "$DESKTOP\Face Crop Studio.lnk" "$INSTDIR\yunet-gui.exe" "" "$INSTDIR\yunet-gui.exe" 0
SectionEnd

Section /o "Add install directory to PATH (Current user)" SecPathUser
  nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -Command "$$p=[Environment]::GetEnvironmentVariable(''Path'',''User'');$$parts=@();if($$p){$$parts=$$p.Split('';'')|Where-Object{$$_ -and $$_.Trim() -ne ''''}};if($$parts -notcontains ''$INSTDIR''){$$parts += ''$INSTDIR'';[Environment]::SetEnvironmentVariable(''Path'',($$parts -join '';''),''User'')}"'
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "AddPathUser" 1
  Call BroadcastEnvironmentChange
SectionEnd

Section /o "Add install directory to PATH (All users)" SecPathSystem
  nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -Command "$$p=[Environment]::GetEnvironmentVariable(''Path'',''Machine'');$$parts=@();if($$p){$$parts=$$p.Split('';'')|Where-Object{$$_ -and $$_.Trim() -ne ''''}};if($$parts -notcontains ''$INSTDIR''){$$parts += ''$INSTDIR'';[Environment]::SetEnvironmentVariable(''Path'',($$parts -join '';''),''Machine'')}"'
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "AddPathSystem" 1
  Call BroadcastEnvironmentChange
SectionEnd

Section "Uninstall"
  ReadRegDWORD $0 HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "AddPathUser"
  ${If} $0 == 1
    nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -Command "$$p=[Environment]::GetEnvironmentVariable(''Path'',''User'');$$parts=@();if($$p){$$parts=$$p.Split('';'')|Where-Object{$$_ -and $$_.Trim() -ne '''' -and $$_ -ne ''$INSTDIR''}};[Environment]::SetEnvironmentVariable(''Path'',($$parts -join '';''),''User'')"'
  ${EndIf}

  ReadRegDWORD $1 HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "AddPathSystem"
  ${If} $1 == 1
    nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -Command "$$p=[Environment]::GetEnvironmentVariable(''Path'',''Machine'');$$parts=@();if($$p){$$parts=$$p.Split('';'')|Where-Object{$$_ -and $$_.Trim() -ne '''' -and $$_ -ne ''$INSTDIR''}};[Environment]::SetEnvironmentVariable(''Path'',($$parts -join '';''),''Machine'')"'
  ${EndIf}

  Call un.BroadcastEnvironmentChange

  Delete "$DESKTOP\Face Crop Studio.lnk"
  Delete "$SMPROGRAMS\Face Crop Studio\Face Crop Studio.lnk"
  Delete "$SMPROGRAMS\Face Crop Studio\Uninstall Face Crop Studio.lnk"
  RMDir "$SMPROGRAMS\Face Crop Studio"

  Delete "$INSTDIR\Uninstall.exe"
  RMDir /r "$INSTDIR"

  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio"
  DeleteRegKey HKLM "Software\Face Crop Studio"
SectionEnd

Function BroadcastEnvironmentChange
  System::Call 'USER32::SendMessageTimeout(p ${HWND_BROADCAST}, i ${WM_SETTINGCHANGE}, p 0, t "Environment", i 0, i 5000, *p .r0)'
FunctionEnd

Function un.BroadcastEnvironmentChange
  System::Call 'USER32::SendMessageTimeout(p ${HWND_BROADCAST}, i ${WM_SETTINGCHANGE}, p 0, t "Environment", i 0, i 5000, *p .r0)'
FunctionEnd

Function .onSelChange
  SectionGetFlags ${SecPathUser} $0
  IntOp $1 $0 & ${SF_SELECTED}
  SectionGetFlags ${SecPathSystem} $2
  IntOp $3 $2 & ${SF_SELECTED}

  ${If} $1 <> 0
  ${AndIf} $3 <> 0
    IntOp $4 $0 & ~${SF_SELECTED}
    SectionSetFlags ${SecPathUser} $4
  ${EndIf}
FunctionEnd
