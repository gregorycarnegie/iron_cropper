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
!define MUI_LICENSEPAGE_TEXT_TOP "Face Crop Studio is dual-licensed (MIT OR Apache-2.0). Review the terms below."
!define MUI_LICENSEPAGE_CHECKBOX

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${DIST_DIR}\LICENSE-MIT"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

; ── Sections ──────────────────────────────────────────────────────────────────

Section "Application Files (required)" SecMain
  SectionIn RO
  SetOutPath "$INSTDIR"
  File /r "${DIST_DIR}\*.*"

  WriteUninstaller "$INSTDIR\Uninstall.exe"

  CreateDirectory "$SMPROGRAMS\Face Crop Studio"
  CreateShortCut "$SMPROGRAMS\Face Crop Studio\Face Crop Studio.lnk" "$INSTDIR\fcs-gui.exe" "" "$INSTDIR\fcs-gui.exe" 0
  CreateShortCut "$SMPROGRAMS\Face Crop Studio\Uninstall Face Crop Studio.lnk" "$INSTDIR\Uninstall.exe" "" "$INSTDIR\Uninstall.exe" 0

  WriteRegStr HKLM "Software\Face Crop Studio" "Install_Dir" "$INSTDIR"

  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "DisplayName" "${APP_NAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "DisplayVersion" "${APP_VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "Publisher" "${APP_PUBLISHER}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "DisplayIcon" "$INSTDIR\fcs-gui.exe"
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "NoRepair" 1
SectionEnd

Section "Desktop Shortcut" SecDesktop
  CreateShortCut "$DESKTOP\Face Crop Studio.lnk" "$INSTDIR\fcs-gui.exe" "" "$INSTDIR\fcs-gui.exe" 0
SectionEnd

Section /o "Add install directory to PATH (Current user)" SecPathUser
  Call AddInstallDirToUserPath
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "AddPathUser" 1
  Call BroadcastEnvironmentChange
SectionEnd

Section /o "Add install directory to PATH (All users)" SecPathSystem
  Call AddInstallDirToSystemPath
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "AddPathSystem" 1
  Call BroadcastEnvironmentChange
SectionEnd

Section "Uninstall"
  InitPluginsDir

  ReadRegDWORD $0 HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "AddPathUser"
  ${If} $0 == 1
    Call un.RemoveInstallDirFromUserPath
  ${EndIf}

  ReadRegDWORD $1 HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "AddPathSystem"
  ${If} $1 == 1
    Call un.RemoveInstallDirFromSystemPath
  ${EndIf}

  Call un.BroadcastEnvironmentChange

  Delete "$DESKTOP\Face Crop Studio.lnk"
  Delete "$SMPROGRAMS\Face Crop Studio\Face Crop Studio.lnk"
  Delete "$SMPROGRAMS\Face Crop Studio\Uninstall Face Crop Studio.lnk"
  RMDir "$SMPROGRAMS\Face Crop Studio"

  ; Delete only files this installer placed — preserves user-added files (e.g. custom models).
  Delete "$INSTDIR\fcs-cli.exe"
  Delete "$INSTDIR\fcs-gui.exe"
  Delete "$INSTDIR\app_icon.ico"
  Delete "$INSTDIR\README.md"
  Delete "$INSTDIR\LICENSE-MIT"
  Delete "$INSTDIR\LICENSE-APACHE"
  Delete "$INSTDIR\models\face_detection_yunet_2023mar_640.onnx"
  RMDir "$INSTDIR\models"
  Delete "$INSTDIR\Uninstall.exe"
  ; Remove the install directory only if it is now empty.
  RMDir "$INSTDIR"

  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio"
  DeleteRegKey HKLM "Software\Face Crop Studio"
SectionEnd

; ── PATH helpers ──────────────────────────────────────────────────────────────
;
; Each function writes a small PowerShell script to $PLUGINSDIR (a per-session
; temp dir that NSIS cleans up on exit), executes it, and returns.  Keeping the
; logic in readable multi-line PS1 files is far easier to audit than embedding
; escaped one-liners inside nsExec strings.

Function AddInstallDirToUserPath
  ; Append $INSTDIR to the current user PATH, skipping if already present.
  FileOpen $R8 "$PLUGINSDIR\fcs_path.ps1" w
  FileWrite $R8 '$$scope = "User"$\r$\n'
  FileWrite $R8 '$$dir   = "$INSTDIR"$\r$\n'
  FileWrite $R8 '$$cur   = [Environment]::GetEnvironmentVariable("Path", $$scope)$\r$\n'
  FileWrite $R8 '$$parts = if ($$cur) { $$cur.Split(";") | Where-Object { $$_.Trim() -ne "" } } else { @() }$\r$\n'
  FileWrite $R8 'if ($$parts -notcontains $$dir) {$\r$\n'
  FileWrite $R8 '    $$parts += $$dir$\r$\n'
  FileWrite $R8 '    [Environment]::SetEnvironmentVariable("Path", ($$parts -join ";"), $$scope)$\r$\n'
  FileWrite $R8 '}$\r$\n'
  FileClose $R8
  nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -File "$PLUGINSDIR\fcs_path.ps1"'
FunctionEnd

Function AddInstallDirToSystemPath
  ; Append $INSTDIR to the system (all-users) PATH, skipping if already present.
  FileOpen $R8 "$PLUGINSDIR\fcs_path.ps1" w
  FileWrite $R8 '$$scope = "Machine"$\r$\n'
  FileWrite $R8 '$$dir   = "$INSTDIR"$\r$\n'
  FileWrite $R8 '$$cur   = [Environment]::GetEnvironmentVariable("Path", $$scope)$\r$\n'
  FileWrite $R8 '$$parts = if ($$cur) { $$cur.Split(";") | Where-Object { $$_.Trim() -ne "" } } else { @() }$\r$\n'
  FileWrite $R8 'if ($$parts -notcontains $$dir) {$\r$\n'
  FileWrite $R8 '    $$parts += $$dir$\r$\n'
  FileWrite $R8 '    [Environment]::SetEnvironmentVariable("Path", ($$parts -join ";"), $$scope)$\r$\n'
  FileWrite $R8 '}$\r$\n'
  FileClose $R8
  nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -File "$PLUGINSDIR\fcs_path.ps1"'
FunctionEnd

Function un.RemoveInstallDirFromUserPath
  ; Remove $INSTDIR from the current user PATH, leaving other entries intact.
  FileOpen $R8 "$PLUGINSDIR\fcs_path.ps1" w
  FileWrite $R8 '$$scope = "User"$\r$\n'
  FileWrite $R8 '$$dir   = "$INSTDIR"$\r$\n'
  FileWrite $R8 '$$cur   = [Environment]::GetEnvironmentVariable("Path", $$scope)$\r$\n'
  FileWrite $R8 '$$parts = if ($$cur) { $$cur.Split(";") | Where-Object { $$_.Trim() -ne "" -and $$_ -ne $$dir } } else { @() }$\r$\n'
  FileWrite $R8 '[Environment]::SetEnvironmentVariable("Path", ($$parts -join ";"), $$scope)$\r$\n'
  FileClose $R8
  nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -File "$PLUGINSDIR\fcs_path.ps1"'
FunctionEnd

Function un.RemoveInstallDirFromSystemPath
  ; Remove $INSTDIR from the system (all-users) PATH, leaving other entries intact.
  FileOpen $R8 "$PLUGINSDIR\fcs_path.ps1" w
  FileWrite $R8 '$$scope = "Machine"$\r$\n'
  FileWrite $R8 '$$dir   = "$INSTDIR"$\r$\n'
  FileWrite $R8 '$$cur   = [Environment]::GetEnvironmentVariable("Path", $$scope)$\r$\n'
  FileWrite $R8 '$$parts = if ($$cur) { $$cur.Split(";") | Where-Object { $$_.Trim() -ne "" -and $$_ -ne $$dir } } else { @() }$\r$\n'
  FileWrite $R8 '[Environment]::SetEnvironmentVariable("Path", ($$parts -join ";"), $$scope)$\r$\n'
  FileClose $R8
  nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -File "$PLUGINSDIR\fcs_path.ps1"'
FunctionEnd

; ── Utility functions ─────────────────────────────────────────────────────────

Function BroadcastEnvironmentChange
  ; Notify running processes that the environment has changed (makes PATH
  ; visible in new terminals without requiring a logoff/reboot).
  System::Call 'USER32::SendMessageTimeout(p ${HWND_BROADCAST}, i ${WM_SETTINGCHANGE}, p 0, t "Environment", i 0, i 5000, *p .r0)'
FunctionEnd

Function un.BroadcastEnvironmentChange
  System::Call 'USER32::SendMessageTimeout(p ${HWND_BROADCAST}, i ${WM_SETTINGCHANGE}, p 0, t "Environment", i 0, i 5000, *p .r0)'
FunctionEnd

; ── Mutual exclusion: prevent user and system PATH from both being selected ───

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
