; Inno Setup script for GroundTruth — turns the PyInstaller one-folder build
; (dist\GroundTruth\) into a Windows installer (Setup.exe) with Start Menu +
; optional desktop shortcuts.
;
; Prereq: build the app first, then install Inno Setup 6 (https://jrsoftware.org/isdl.php).
;   1) pyinstaller build\GroundTruth.spec --noconfirm      -> dist\GroundTruth\
;   2) "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\GroundTruth.iss
;   -> installer\Output\GroundTruth-Setup.exe
;
; Override the source dir if you built elsewhere:
;   ISCC.exe /DSourceDir="C:\Users\You\GroundTruth-dist\GroundTruth" installer\GroundTruth.iss

#ifndef SourceDir
  #define SourceDir "..\dist\GroundTruth"
#endif
#define AppName "GroundTruth"
#define AppVersion "1.0.1"
#define AppPublisher "AxialForge"
#define AppExe "GroundTruth.exe"

[Setup]
AppId={{A7E3C1B2-4D5F-4A6B-9C8D-1E2F3A4B5C6D}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\{#AppExe}
OutputDir=Output
OutputBaseFilename={#AppName}-Setup
SetupIconFile=..\assets\groundtruth.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
