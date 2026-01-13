; TuroArnis Installer Script for Inno Setup
; Creates a professional Windows installer

#define MyAppName "TuroArnis"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Your Name/Organization"
#define MyAppURL "https://yourwebsite.com"
#define MyAppExeName "TuroArnis.exe"

[Setup]
; Basic app information
AppId={{YOUR-UNIQUE-APP-ID-HERE}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; Output configuration
OutputDir=installer_output
OutputBaseFilename=TuroArnis_Setup_v{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; User privileges
PrivilegesRequired=admin
; Icon (optional - uncomment if you have icon)
; SetupIconFile=icon.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main executable
Source: "dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
; Additional files (uncomment if needed)
; Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
; Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[Code]
// Custom message during installation
function InitializeSetup(): Boolean;
begin
  MsgBox('Welcome to TuroArnis Arnis Form Correction System Setup!' + #13#10 + #13#10 + 
         'This will install the application on your computer.' + #13#10 +
         'Please ensure you have at least 1GB of free disk space.', 
         mbInformation, MB_OK);
  Result := True;
end;
