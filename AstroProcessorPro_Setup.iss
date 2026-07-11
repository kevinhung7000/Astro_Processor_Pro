; ============================================================
; Astro Processor Pro - Inno Setup 安裝檔製作腳本
; ============================================================
; 使用方式：
;   1. 到 https://jrsoftware.org/isdl.php 下載並安裝 Inno Setup（免費）
;   2. 用 Inno Setup 打開這個 .iss 檔（雙擊即可）
;   3. 按上方工具列的「Compile」（或按 F9）
;   4. 編譯完成後，安裝檔會出現在本檔案同層的 Output 資料夾裡
;
; 注意：此檔案預設路徑是「這個 .iss 檔案放在跟 dist 資料夾同一層」
;   例如：
;     C:\Users\hungk\Documents\userdocument\圖檔\Astro_Processor_Pro\AstroProcessorPro_Setup.iss
;     C:\Users\hungk\Documents\userdocument\圖檔\Astro_Processor_Pro\dist\AstroProcessorPro\...
;   如果你把這個 .iss 檔放到別的地方，請修改下面 [Files] 區塊的來源路徑。
; ============================================================

#define MyAppName "Astro Processor Pro"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Your Name"
#define MyAppExeName "AstroProcessorPro.exe"
; 打包時的來源資料夾（PyInstaller --onedir 產出的整個資料夾）
#define MySourceDir "dist\AstroProcessorPro"

[Setup]
; 每個 App 都要有一個獨一無二的 GUID，這裡先隨機生成好，不用改
AppId={{8F3A2C1D-9B4E-4A2F-A6C7-1D5E8F2B3C90}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
; 安裝檔輸出的資料夾與檔名
OutputDir=Output
OutputBaseFilename=AstroProcessorPro_Setup
; 因為含 torch，體積會很大，用 lzma2 壓縮比較省空間（但編譯較慢）
Compression=lzma2
SolidCompression=yes
; 64 位元系統安裝在 64 位元 Program Files
ArchitecturesInstallIn64BitMode=x64compatible
; 安裝程式本身的圖示（可省略，沒有圖示就拿掉這行）
; SetupIconFile=icon.ico
WizardStyle=modern
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "建立桌面捷徑"; GroupDescription: "額外圖示:"; Flags: unchecked

[Files]
; 把整個 dist\AstroProcessorPro 資料夾（含 _internal）全部複製進安裝目錄
Source: "{#MySourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\解除安裝 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; 安裝完成後可選擇立即開啟程式
Filename: "{app}\{#MyAppExeName}"; Description: "立即執行 {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 解除安裝時把整個資料夾清乾淨（包含使用者執行期間可能產生的暫存檔）
Type: filesandordirs; Name: "{app}"
