param(
    [string]$RootDir = (Get-Location).Path,
    [string]$OutputDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $RootDir "backend\data\benchmark_raw\format_coverage\sources"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Add-Type -AssemblyName System.Drawing

$fontCandidates = @(
    "C:\Windows\Fonts\NotoSansSC-VF.ttf",
    "C:\Windows\Fonts\msyh.ttc",
    "C:\Windows\Fonts\simhei.ttf",
    "C:\Windows\Fonts\simsun.ttc",
    "C:\Windows\Fonts\Deng.ttf"
)
$fontPath = $fontCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $fontPath) {
    throw "No Chinese-capable font found. Checked: $($fontCandidates -join ', ')"
}

$fontCollection = [System.Drawing.Text.PrivateFontCollection]::new()
$fontCollection.AddFontFile($fontPath)
$fontFamily = $fontCollection.Families[0]

$quote = "数据处理者向境外提供数据，应当履行数据安全保护义务，采取技术措施和其他必要措施。"
$title = "中文企业文档图片 OCR 样本"
$source = "来源：促进和规范数据跨境流动规定"

function New-FormatCoverageImage {
    param(
        [string]$Path,
        [System.Drawing.Imaging.ImageFormat]$Format
    )

    $bitmap = [System.Drawing.Bitmap]::new(1600, 720)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.Clear([System.Drawing.Color]::White)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

    $titleFont = [System.Drawing.Font]::new($fontFamily, 42, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
    $bodyFont = [System.Drawing.Font]::new($fontFamily, 46, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
    $smallFont = [System.Drawing.Font]::new($fontFamily, 28, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
    $black = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(20, 20, 20))
    $muted = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(80, 80, 80))
    $linePen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(210, 210, 210), 3)

    $graphics.DrawString($title, $titleFont, $black, [System.Drawing.RectangleF]::new(90, 70, 1420, 80))
    $graphics.DrawLine($linePen, 90, 160, 1510, 160)
    $graphics.DrawString($quote, $bodyFont, $black, [System.Drawing.RectangleF]::new(120, 230, 1360, 260))
    $graphics.DrawString($source, $smallFont, $muted, [System.Drawing.RectangleF]::new(120, 570, 1360, 70))

    $bitmap.Save($Path, $Format)

    $linePen.Dispose()
    $black.Dispose()
    $muted.Dispose()
    $titleFont.Dispose()
    $bodyFont.Dispose()
    $smallFont.Dispose()
    $graphics.Dispose()
    $bitmap.Dispose()
}

New-FormatCoverageImage -Path (Join-Path $OutputDir "data_export_security_assessment_ocr.png") -Format ([System.Drawing.Imaging.ImageFormat]::Png)
New-FormatCoverageImage -Path (Join-Path $OutputDir "data_export_security_assessment_ocr.jpg") -Format ([System.Drawing.Imaging.ImageFormat]::Jpeg)
New-FormatCoverageImage -Path (Join-Path $OutputDir "data_export_security_assessment_ocr.jpeg") -Format ([System.Drawing.Imaging.ImageFormat]::Jpeg)

Write-Output "Wrote image fixtures to $OutputDir"
