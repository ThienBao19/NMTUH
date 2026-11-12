# Cập nhật mục lục bằng Word, xuất PDF và in số trang.
param([string]$Path)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Path)
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Repaginate()
    foreach ($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() }
    $doc.Save()
    $pdf = [System.IO.Path]::ChangeExtension($Path, '.pdf')
    if (Test-Path -LiteralPath $pdf) { Remove-Item -LiteralPath $pdf -Confirm:$false }
    $doc.ExportAsFixedFormat($pdf, 17)
    "pages: " + $doc.ComputeStatistics(2)
    "words: " + $doc.ComputeStatistics(0)
    $doc.Close(0)
} finally {
    $word.Quit()
}
