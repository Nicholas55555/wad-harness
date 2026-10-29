# Re-render architecture.svg to architecture.pdf (Overleaf) and architecture.png (2x, slides fallback) with Edge.
# Usage (PowerShell, from this folder):  .\render_architecture.ps1
# Edge splits arguments at spaces in paths, so it works in a space-free temp folder and copies the results back.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$tmp = Join-Path $env:TEMP "wad_arch_render"
New-Item -ItemType Directory -Force $tmp | Out-Null
Copy-Item (Join-Path $here "architecture.svg") $tmp -Force
[xml]$svg = Get-Content (Join-Path $tmp "architecture.svg") -Raw
$w = [int]$svg.svg.width; $h = [int]$svg.svg.height
$html = "<!doctype html><html><head><meta charset='utf-8'><style>@page{size:${w}px ${h}px;margin:0}" +
        "html,body{margin:0;padding:0;background:#fff}img{display:block;width:${w}px;height:${h}px}</style></head>" +
        "<body><img src='architecture.svg'></body></html>"
Set-Content -Path (Join-Path $tmp "page.html") -Value $html -Encoding utf8
$edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
$url = "file:///" + $tmp.Replace([string][char]92, '/') + "/page.html"
$common = @("--headless=new", "--disable-gpu", "--virtual-time-budget=3000", "--user-data-dir=$tmp\profile")
Start-Process -FilePath $edge -Wait -NoNewWindow -ArgumentList ($common + @("--hide-scrollbars", "--force-device-scale-factor=2",
    "--window-size=$w,$h", "--screenshot=$tmp\architecture.png", $url))
Start-Process -FilePath $edge -Wait -NoNewWindow -ArgumentList ($common + @("--no-pdf-header-footer",
    "--print-to-pdf=$tmp\architecture.pdf", $url))
Copy-Item "$tmp\architecture.png", "$tmp\architecture.pdf" $here -Force
Get-ChildItem (Join-Path $here "architecture.*") | Select-Object Name, Length, LastWriteTime
