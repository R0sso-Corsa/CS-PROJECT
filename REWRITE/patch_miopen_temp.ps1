param(
    [int]$TimeoutSeconds = 300,
    [int]$PollIntervalSeconds = 1
)

Write-Host "Watching for HIPRTC comgr-* temp folders for up to $TimeoutSeconds seconds..."
$stop = (Get-Date).AddSeconds($TimeoutSeconds)
$patched = $false

while ((Get-Date) -lt $stop -and -not $patched) {
    $comgrDirs = Get-ChildItem -Path $env:TEMP -Filter 'comgr-*' -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    foreach ($d in $comgrDirs) {
        $incPath = Join-Path $d.FullName ('include\miopen_utility.hpp')
        if (Test-Path $incPath) {
            Write-Host "Found: $incPath -- attempting patch"

            $content = Get-Content -Raw -LiteralPath $incPath
            if ($content -match 'MIOPEN_UTILITY_PATCHED') {
                Write-Host "Already patched; skipping."
                $patched = $true
                break
            }

            # Prepend safe include guard to prefer <utility> when available
            $header = @'
// MIOPEN temp patch injected by patch_miopen_temp.ps1
#if defined(__has_include)
#  if __has_include(<utility>)
#    include <utility>
#    define MIOPEN_UTILITY_SKIP_FORWARD
#  endif
#endif

/* MIOPEN_UTILITY_PATCHED */
'@

            # Insert header at top
            $newContent = $header + $content

            # Attempt to wrap the two offending 'forward' template definitions
            $lines = $newContent -split "`n"
            $indices = @()
            for ($i=0; $i -lt $lines.Length; $i++) {
                if ($lines[$i] -match 'constexpr\s+T&&\s+forward\s*\(') { $indices += $i }
            }

            if ($indices.Count -ge 2) {
                $first = $indices[0]
                $second = $indices[1]
                # find the closing brace after second (search next 20 lines)
                $endIdx = -1
                for ($j=$second; $j -lt [Math]::Min($lines.Length, $second + 30); $j++) {
                    if ($lines[$j].Trim() -eq '}') { $endIdx = $j; break }
                }
                if ($endIdx -eq -1) { $endIdx = $second + 2 }

                $lines = $lines[0..($first-1)] + @('#ifndef MIOPEN_UTILITY_SKIP_FORWARD') + $lines[$first..$endIdx] + @('#endif // MIOPEN_UTILITY_SKIP_FORWARD') + $lines[($endIdx+1)..($lines.Length-1)]
                $final = $lines -join "`n"
            } else {
                # Fallback: only prepend header (will include <utility> and define skip macro)
                $final = $newContent
            }

            # Backup original
            $bak = "$incPath.bak.$((Get-Date).ToString('yyyyMMddHHmmss'))"
            Copy-Item -LiteralPath $incPath -Destination $bak -Force
            Write-Host "Backup created: $bak"

            # Write patched file
            Set-Content -LiteralPath $incPath -Value $final -Encoding UTF8
            Write-Host "Patched $incPath successfully."
            $patched = $true
            break
        }
    }

    if (-not $patched) { Start-Sleep -Seconds $PollIntervalSeconds }
}

if ($patched) { Write-Host "Patch applied. Now re-run your Python training (or continue)."; exit 0 }
else { Write-Host "Timeout waiting for comgr-* folder. Make sure you start the Python run in the same shell so the temp folder is created."; exit 2 }
